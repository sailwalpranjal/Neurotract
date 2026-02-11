"""
Preprocessing Pipeline Orchestrator

Coordinates all preprocessing steps for diffusion MRI data:
1. Gradient table validation and correction
2. Motion and eddy current correction
3. Brain extraction
4. Bias field correction
5. Quality control and validation

Manages data flow, intermediate outputs, and comprehensive logging.
Generates detailed QC reports and saves decision logs.
"""

import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import nibabel as nib

from .gradient_correction import GradientCorrector, GradientQualityMetrics
from .motion_correction import MotionCorrector, MotionMetrics, save_motion_corrected_data
from .brain_extraction import BrainExtractor, MaskQualityMetrics, save_brain_mask, apply_mask
from .bias_correction import BiasFieldCorrector, BiasCorrectionMetrics, save_bias_corrected_data

from ..data.data_loader import DataLoader, DiffusionData
from ..utils.logger import get_logger, log_decision
from ..utils.memory_manager import get_memory_manager

logger = get_logger(__name__)


class PreprocessingError(Exception):
    """Exception raised for preprocessing failures"""
    pass


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for diffusion MRI data

    Orchestrates all preprocessing steps with robust error handling,
    quality control, and comprehensive logging.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        skip_motion_correction: bool = False,
        skip_brain_extraction: bool = False,
        skip_bias_correction: bool = False,
        motion_registration: str = "affine",
        brain_method: str = "median_otsu",
        bias_method: str = "auto",
        save_intermediate: bool = True,
        save_qc_reports: bool = True,
    ):
        """
        Initialize preprocessing pipeline

        Parameters
        ----------
        output_dir : str or Path
            Directory for output files
        skip_motion_correction : bool
            Skip motion correction step (default: False)
        skip_brain_extraction : bool
            Skip brain extraction step (default: False)
        skip_bias_correction : bool
            Skip bias correction step (default: False)
        motion_registration : str
            Registration type for motion correction: 'rigid' or 'affine'
        brain_method : str
            Brain extraction method: 'median_otsu' or 'threshold'
        bias_method : str
            Bias correction method: 'auto', 'n4', 'polynomial', or 'histogram'
        save_intermediate : bool
            Save intermediate outputs (default: True)
        save_qc_reports : bool
            Generate and save QC reports (default: True)
        """
        self.output_dir = Path(output_dir)
        self.skip_motion_correction = skip_motion_correction
        self.skip_brain_extraction = skip_brain_extraction
        self.skip_bias_correction = skip_bias_correction
        self.save_intermediate = save_intermediate
        self.save_qc_reports = save_qc_reports

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir = self.output_dir / "intermediate"
        self.qc_dir = self.output_dir / "qc"

        if self.save_intermediate:
            self.intermediate_dir.mkdir(exist_ok=True)
        if self.save_qc_reports:
            self.qc_dir.mkdir(exist_ok=True)

        # Initialize components
        self.gradient_corrector = GradientCorrector(normalize_bvecs=True)

        self.motion_corrector = MotionCorrector(
            registration_type=motion_registration,
            metric="MI",
            level_iters=[10000, 1000, 100],
        )

        self.brain_extractor = BrainExtractor(
            method=brain_method,
            median_radius=4,
            num_pass=4,
            dilate=1,
        )

        self.bias_corrector = BiasFieldCorrector(
            method=bias_method,
            convergence_threshold=0.001,
            max_iterations=50,
        )

        # Initialize utilities
        self.data_loader = DataLoader(use_mmap=True, validate=True)
        self.memory_manager = get_memory_manager()

        # Pipeline state
        self.pipeline_start_time = None
        self.pipeline_metrics = {}
        self.checksums = {}

        logger.info(
            f"PreprocessingPipeline initialized: "
            f"output_dir={output_dir}, "
            f"motion={not skip_motion_correction}, "
            f"brain={not skip_brain_extraction}, "
            f"bias={not skip_bias_correction}"
        )

    def run(
        self,
        dwi_path: Union[str, Path],
        bval_path: Optional[Union[str, Path]] = None,
        bvec_path: Optional[Union[str, Path]] = None,
        output_prefix: str = "preprocessed",
    ) -> Dict[str, Path]:
        """
        Run complete preprocessing pipeline

        Parameters
        ----------
        dwi_path : str or Path
            Path to input DWI NIfTI file
        bval_path : str or Path, optional
            Path to b-values file (auto-detected if None)
        bvec_path : str or Path, optional
            Path to b-vectors file (auto-detected if None)
        output_prefix : str
            Prefix for output files (default: 'preprocessed')

        Returns
        -------
        outputs : dict
            Dictionary containing paths to output files:
            - 'dwi': preprocessed DWI data
            - 'bval': corrected b-values
            - 'bvec': corrected b-vectors
            - 'mask': brain mask
            - 'qc_report': quality control report
        """
        self.pipeline_start_time = datetime.now()
        logger.info("=" * 70)
        logger.info("Starting Preprocessing Pipeline")
        logger.info("=" * 70)

        try:
            # Step 1: Load data
            logger.info("\n[Step 1/5] Loading diffusion data...")
            dwi_data = self._load_data(dwi_path, bval_path, bvec_path)

            # Step 2: Gradient table correction
            logger.info("\n[Step 2/5] Validating and correcting gradient table...")
            corrected_bvals, corrected_bvecs, gradient_metrics = self._correct_gradients(
                dwi_data.bvals, dwi_data.bvecs
            )

            # Step 3: Motion correction
            if not self.skip_motion_correction:
                logger.info("\n[Step 3/5] Performing motion and eddy current correction...")
                corrected_data, corrected_bvecs, motion_metrics = self._correct_motion(
                    dwi_data.data, corrected_bvals, corrected_bvecs, dwi_data.affine
                )
            else:
                logger.info("\n[Step 3/5] Skipping motion correction...")
                corrected_data = dwi_data.data
                motion_metrics = None

            # Step 4: Brain extraction
            if not self.skip_brain_extraction:
                logger.info("\n[Step 4/5] Extracting brain mask...")
                masked_data, mask, mask_metrics = self._extract_brain(
                    corrected_data, dwi_data.affine
                )
            else:
                logger.info("\n[Step 4/5] Skipping brain extraction...")
                masked_data = corrected_data
                mask = np.ones(corrected_data.shape[:3], dtype=np.uint8)
                mask_metrics = None

            # Step 5: Bias field correction
            if not self.skip_bias_correction:
                logger.info("\n[Step 5/5] Performing bias field correction...")
                final_data, bias_field, bias_metrics = self._correct_bias(
                    masked_data, dwi_data.affine, mask
                )
            else:
                logger.info("\n[Step 5/5] Skipping bias field correction...")
                final_data = masked_data
                bias_field = None
                bias_metrics = None

            # Save final outputs
            logger.info("\nSaving preprocessed data...")
            outputs = self._save_outputs(
                final_data,
                corrected_bvals,
                corrected_bvecs,
                mask,
                bias_field,
                dwi_data.affine,
                dwi_data.header,
                output_prefix,
            )

            # Generate QC report
            if self.save_qc_reports:
                logger.info("Generating QC report...")
                qc_report = self._generate_qc_report(
                    dwi_data,
                    gradient_metrics,
                    motion_metrics,
                    mask_metrics,
                    bias_metrics,
                    output_prefix,
                )
                outputs['qc_report'] = qc_report

            # Log pipeline completion
            self._log_pipeline_completion(outputs)

            logger.info("=" * 70)
            logger.info("Preprocessing Pipeline Completed Successfully")
            logger.info("=" * 70)

            return outputs

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise PreprocessingError(f"Preprocessing pipeline failed: {e}")

    def _load_data(
        self,
        dwi_path: Union[str, Path],
        bval_path: Optional[Union[str, Path]],
        bvec_path: Optional[Union[str, Path]],
    ) -> DiffusionData:
        """Load diffusion data with validation"""
        try:
            dwi_data = self.data_loader.load_diffusion(
                dwi_path, bval_path, bvec_path, auto_detect=True
            )

            logger.info(
                f"Loaded: {dwi_data.shape}, "
                f"{dwi_data.num_volumes} volumes, "
                f"{dwi_data.num_shells} shells: {dwi_data.get_shells()}"
            )

            # Compute and store checksum
            self.checksums['input_dwi'] = self._compute_checksum(dwi_data.data)

            return dwi_data

        except Exception as e:
            raise PreprocessingError(f"Failed to load data: {e}")

    def _correct_gradients(
        self,
        bvals: np.ndarray,
        bvecs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, GradientQualityMetrics]:
        """Validate and correct gradient table"""
        output_dir = self.qc_dir / "gradients" if self.save_qc_reports else None

        corrected_bvals, corrected_bvecs, metrics = self.gradient_corrector.correct(
            bvals, bvecs, output_dir=output_dir
        )

        self.pipeline_metrics['gradient'] = metrics.metrics

        return corrected_bvals, corrected_bvecs, metrics

    def _correct_motion(
        self,
        data: np.ndarray,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        affine: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, MotionMetrics]:
        """Perform motion and eddy current correction"""
        output_dir = self.qc_dir / "motion" if self.save_qc_reports else None

        corrected_data, corrected_bvecs, metrics = self.motion_corrector.correct(
            data, bvals, bvecs, affine,
            mask=None,
            reference_b0=True,
            output_dir=output_dir,
        )

        self.pipeline_metrics['motion'] = metrics.compute_summary_stats()

        # Save intermediate if requested
        if self.save_intermediate:
            self._save_intermediate_dwi(
                corrected_data, corrected_bvecs, bvals, affine,
                "motion_corrected"
            )

        # Compute checksum
        self.checksums['motion_corrected'] = self._compute_checksum(corrected_data)

        return corrected_data, corrected_bvecs, metrics

    def _extract_brain(
        self,
        data: np.ndarray,
        affine: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, MaskQualityMetrics]:
        """Extract brain mask"""
        output_dir = self.qc_dir / "brain_mask" if self.save_qc_reports else None

        masked_data, mask, metrics = self.brain_extractor.extract(
            data, affine,
            vol_idx=None,  # Use mean volume
            output_dir=output_dir,
        )

        self.pipeline_metrics['brain_mask'] = metrics.metrics

        # Save intermediate if requested
        if self.save_intermediate:
            mask_path = self.intermediate_dir / "brain_mask.nii.gz"
            mask_img = nib.Nifti1Image(mask, affine)
            nib.save(mask_img, mask_path)
            logger.info(f"Saved intermediate brain mask to {mask_path}")

        return masked_data, mask, metrics

    def _correct_bias(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], BiasCorrectionMetrics]:
        """Perform bias field correction"""
        output_dir = self.qc_dir / "bias_correction" if self.save_qc_reports else None

        corrected_data, bias_field, metrics = self.bias_corrector.correct(
            data, affine,
            mask=mask,
            output_dir=output_dir,
        )

        self.pipeline_metrics['bias_correction'] = metrics.metrics

        # Save intermediate if requested
        if self.save_intermediate and bias_field is not None:
            bias_path = self.intermediate_dir / "bias_field.nii.gz"
            bias_img = nib.Nifti1Image(bias_field, affine)
            nib.save(bias_img, bias_path)
            logger.info(f"Saved intermediate bias field to {bias_path}")

        # Compute checksum
        self.checksums['bias_corrected'] = self._compute_checksum(corrected_data)

        return corrected_data, bias_field, metrics

    def _save_outputs(
        self,
        data: np.ndarray,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        mask: np.ndarray,
        bias_field: Optional[np.ndarray],
        affine: np.ndarray,
        header: nib.Nifti1Header,
        prefix: str,
    ) -> Dict[str, Path]:
        """Save final preprocessed outputs"""
        outputs = {}

        # Save DWI data
        dwi_path = self.output_dir / f"{prefix}_dwi.nii.gz"
        dwi_img = nib.Nifti1Image(data, affine, header)
        dwi_img.header['descrip'] = b'Preprocessed by NeuroTract'
        nib.save(dwi_img, dwi_path)
        outputs['dwi'] = dwi_path
        logger.info(f"Saved preprocessed DWI to {dwi_path}")

        # Save bvals
        bval_path = self.output_dir / f"{prefix}_dwi.bval"
        np.savetxt(bval_path, bvals, fmt='%d')
        outputs['bval'] = bval_path
        logger.info(f"Saved b-values to {bval_path}")

        # Save bvecs
        bvec_path = self.output_dir / f"{prefix}_dwi.bvec"
        np.savetxt(bvec_path, bvecs, fmt='%.6f')
        outputs['bvec'] = bvec_path
        logger.info(f"Saved b-vectors to {bvec_path}")

        # Save brain mask
        mask_path = self.output_dir / f"{prefix}_brain_mask.nii.gz"
        mask_img = nib.Nifti1Image(mask, affine)
        mask_img.header['descrip'] = b'Brain mask by NeuroTract'
        nib.save(mask_img, mask_path)
        outputs['mask'] = mask_path
        logger.info(f"Saved brain mask to {mask_path}")

        # Save bias field if available
        if bias_field is not None:
            bias_path = self.output_dir / f"{prefix}_bias_field.nii.gz"
            bias_img = nib.Nifti1Image(bias_field, affine)
            nib.save(bias_img, bias_path)
            outputs['bias_field'] = bias_path
            logger.info(f"Saved bias field to {bias_path}")

        # Save checksums
        checksum_path = self.output_dir / f"{prefix}_checksums.json"
        with open(checksum_path, 'w') as f:
            json.dump(self.checksums, f, indent=2)
        outputs['checksums'] = checksum_path

        return outputs

    def _save_intermediate_dwi(
        self,
        data: np.ndarray,
        bvecs: np.ndarray,
        bvals: np.ndarray,
        affine: np.ndarray,
        name: str,
    ):
        """Save intermediate DWI data"""
        # Save DWI
        dwi_path = self.intermediate_dir / f"{name}.nii.gz"
        dwi_img = nib.Nifti1Image(data, affine)
        nib.save(dwi_img, dwi_path)

        # Save gradients
        bval_path = self.intermediate_dir / f"{name}.bval"
        np.savetxt(bval_path, bvals, fmt='%d')

        bvec_path = self.intermediate_dir / f"{name}.bvec"
        np.savetxt(bvec_path, bvecs, fmt='%.6f')

        logger.info(f"Saved intermediate data: {name}")

    def _generate_qc_report(
        self,
        dwi_data: DiffusionData,
        gradient_metrics: GradientQualityMetrics,
        motion_metrics: Optional[MotionMetrics],
        mask_metrics: Optional[MaskQualityMetrics],
        bias_metrics: Optional[BiasCorrectionMetrics],
        prefix: str,
    ) -> Path:
        """Generate comprehensive QC report"""
        report_path = self.qc_dir / f"{prefix}_qc_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("NEUROTRACT PREPROCESSING QC REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {dwi_data.file_path}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")

            # Input data summary
            f.write("-" * 70 + "\n")
            f.write("INPUT DATA\n")
            f.write("-" * 70 + "\n")
            f.write(f"Shape: {dwi_data.shape}\n")
            f.write(f"Voxel size: {dwi_data.voxel_size} mm\n")
            f.write(f"Number of volumes: {dwi_data.num_volumes}\n")
            f.write(f"Number of shells: {dwi_data.num_shells}\n")
            f.write(f"Shell b-values: {dwi_data.get_shells()}\n\n")

            # Gradient table QC
            f.write("-" * 70 + "\n")
            f.write("GRADIENT TABLE QC\n")
            f.write("-" * 70 + "\n")
            for key, value in gradient_metrics.metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            is_valid, msg = gradient_metrics.is_valid()
            f.write(f"\nValidation: {'PASS' if is_valid else 'FAIL'}\n")
            f.write(f"Message: {msg}\n\n")

            # Motion correction QC
            if motion_metrics is not None:
                f.write("-" * 70 + "\n")
                f.write("MOTION CORRECTION QC\n")
                f.write("-" * 70 + "\n")
                summary = motion_metrics.compute_summary_stats()
                for key, value in summary.items():
                    f.write(f"{key}: {value:.4f}\n")
                f.write("\n")

            # Brain mask QC
            if mask_metrics is not None:
                f.write("-" * 70 + "\n")
                f.write("BRAIN MASK QC\n")
                f.write("-" * 70 + "\n")
                for key, value in mask_metrics.metrics.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                is_valid, msg = mask_metrics.is_valid()
                f.write(f"\nValidation: {'PASS' if is_valid else 'FAIL'}\n")
                f.write(f"Message: {msg}\n\n")

            # Bias correction QC
            if bias_metrics is not None:
                f.write("-" * 70 + "\n")
                f.write("BIAS CORRECTION QC\n")
                f.write("-" * 70 + "\n")
                for key, value in bias_metrics.metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
                f.write("\n")

            # Pipeline summary
            f.write("-" * 70 + "\n")
            f.write("PIPELINE SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"Motion correction: {'Applied' if not self.skip_motion_correction else 'Skipped'}\n")
            f.write(f"Brain extraction: {'Applied' if not self.skip_brain_extraction else 'Skipped'}\n")
            f.write(f"Bias correction: {'Applied' if not self.skip_bias_correction else 'Skipped'}\n")

            elapsed = datetime.now() - self.pipeline_start_time
            f.write(f"\nTotal processing time: {elapsed}\n")

            f.write("\n" + "=" * 70 + "\n")

        logger.info(f"QC report saved to {report_path}")
        return report_path

    def _compute_checksum(self, data: np.ndarray) -> str:
        """Compute MD5 checksum for data integrity"""
        # Use a small sample for efficiency
        sample = data.ravel()[::1000]  # Every 1000th element
        checksum = hashlib.md5(sample.tobytes()).hexdigest()
        return checksum

    def _log_pipeline_completion(self, outputs: Dict[str, Path]):
        """Log pipeline completion with all decisions"""
        log_decision(
            decision_id="preprocessing_pipeline_complete",
            component="preprocessing.pipeline",
            decision="Preprocessing pipeline completed successfully",
            rationale=(
                f"All steps executed: "
                f"motion={'Yes' if not self.skip_motion_correction else 'No'}, "
                f"brain={'Yes' if not self.skip_brain_extraction else 'No'}, "
                f"bias={'Yes' if not self.skip_bias_correction else 'No'}"
            ),
            parameters={
                'output_dir': str(self.output_dir),
                'output_files': {k: str(v) for k, v in outputs.items()},
                'pipeline_metrics': self.pipeline_metrics,
                'checksums': self.checksums,
                'processing_time': str(datetime.now() - self.pipeline_start_time),
            },
        )
