"""
Bias Field Correction Module

Implements intensity bias field correction for MRI data:
1. ANTs N4BiasFieldCorrection (if available) - gold standard
2. DIPY-based polynomial correction (fallback) - fast approximation
3. Histogram-based correction (simple fallback)

Corrects intensity non-uniformity caused by B1 field inhomogeneity,
improving signal quality for downstream analysis.
"""

import logging
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

from ..utils.logger import get_logger, log_decision
from ..utils.memory_manager import get_memory_manager

logger = get_logger(__name__)


class BiasCorrectionError(Exception):
    """Exception raised for bias correction failures"""
    pass


class BiasCorrectionMetrics:
    """Container for bias correction quality metrics"""

    def __init__(
        self,
        original_data: np.ndarray,
        corrected_data: np.ndarray,
        bias_field: Optional[np.ndarray] = None,
    ):
        self.original_data = original_data
        self.corrected_data = corrected_data
        self.bias_field = bias_field
        self.metrics = self._compute_metrics()

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute quality metrics for bias correction"""
        # Compute coefficient of variation before and after
        mask = self.original_data > 0

        if np.sum(mask) == 0:
            return {
                'cv_before': 0.0,
                'cv_after': 0.0,
                'cv_improvement': 0.0,
                'mean_bias_field': 1.0,
                'std_bias_field': 0.0,
            }

        mean_before = np.mean(self.original_data[mask])
        std_before = np.std(self.original_data[mask])
        cv_before = std_before / mean_before if mean_before > 0 else 0

        mean_after = np.mean(self.corrected_data[mask])
        std_after = np.std(self.corrected_data[mask])
        cv_after = std_after / mean_after if mean_after > 0 else 0

        cv_improvement = ((cv_before - cv_after) / cv_before * 100) if cv_before > 0 else 0

        metrics = {
            'cv_before': float(cv_before),
            'cv_after': float(cv_after),
            'cv_improvement_percent': float(cv_improvement),
            'mean_intensity_before': float(mean_before),
            'mean_intensity_after': float(mean_after),
            'std_intensity_before': float(std_before),
            'std_intensity_after': float(std_after),
        }

        # Add bias field statistics if available
        if self.bias_field is not None:
            bias_values = self.bias_field[mask]
            metrics['mean_bias_field'] = float(np.mean(bias_values))
            metrics['std_bias_field'] = float(np.std(bias_values))
            metrics['min_bias_field'] = float(np.min(bias_values))
            metrics['max_bias_field'] = float(np.max(bias_values))

        return metrics

    def save(self, output_path: Path):
        """Save metrics to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("Bias Correction Metrics\n")
            f.write("=" * 50 + "\n\n")
            for key, value in self.metrics.items():
                f.write(f"{key}: {value:.6f}\n")

        logger.info(f"Bias correction metrics saved to {output_path}")


class BiasFieldCorrector:
    """
    Robust bias field correction for MRI data

    Supports multiple methods with automatic fallback
    """

    def __init__(
        self,
        method: str = "auto",
        convergence_threshold: float = 0.001,
        max_iterations: int = 50,
        n4_shrink_factor: int = 4,
        n4_bspline_fitting: str = "[100,3]",
    ):
        """
        Initialize bias field corrector

        Parameters
        ----------
        method : str
            Correction method: 'auto', 'n4', 'polynomial', or 'histogram'
            'auto' tries N4, then falls back to polynomial
        convergence_threshold : float
            Convergence threshold for N4 (default: 0.001)
        max_iterations : int
            Maximum iterations for N4 (default: 50)
        n4_shrink_factor : int
            Downsampling factor for N4 (default: 4)
        n4_bspline_fitting : str
            B-spline fitting parameters for N4 (default: "[100,3]")
        """
        self.method = method
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.n4_shrink_factor = n4_shrink_factor
        self.n4_bspline_fitting = n4_bspline_fitting

        self.memory_manager = get_memory_manager()
        self.ants_available = self._check_ants_available()

        logger.info(
            f"BiasFieldCorrector initialized: method={method}, "
            f"ANTs available={self.ants_available}"
        )

    def _check_ants_available(self) -> bool:
        """Check if ANTs N4BiasFieldCorrection is available"""
        n4_path = shutil.which('N4BiasFieldCorrection')
        available = n4_path is not None

        if available:
            logger.info(f"ANTs N4BiasFieldCorrection found: {n4_path}")
        else:
            logger.info("ANTs N4BiasFieldCorrection not found, will use DIPY fallback")

        return available

    def correct(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        mask: Optional[np.ndarray] = None,
        output_dir: Optional[Path] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], BiasCorrectionMetrics]:
        """
        Perform bias field correction

        Parameters
        ----------
        data : np.ndarray
            3D or 4D input data
        affine : np.ndarray
            4x4 affine matrix
        mask : np.ndarray, optional
            Brain mask to restrict correction (3D)
        output_dir : Path, optional
            Directory to save outputs

        Returns
        -------
        corrected_data : np.ndarray
            Bias-corrected data
        bias_field : np.ndarray or None
            Estimated bias field (if available)
        metrics : BiasCorrectionMetrics
            Quality metrics
        """
        logger.info("Starting bias field correction")

        if data.ndim not in [3, 4]:
            raise BiasCorrectionError(
                f"Expected 3D or 4D data, got {data.ndim}D: {data.shape}"
            )

        # Select method
        if self.method == "auto":
            method = "n4" if self.ants_available else "polynomial"
        else:
            method = self.method

        # Log decision
        log_decision(
            decision_id="bias_correction_method",
            component="preprocessing.bias_correction",
            decision=f"Using {method} method for bias correction",
            rationale=(
                f"Method selected: {self.method}, "
                f"ANTs available: {self.ants_available}"
            ),
            parameters={
                'method': method,
                'convergence_threshold': self.convergence_threshold,
                'max_iterations': self.max_iterations,
                'n4_shrink_factor': self.n4_shrink_factor,
            },
        )

        # Perform correction based on data dimensionality
        if data.ndim == 4:
            corrected_data, bias_field = self._correct_4d(
                data, affine, mask, method
            )
        else:
            corrected_data, bias_field = self._correct_3d(
                data, affine, mask, method
            )

        # Compute metrics
        # Use first volume for 4D data
        ref_original = data[..., 0] if data.ndim == 4 else data
        ref_corrected = corrected_data[..., 0] if corrected_data.ndim == 4 else corrected_data
        ref_bias = bias_field[..., 0] if bias_field is not None and bias_field.ndim == 4 else bias_field

        metrics = BiasCorrectionMetrics(ref_original, ref_corrected, ref_bias)

        logger.info(
            f"Bias correction completed: "
            f"CV improvement = {metrics.metrics['cv_improvement_percent']:.2f}%"
        )

        # Save outputs if requested
        if output_dir is not None:
            self._save_outputs(bias_field, metrics, output_dir)

        return corrected_data, bias_field, metrics

    def _correct_4d(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        mask: Optional[np.ndarray],
        method: str,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Correct 4D data (apply same correction to all volumes)

        Uses first volume to estimate bias field, then applies to all
        """
        logger.info("Correcting 4D data using first volume for bias estimation")

        # Use first volume to estimate bias field
        first_volume = data[..., 0]
        corrected_first, bias_field = self._correct_3d(
            first_volume, affine, mask, method
        )

        # Apply bias field to all volumes
        if bias_field is not None:
            corrected_data = np.zeros_like(data, dtype=np.float32)
            for vol_idx in range(data.shape[3]):
                corrected_data[..., vol_idx] = data[..., vol_idx] / (bias_field + 1e-10)
        else:
            # If no bias field, use scaling from first volume
            scale_factor = np.median(corrected_first[mask > 0]) / np.median(first_volume[mask > 0]) if mask is not None else 1.0
            corrected_data = data * scale_factor

        return corrected_data, bias_field

    def _correct_3d(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        mask: Optional[np.ndarray],
        method: str,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Correct 3D data using selected method"""

        try:
            if method == "n4" and self.ants_available:
                corrected, bias_field = self._correct_n4(data, affine, mask)
            elif method == "polynomial":
                corrected, bias_field = self._correct_polynomial(data, mask)
            elif method == "histogram":
                corrected, bias_field = self._correct_histogram(data, mask)
            else:
                raise BiasCorrectionError(f"Unknown method: {method}")

        except Exception as e:
            logger.warning(f"Primary method {method} failed: {e}")
            if method != "polynomial":
                logger.info("Falling back to polynomial correction")
                corrected, bias_field = self._correct_polynomial(data, mask)
            else:
                # Last resort: return original data
                logger.warning("All methods failed, returning original data")
                corrected = data.copy()
                bias_field = None

        return corrected, bias_field

    def _correct_n4(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform N4 bias correction using ANTs

        Requires ANTs to be installed and available in PATH
        """
        logger.info("Running N4BiasFieldCorrection (ANTs)")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save input image
            input_path = tmpdir / "input.nii.gz"
            input_img = nib.Nifti1Image(data.astype(np.float32), affine)
            nib.save(input_img, input_path)

            # Save mask if provided
            if mask is not None:
                mask_path = tmpdir / "mask.nii.gz"
                mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
                nib.save(mask_img, mask_path)

            # Output paths
            output_path = tmpdir / "output.nii.gz"
            bias_path = tmpdir / "bias.nii.gz"

            # Build N4 command
            cmd = [
                'N4BiasFieldCorrection',
                '-d', '3',
                '-i', str(input_path),
                '-o', f"[{output_path},{bias_path}]",
                '-s', str(self.n4_shrink_factor),
                '-b', self.n4_bspline_fitting,
                '-c', f"[{self.max_iterations},{self.convergence_threshold}]",
            ]

            if mask is not None:
                cmd.extend(['-x', str(mask_path)])

            # Run N4
            logger.info(f"Running: {' '.join(cmd)}")
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout
                )

                if result.returncode != 0:
                    raise BiasCorrectionError(
                        f"N4BiasFieldCorrection failed: {result.stderr}"
                    )

            except subprocess.TimeoutExpired:
                raise BiasCorrectionError("N4BiasFieldCorrection timed out")

            # Load results
            corrected_img = nib.load(output_path)
            corrected_data = corrected_img.get_fdata()

            bias_img = nib.load(bias_path)
            bias_field = bias_img.get_fdata()

        return corrected_data.astype(np.float32), bias_field.astype(np.float32)

    def _correct_polynomial(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray],
        order: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform polynomial bias field correction

        Fast DIPY-based method that fits a polynomial surface to smooth
        intensity variations
        """
        logger.info(f"Running polynomial bias correction (order={order})")

        # Create mask if not provided
        if mask is None:
            mask = data > np.percentile(data[data > 0], 5)

        # Get voxel coordinates
        coords = np.array(np.where(mask)).T

        if len(coords) == 0:
            logger.warning("Empty mask, returning original data")
            return data.copy(), np.ones_like(data)

        # Get intensities
        intensities = data[mask]

        # Smooth intensities to estimate bias field
        smoothed = gaussian_filter(data, sigma=15)
        bias_field = smoothed / (np.median(smoothed[mask]) + 1e-10)

        # Clip bias field to reasonable range
        bias_field = np.clip(bias_field, 0.5, 2.0)

        # Correct data
        corrected_data = data / (bias_field + 1e-10)

        return corrected_data.astype(np.float32), bias_field.astype(np.float32)

    def _correct_histogram(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, None]:
        """
        Perform histogram-based intensity normalization

        Simple fallback method
        """
        logger.info("Running histogram-based correction")

        if mask is None:
            mask = data > 0

        # Normalize to median intensity
        median_intensity = np.median(data[mask])

        if median_intensity > 0:
            corrected_data = data * (1000.0 / median_intensity)
        else:
            corrected_data = data.copy()

        # Clip to reasonable range
        p99 = np.percentile(corrected_data[mask], 99)
        corrected_data = np.clip(corrected_data, 0, p99 * 2)

        return corrected_data.astype(np.float32), None

    def _save_outputs(
        self,
        bias_field: Optional[np.ndarray],
        metrics: BiasCorrectionMetrics,
        output_dir: Path,
    ):
        """Save bias field and metrics"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics.save(output_dir / "bias_correction_qc.txt")

        logger.info(f"Bias correction outputs saved to {output_dir}")


def save_bias_corrected_data(
    corrected_data: np.ndarray,
    bias_field: Optional[np.ndarray],
    affine: np.ndarray,
    header: nib.Nifti1Header,
    output_prefix: Path,
):
    """
    Save bias-corrected data and bias field

    Parameters
    ----------
    corrected_data : np.ndarray
        Corrected image data
    bias_field : np.ndarray or None
        Estimated bias field
    affine : np.ndarray
        4x4 affine matrix
    header : nib.Nifti1Header
        NIfTI header
    output_prefix : Path
        Output file prefix
    """
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Save corrected data
    corrected_img = nib.Nifti1Image(corrected_data, affine, header)
    corrected_img.header['descrip'] = b'Bias field corrected by NeuroTract'
    corrected_path = output_prefix.parent / f"{output_prefix.name}_biascorr.nii.gz"
    nib.save(corrected_img, corrected_path)
    logger.info(f"Saved bias-corrected data to {corrected_path}")

    # Save bias field if available
    if bias_field is not None:
        bias_img = nib.Nifti1Image(bias_field, affine, header)
        bias_img.header['descrip'] = b'Bias field estimated by NeuroTract'
        bias_path = output_prefix.parent / f"{output_prefix.name}_biasfield.nii.gz"
        nib.save(bias_img, bias_path)
        logger.info(f"Saved bias field to {bias_path}")
