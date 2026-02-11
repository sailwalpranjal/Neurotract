#!/usr/bin/env python
"""
Metadata Extractor Module for NeuroTract

Extracts and validates metadata from neuroimaging data.
Generates comprehensive scan manifests and quality reports.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings

import nibabel as nib
import numpy as np


logger = logging.getLogger(__name__)


class MetadataExtractionError(Exception):
    """Exception raised for metadata extraction errors"""
    pass


class QualityCheckError(Exception):
    """Exception raised for quality check failures"""
    pass


class MetadataExtractor:
    """
    Extract and validate metadata from neuroimaging data

    Features:
    - Extract voxel sizes, dimensions, orientation
    - Parse b-values and gradient tables
    - Extract scanner parameters (TR, TE, etc.)
    - Validate data quality
    - Generate per-scan manifest JSON
    """

    def __init__(self):
        """Initialize MetadataExtractor"""
        self.quality_checks = {
            'passed': [],
            'warnings': [],
            'failures': []
        }

    def extract_nifti_metadata(
        self,
        nifti_path: Union[str, Path],
        load_data: bool = False
    ) -> Dict:
        """
        Extract comprehensive metadata from NIfTI file

        Parameters
        ----------
        nifti_path : str or Path
            Path to NIfTI file
        load_data : bool
            Whether to load data for advanced checks (default: False)

        Returns
        -------
        dict
            Dictionary containing metadata
        """
        nifti_path = Path(nifti_path)

        if not nifti_path.exists():
            raise MetadataExtractionError(f"File not found: {nifti_path}")

        logger.info(f"Extracting metadata from {nifti_path}")

        try:
            img = nib.load(nifti_path)
            header = img.header
        except Exception as e:
            raise MetadataExtractionError(f"Failed to load NIfTI: {e}")

        metadata = {
            'file_path': str(nifti_path),
            'file_name': nifti_path.name,
            'file_size_mb': nifti_path.stat().st_size / (1024 ** 2),
            'extraction_timestamp': datetime.now().isoformat(),
        }

        # Basic image properties
        metadata.update(self._extract_basic_properties(header, img))

        # Spatial properties
        metadata.update(self._extract_spatial_properties(header, img))

        # Orientation
        metadata.update(self._extract_orientation(img))

        # Scanner parameters
        metadata.update(self._extract_scanner_parameters(header))

        # Data type information
        metadata.update(self._extract_datatype_info(header))

        # Optional: Load data for advanced checks
        if load_data:
            data = np.asarray(img.dataobj)
            metadata.update(self._extract_data_statistics(data))

        return metadata

    def extract_diffusion_metadata(
        self,
        nifti_path: Union[str, Path],
        bval_path: Union[str, Path],
        bvec_path: Union[str, Path],
        json_path: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Extract comprehensive metadata from diffusion MRI data

        Parameters
        ----------
        nifti_path : str or Path
            Path to diffusion NIfTI file
        bval_path : str or Path
            Path to bval file
        bvec_path : str or Path
            Path to bvec file
        json_path : str or Path, optional
            Path to JSON sidecar with additional metadata

        Returns
        -------
        dict
            Dictionary containing diffusion metadata
        """
        # Get basic NIfTI metadata
        metadata = self.extract_nifti_metadata(nifti_path, load_data=False)
        metadata['modality'] = 'dwi'

        # Load and analyze bvals
        bvals = np.loadtxt(bval_path)
        if bvals.ndim > 1:
            bvals = bvals.flatten()

        metadata['bvals'] = {
            'file_path': str(bval_path),
            'num_values': len(bvals),
            'unique_bvals': self._get_unique_bvals(bvals),
            'shells': self._identify_shells(bvals),
            'num_shells': len(self._identify_shells(bvals)),
            'max_bval': float(np.max(bvals)),
            'min_bval': float(np.min(bvals)),
        }

        # Count b0 volumes
        b0_count = np.sum(bvals < 50)
        metadata['bvals']['num_b0'] = int(b0_count)

        # Load and analyze bvecs
        bvecs = np.loadtxt(bvec_path)
        if bvecs.shape[0] != 3:
            bvecs = bvecs.T

        metadata['bvecs'] = {
            'file_path': str(bvec_path),
            'shape': list(bvecs.shape),
            'num_directions': bvecs.shape[1],
        }

        # Validate gradient vectors
        metadata['bvecs']['normalized'] = self._check_gradient_normalization(bvecs, bvals)

        # Load JSON sidecar if available
        if json_path and Path(json_path).exists():
            try:
                with open(json_path, 'r') as f:
                    json_metadata = json.load(f)
                metadata['json_sidecar'] = json_metadata
                logger.debug(f"Loaded JSON sidecar from {json_path}")
            except Exception as e:
                logger.warning(f"Failed to load JSON sidecar: {e}")

        return metadata

    def extract_anatomical_metadata(
        self,
        nifti_path: Union[str, Path],
        modality: str = "T1w",
        json_path: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Extract metadata from anatomical MRI data

        Parameters
        ----------
        nifti_path : str or Path
            Path to anatomical NIfTI file
        modality : str
            Modality type (T1w, T2w, FLAIR, etc.)
        json_path : str or Path, optional
            Path to JSON sidecar

        Returns
        -------
        dict
            Dictionary containing anatomical metadata
        """
        # Get basic NIfTI metadata
        metadata = self.extract_nifti_metadata(nifti_path, load_data=False)
        metadata['modality'] = modality

        # Load JSON sidecar if available
        if json_path and Path(json_path).exists():
            try:
                with open(json_path, 'r') as f:
                    json_metadata = json.load(f)
                metadata['json_sidecar'] = json_metadata
            except Exception as e:
                logger.warning(f"Failed to load JSON sidecar: {e}")

        return metadata

    def validate_data_quality(
        self,
        nifti_path: Union[str, Path],
        bval_path: Optional[Union[str, Path]] = None,
        bvec_path: Optional[Union[str, Path]] = None,
        strict: bool = False
    ) -> Dict[str, List[str]]:
        """
        Validate data quality with comprehensive checks

        Parameters
        ----------
        nifti_path : str or Path
            Path to NIfTI file
        bval_path : str or Path, optional
            Path to bval file (for diffusion data)
        bvec_path : str or Path, optional
            Path to bvec file (for diffusion data)
        strict : bool
            If True, raise exception on warnings

        Returns
        -------
        dict
            Dictionary with 'passed', 'warnings', 'failures' lists
        """
        self.quality_checks = {
            'passed': [],
            'warnings': [],
            'failures': []
        }

        nifti_path = Path(nifti_path)

        # Load image
        try:
            img = nib.load(nifti_path)
            data = np.asarray(img.dataobj)
        except Exception as e:
            self.quality_checks['failures'].append(f"Failed to load NIfTI: {e}")
            return self.quality_checks

        # Check 1: File integrity
        self._check_file_integrity(nifti_path)

        # Check 2: Data dimensions
        self._check_dimensions(data, img.header)

        # Check 3: NaN and Inf values
        self._check_invalid_values(data)

        # Check 4: Data range
        self._check_data_range(data)

        # Check 5: Header consistency
        self._check_header_consistency(img.header)

        # Check 6: Orientation
        self._check_orientation(img)

        # For diffusion data
        if bval_path and bvec_path:
            self._check_diffusion_quality(data, bval_path, bvec_path)

        # Check for any failures
        if self.quality_checks['failures']:
            logger.error(f"Quality validation failed: {len(self.quality_checks['failures'])} failures")
            if strict:
                raise QualityCheckError(f"Validation failures: {self.quality_checks['failures']}")

        # Check for warnings
        if self.quality_checks['warnings']:
            logger.warning(f"Quality validation warnings: {len(self.quality_checks['warnings'])}")
            if strict:
                raise QualityCheckError(f"Validation warnings: {self.quality_checks['warnings']}")

        logger.info(
            f"Quality validation complete: "
            f"{len(self.quality_checks['passed'])} passed, "
            f"{len(self.quality_checks['warnings'])} warnings, "
            f"{len(self.quality_checks['failures'])} failures"
        )

        return self.quality_checks

    def generate_scan_manifest(
        self,
        nifti_path: Union[str, Path],
        bval_path: Optional[Union[str, Path]] = None,
        bvec_path: Optional[Union[str, Path]] = None,
        json_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        validate: bool = True
    ) -> Dict:
        """
        Generate comprehensive scan manifest

        Parameters
        ----------
        nifti_path : str or Path
            Path to NIfTI file
        bval_path : str or Path, optional
            Path to bval file
        bvec_path : str or Path, optional
            Path to bvec file
        json_path : str or Path, optional
            Path to JSON sidecar
        output_path : str or Path, optional
            Path to save manifest JSON
        validate : bool
            Perform quality validation

        Returns
        -------
        dict
            Comprehensive scan manifest
        """
        manifest = {
            'manifest_version': '1.0',
            'generation_timestamp': datetime.now().isoformat(),
        }

        # Determine data type and extract metadata
        if bval_path and bvec_path:
            # Diffusion data
            manifest['data_type'] = 'diffusion'
            manifest['metadata'] = self.extract_diffusion_metadata(
                nifti_path, bval_path, bvec_path, json_path
            )
        else:
            # Anatomical or other data
            manifest['data_type'] = 'anatomical'
            manifest['metadata'] = self.extract_anatomical_metadata(
                nifti_path, json_path=json_path
            )

        # Perform quality validation
        if validate:
            manifest['quality_checks'] = self.validate_data_quality(
                nifti_path, bval_path, bvec_path, strict=False
            )

        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            with open(output_path, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            logger.info(f"Saved manifest to {output_path}")

        return manifest

    # Private helper methods

    def _extract_basic_properties(
        self,
        header: nib.nifti1.Nifti1Header,
        img: nib.nifti1.Nifti1Image
    ) -> Dict:
        """Extract basic image properties"""
        dims = header['dim'][1:header['dim'][0]+1].tolist()

        return {
            'ndim': int(header['dim'][0]),
            'dims': dims,
            'nvox': int(np.prod(dims)),
        }

    def _extract_spatial_properties(
        self,
        header: nib.nifti1.Nifti1Header,
        img: nib.nifti1.Nifti1Image
    ) -> Dict:
        """Extract spatial properties"""
        zooms = header.get_zooms()

        return {
            'voxel_size': list(zooms[:3]),
            'voxel_units': header.get_xyzt_units()[0],
            'time_units': header.get_xyzt_units()[1] if len(zooms) > 3 else None,
            'fov_mm': [d * v for d, v in zip(header['dim'][1:4], zooms[:3])],
        }

    def _extract_orientation(self, img: nib.nifti1.Nifti1Image) -> Dict:
        """Extract orientation information"""
        header = img.header

        # Get orientation codes
        orientation = nib.aff2axcodes(img.affine)

        return {
            'orientation': ''.join(orientation),
            'qform_code': int(header['qform_code']),
            'sform_code': int(header['sform_code']),
            'qform_code_description': self._get_form_code_description(header['qform_code']),
            'sform_code_description': self._get_form_code_description(header['sform_code']),
        }

    def _extract_scanner_parameters(self, header: nib.nifti1.Nifti1Header) -> Dict:
        """Extract scanner parameters from header"""
        params = {}

        # TR (repetition time)
        try:
            if 'pixdim' in header and len(header['pixdim']) > 4:
                tr = float(header['pixdim'][4])
                if tr > 0:
                    params['TR'] = tr
        except Exception:
            pass

        # Description field
        try:
            descrip = header['descrip'].tobytes().decode('utf-8', errors='ignore').strip('\x00')
            if descrip:
                params['description'] = descrip
        except Exception:
            pass

        # Auxiliary file
        try:
            aux_file = header['aux_file'].tobytes().decode('utf-8', errors='ignore').strip('\x00')
            if aux_file:
                params['aux_file'] = aux_file
        except Exception:
            pass

        return {'scanner_parameters': params}

    def _extract_datatype_info(self, header: nib.nifti1.Nifti1Header) -> Dict:
        """Extract datatype information"""
        return {
            'datatype': header.get_data_dtype().name,
            'bitpix': int(header['bitpix']),
            'scl_slope': float(header['scl_slope']) if header['scl_slope'] != 0 else 1.0,
            'scl_inter': float(header['scl_inter']),
        }

    def _extract_data_statistics(self, data: np.ndarray) -> Dict:
        """Extract data statistics"""
        return {
            'data_stats': {
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'median': float(np.median(data)),
                'percentile_1': float(np.percentile(data, 1)),
                'percentile_99': float(np.percentile(data, 99)),
            }
        }

    def _get_unique_bvals(self, bvals: np.ndarray, threshold: int = 50) -> List[int]:
        """Get unique b-values rounded to nearest 100"""
        rounded = np.round(bvals, -2)
        unique = np.unique(rounded)
        return sorted([int(b) for b in unique])

    def _identify_shells(self, bvals: np.ndarray, threshold: int = 50) -> List[Dict]:
        """Identify b-value shells"""
        rounded = np.round(bvals, -2)
        unique_bvals = np.unique(rounded)

        shells = []
        for bval in unique_bvals:
            if bval < threshold:
                shell_name = "b0"
            else:
                shell_name = f"b{int(bval)}"

            count = np.sum(rounded == bval)
            shells.append({
                'name': shell_name,
                'bvalue': int(bval),
                'num_directions': int(count)
            })

        return shells

    def _check_gradient_normalization(
        self,
        bvecs: np.ndarray,
        bvals: np.ndarray,
        tolerance: float = 0.1
    ) -> bool:
        """Check if gradient vectors are normalized"""
        b0_threshold = 50
        non_b0 = bvals > b0_threshold

        if not np.any(non_b0):
            return True

        norms = np.linalg.norm(bvecs[:, non_b0], axis=0)
        return bool(np.allclose(norms, 1.0, atol=tolerance))

    def _get_form_code_description(self, code: int) -> str:
        """Get description for qform/sform code"""
        descriptions = {
            0: 'Unknown',
            1: 'Scanner Anat',
            2: 'Aligned Anat',
            3: 'Talairach',
            4: 'MNI152'
        }
        return descriptions.get(int(code), f'Unknown ({code})')

    # Quality check methods

    def _check_file_integrity(self, file_path: Path):
        """Check file integrity"""
        try:
            if file_path.stat().st_size == 0:
                self.quality_checks['failures'].append("File is empty")
            else:
                self.quality_checks['passed'].append("File integrity: OK")
        except Exception as e:
            self.quality_checks['failures'].append(f"File access error: {e}")

    def _check_dimensions(self, data: np.ndarray, header: nib.nifti1.Nifti1Header):
        """Check data dimensions"""
        expected_dims = header['dim'][1:header['dim'][0]+1]
        actual_dims = data.shape

        if tuple(expected_dims) != actual_dims:
            self.quality_checks['warnings'].append(
                f"Dimension mismatch: header={expected_dims}, data={actual_dims}"
            )
        else:
            self.quality_checks['passed'].append("Dimensions: OK")

        # Check for reasonable dimensions
        if any(d < 2 for d in actual_dims[:3]):
            self.quality_checks['failures'].append(
                f"Suspicious spatial dimensions: {actual_dims[:3]}"
            )

    def _check_invalid_values(self, data: np.ndarray):
        """Check for NaN and Inf values"""
        nan_count = np.sum(np.isnan(data))
        inf_count = np.sum(np.isinf(data))

        if nan_count > 0:
            self.quality_checks['failures'].append(f"Found {nan_count} NaN values")
        else:
            self.quality_checks['passed'].append("No NaN values: OK")

        if inf_count > 0:
            self.quality_checks['failures'].append(f"Found {inf_count} Inf values")
        else:
            self.quality_checks['passed'].append("No Inf values: OK")

    def _check_data_range(self, data: np.ndarray):
        """Check data range"""
        data_min = np.min(data)
        data_max = np.max(data)

        if data_min < 0:
            self.quality_checks['warnings'].append(
                f"Data contains negative values (min={data_min:.2f})"
            )

        if data_max > 1e6:
            self.quality_checks['warnings'].append(
                f"Data contains very large values (max={data_max:.2e})"
            )

        if data_min == data_max:
            self.quality_checks['failures'].append("Data is constant (no variation)")
        else:
            self.quality_checks['passed'].append("Data range: OK")

    def _check_header_consistency(self, header: nib.nifti1.Nifti1Header):
        """Check header consistency"""
        # Check qform and sform
        qform_code = int(header['qform_code'])
        sform_code = int(header['sform_code'])

        if qform_code == 0 and sform_code == 0:
            self.quality_checks['warnings'].append(
                "Both qform and sform codes are 0 (unknown orientation)"
            )
        else:
            self.quality_checks['passed'].append("Orientation codes: OK")

    def _check_orientation(self, img: nib.nifti1.Nifti1Image):
        """Check image orientation"""
        try:
            orientation = nib.aff2axcodes(img.affine)
            orientation_str = ''.join(orientation)
            self.quality_checks['passed'].append(f"Orientation: {orientation_str}")
        except Exception as e:
            self.quality_checks['warnings'].append(f"Could not determine orientation: {e}")

    def _check_diffusion_quality(
        self,
        data: np.ndarray,
        bval_path: Path,
        bvec_path: Path
    ):
        """Additional quality checks for diffusion data"""
        # Load bvals and bvecs
        try:
            bvals = np.loadtxt(bval_path)
            bvecs = np.loadtxt(bvec_path)

            if bvals.ndim > 1:
                bvals = bvals.flatten()
            if bvecs.shape[0] != 3:
                bvecs = bvecs.T

            # Check number of volumes matches
            num_volumes = data.shape[3] if data.ndim == 4 else 1

            if len(bvals) != num_volumes:
                self.quality_checks['failures'].append(
                    f"bvals count ({len(bvals)}) != volumes ({num_volumes})"
                )
            else:
                self.quality_checks['passed'].append("bvals count matches volumes: OK")

            if bvecs.shape[1] != num_volumes:
                self.quality_checks['failures'].append(
                    f"bvecs count ({bvecs.shape[1]}) != volumes ({num_volumes})"
                )
            else:
                self.quality_checks['passed'].append("bvecs count matches volumes: OK")

            # Check gradient normalization
            if self._check_gradient_normalization(bvecs, bvals):
                self.quality_checks['passed'].append("Gradient vectors normalized: OK")
            else:
                self.quality_checks['warnings'].append(
                    "Gradient vectors are not unit normalized"
                )

        except Exception as e:
            self.quality_checks['failures'].append(f"Diffusion validation error: {e}")


def setup_logging(level: int = logging.INFO):
    """Configure logging for metadata extractor"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


if __name__ == "__main__":
    # Example usage
    setup_logging(logging.DEBUG)

    extractor = MetadataExtractor()

    # Example: Extract metadata
    # metadata = extractor.extract_nifti_metadata("path/to/file.nii.gz")
    # print(json.dumps(metadata, indent=2))

    # Example: Validate quality
    # quality = extractor.validate_data_quality("path/to/file.nii.gz")
    # print(quality)

    # Example: Generate manifest
    # manifest = extractor.generate_scan_manifest(
    #     nifti_path="path/to/dwi.nii.gz",
    #     bval_path="path/to/dwi.bval",
    #     bvec_path="path/to/dwi.bvec",
    #     output_path="manifest.json"
    # )

    print("MetadataExtractor module ready")
