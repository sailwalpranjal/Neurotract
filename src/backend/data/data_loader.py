#!/usr/bin/env python
"""
Data Loader Module for NeuroTract

Provides robust NIfTI/BIDS data loading capabilities with nibabel.
Handles diffusion MRI data, anatomical images, and associated metadata.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import nibabel as nib
import numpy as np


logger = logging.getLogger(__name__)


class NIfTILoadError(Exception):
    """Exception raised for errors during NIfTI loading"""
    pass


class DataValidationError(Exception):
    """Exception raised for data validation failures"""
    pass


class DiffusionData:
    """Container for diffusion MRI data and associated information"""

    def __init__(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        header: nib.nifti1.Nifti1Header,
        file_path: Path,
        metadata: Optional[Dict] = None
    ):
        self.data = data
        self.affine = affine
        self.bvals = bvals
        self.bvecs = bvecs
        self.header = header
        self.file_path = file_path
        self.metadata = metadata or {}

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get data shape"""
        return self.data.shape

    @property
    def voxel_size(self) -> Tuple[float, float, float]:
        """Get voxel dimensions in mm"""
        return tuple(self.header.get_zooms()[:3])

    @property
    def num_volumes(self) -> int:
        """Get number of diffusion volumes"""
        return self.data.shape[3] if len(self.data.shape) == 4 else 1

    @property
    def num_shells(self) -> int:
        """Get number of b-value shells"""
        unique_bvals = np.unique(self.bvals.round(-2))  # Round to nearest 100
        return len(unique_bvals[unique_bvals > 50])  # Exclude b0

    def get_shells(self) -> List[int]:
        """Get list of b-value shells"""
        unique_bvals = np.unique(self.bvals.round(-2))
        return sorted([int(b) for b in unique_bvals if b > 50])


class AnatomicalData:
    """Container for anatomical MRI data"""

    def __init__(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        header: nib.nifti1.Nifti1Header,
        file_path: Path,
        modality: str = "T1w",
        metadata: Optional[Dict] = None
    ):
        self.data = data
        self.affine = affine
        self.header = header
        self.file_path = file_path
        self.modality = modality
        self.metadata = metadata or {}

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get data shape"""
        return self.data.shape

    @property
    def voxel_size(self) -> Tuple[float, float, float]:
        """Get voxel dimensions in mm"""
        return tuple(self.header.get_zooms()[:3])


class DataLoader:
    """
    Robust NIfTI and BIDS data loader with comprehensive validation

    Features:
    - Load diffusion MRI data with bvals/bvecs
    - Load anatomical T1w/T2w images
    - Memory-mapped loading for large files
    - Header and dimension validation
    - Scanner metadata extraction
    - Support for both BIDS and non-BIDS formats
    """

    def __init__(self, use_mmap: bool = True, validate: bool = True):
        """
        Initialize DataLoader

        Parameters
        ----------
        use_mmap : bool
            Use memory-mapped loading for large files (default: True)
        validate : bool
            Perform data validation checks (default: True)
        """
        self.use_mmap = use_mmap
        self.validate = validate

    def load_diffusion(
        self,
        nifti_path: Union[str, Path],
        bval_path: Optional[Union[str, Path]] = None,
        bvec_path: Optional[Union[str, Path]] = None,
        auto_detect: bool = True
    ) -> DiffusionData:
        """
        Load diffusion MRI data from NIfTI file with bvals/bvecs

        Parameters
        ----------
        nifti_path : str or Path
            Path to diffusion NIfTI file (.nii or .nii.gz)
        bval_path : str or Path, optional
            Path to bval file. Auto-detected if None and auto_detect=True
        bvec_path : str or Path, optional
            Path to bvec file. Auto-detected if None and auto_detect=True
        auto_detect : bool
            Automatically detect bval/bvec files by replacing extension

        Returns
        -------
        DiffusionData
            Container with diffusion data and metadata

        Raises
        ------
        NIfTILoadError
            If file cannot be loaded or required files are missing
        DataValidationError
            If validation fails
        """
        nifti_path = Path(nifti_path)

        if not nifti_path.exists():
            raise NIfTILoadError(f"NIfTI file not found: {nifti_path}")

        logger.info(f"Loading diffusion data from {nifti_path}")

        # Auto-detect bval/bvec if not provided
        if auto_detect:
            if bval_path is None:
                bval_path = self._find_bval_bvec(nifti_path, "bval")
            if bvec_path is None:
                bvec_path = self._find_bval_bvec(nifti_path, "bvec")

        # Load bvals and bvecs
        if bval_path is None:
            raise NIfTILoadError(f"bval file not found for {nifti_path}")
        if bvec_path is None:
            raise NIfTILoadError(f"bvec file not found for {nifti_path}")

        bvals = self._load_bval(bval_path)
        bvecs = self._load_bvec(bvec_path)

        # Load NIfTI
        try:
            if self.use_mmap:
                img = nib.load(nifti_path, mmap=True)
                data = np.asarray(img.dataobj)
            else:
                img = nib.load(nifti_path)
                data = img.get_fdata()
        except Exception as e:
            raise NIfTILoadError(f"Failed to load NIfTI file: {e}")

        # Validate data
        if self.validate:
            self._validate_diffusion_data(data, bvals, bvecs, nifti_path)

        # Extract metadata from header
        metadata = self._extract_nifti_metadata(img.header)

        diffusion_data = DiffusionData(
            data=data,
            affine=img.affine,
            bvals=bvals,
            bvecs=bvecs,
            header=img.header,
            file_path=nifti_path,
            metadata=metadata
        )

        logger.info(
            f"Loaded diffusion data: shape={diffusion_data.shape}, "
            f"voxel_size={diffusion_data.voxel_size}, "
            f"num_volumes={diffusion_data.num_volumes}, "
            f"shells={diffusion_data.get_shells()}"
        )

        return diffusion_data

    def load_anatomical(
        self,
        nifti_path: Union[str, Path],
        modality: str = "T1w"
    ) -> AnatomicalData:
        """
        Load anatomical MRI data from NIfTI file

        Parameters
        ----------
        nifti_path : str or Path
            Path to anatomical NIfTI file (.nii or .nii.gz)
        modality : str
            Modality type (T1w, T2w, etc.)

        Returns
        -------
        AnatomicalData
            Container with anatomical data and metadata

        Raises
        ------
        NIfTILoadError
            If file cannot be loaded
        DataValidationError
            If validation fails
        """
        nifti_path = Path(nifti_path)

        if not nifti_path.exists():
            raise NIfTILoadError(f"NIfTI file not found: {nifti_path}")

        logger.info(f"Loading anatomical {modality} data from {nifti_path}")

        # Load NIfTI
        try:
            if self.use_mmap:
                img = nib.load(nifti_path, mmap=True)
                data = np.asarray(img.dataobj)
            else:
                img = nib.load(nifti_path)
                data = img.get_fdata()
        except Exception as e:
            raise NIfTILoadError(f"Failed to load NIfTI file: {e}")

        # Validate data
        if self.validate:
            self._validate_anatomical_data(data, nifti_path)

        # Extract metadata from header
        metadata = self._extract_nifti_metadata(img.header)

        anatomical_data = AnatomicalData(
            data=data,
            affine=img.affine,
            header=img.header,
            file_path=nifti_path,
            modality=modality,
            metadata=metadata
        )

        logger.info(
            f"Loaded anatomical data: shape={anatomical_data.shape}, "
            f"voxel_size={anatomical_data.voxel_size}, "
            f"modality={modality}"
        )

        return anatomical_data

    def load_nifti_generic(
        self,
        nifti_path: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray, nib.nifti1.Nifti1Header]:
        """
        Load any NIfTI file and return data, affine, and header

        Parameters
        ----------
        nifti_path : str or Path
            Path to NIfTI file

        Returns
        -------
        data : np.ndarray
            Image data
        affine : np.ndarray
            4x4 affine transformation matrix
        header : Nifti1Header
            NIfTI header object
        """
        nifti_path = Path(nifti_path)

        if not nifti_path.exists():
            raise NIfTILoadError(f"NIfTI file not found: {nifti_path}")

        try:
            if self.use_mmap:
                img = nib.load(nifti_path, mmap=True)
                data = np.asarray(img.dataobj)
            else:
                img = nib.load(nifti_path)
                data = img.get_fdata()
        except Exception as e:
            raise NIfTILoadError(f"Failed to load NIfTI file: {e}")

        return data, img.affine, img.header

    def _find_bval_bvec(self, nifti_path: Path, extension: str) -> Optional[Path]:
        """
        Auto-detect bval or bvec file by replacing NIfTI extension

        Tries multiple patterns:
        - Same name with .bval/.bvec extension
        - BIDS sidecar pattern
        """
        # Pattern 1: Replace .nii.gz or .nii with .bval/.bvec
        if nifti_path.name.endswith('.nii.gz'):
            candidate = nifti_path.parent / (nifti_path.name[:-7] + f'.{extension}')
        elif nifti_path.name.endswith('.nii'):
            candidate = nifti_path.parent / (nifti_path.name[:-4] + f'.{extension}')
        else:
            return None

        if candidate.exists():
            logger.debug(f"Found {extension} file: {candidate}")
            return candidate

        # Pattern 2: BIDS pattern - check for _dwi.bval/bvec
        if '_dwi' in nifti_path.name:
            base_name = nifti_path.name.split('_dwi')[0]
            candidate = nifti_path.parent / f"{base_name}_dwi.{extension}"
            if candidate.exists():
                logger.debug(f"Found BIDS {extension} file: {candidate}")
                return candidate

        return None

    def _load_bval(self, bval_path: Union[str, Path]) -> np.ndarray:
        """Load b-values from file"""
        bval_path = Path(bval_path)

        try:
            bvals = np.loadtxt(bval_path)
            # Ensure 1D array
            if bvals.ndim > 1:
                bvals = bvals.flatten()
            return bvals
        except Exception as e:
            raise NIfTILoadError(f"Failed to load bval file {bval_path}: {e}")

    def _load_bvec(self, bvec_path: Union[str, Path]) -> np.ndarray:
        """Load gradient vectors from file"""
        bvec_path = Path(bvec_path)

        try:
            bvecs = np.loadtxt(bvec_path)

            # Ensure shape is (3, N) where N is number of volumes
            if bvecs.shape[0] == 3:
                pass  # Already correct shape
            elif bvecs.shape[1] == 3:
                # Transpose to (3, N)
                bvecs = bvecs.T
                logger.debug("Transposed bvecs from (N, 3) to (3, N)")
            else:
                raise DataValidationError(
                    f"Invalid bvec shape: {bvecs.shape}. Expected (3, N) or (N, 3)"
                )

            return bvecs
        except Exception as e:
            raise NIfTILoadError(f"Failed to load bvec file {bvec_path}: {e}")

    def _validate_diffusion_data(
        self,
        data: np.ndarray,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        file_path: Path
    ):
        """
        Validate diffusion MRI data

        Checks:
        - Data dimensions (must be 4D)
        - No NaN or Inf values
        - bvals/bvecs match number of volumes
        - Gradient vectors are unit normalized
        - Data range is reasonable
        """
        # Check dimensions
        if data.ndim != 4:
            raise DataValidationError(
                f"Expected 4D diffusion data, got {data.ndim}D: {data.shape}"
            )

        num_volumes = data.shape[3]

        # Check for NaN or Inf
        if np.any(np.isnan(data)):
            raise DataValidationError(f"Data contains NaN values: {file_path}")
        if np.any(np.isinf(data)):
            raise DataValidationError(f"Data contains Inf values: {file_path}")

        # Check bvals match
        if len(bvals) != num_volumes:
            raise DataValidationError(
                f"Number of bvals ({len(bvals)}) does not match "
                f"number of volumes ({num_volumes})"
            )

        # Check bvecs match
        if bvecs.shape[1] != num_volumes:
            raise DataValidationError(
                f"Number of bvecs ({bvecs.shape[1]}) does not match "
                f"number of volumes ({num_volumes})"
            )

        # Validate gradient vectors (should be unit normalized, except for b0)
        b0_threshold = 50
        non_b0_indices = bvals > b0_threshold

        if np.any(non_b0_indices):
            norms = np.linalg.norm(bvecs[:, non_b0_indices], axis=0)
            # Allow small deviation from unit norm
            if not np.allclose(norms, 1.0, atol=0.1):
                warnings.warn(
                    f"Gradient vectors are not unit normalized. "
                    f"Norms range: [{norms.min():.3f}, {norms.max():.3f}]"
                )

        # Check data range (diffusion data should be positive)
        if np.any(data < 0):
            warnings.warn(f"Data contains negative values: {file_path}")

        # Check for suspiciously large values
        if np.max(data) > 1e6:
            warnings.warn(
                f"Data contains very large values (max={np.max(data):.2e}). "
                "Check scaling."
            )

        logger.debug(f"Diffusion data validation passed for {file_path}")

    def _validate_anatomical_data(self, data: np.ndarray, file_path: Path):
        """
        Validate anatomical MRI data

        Checks:
        - Data dimensions (must be 3D)
        - No NaN or Inf values
        - Data range is reasonable
        """
        # Check dimensions
        if data.ndim != 3:
            raise DataValidationError(
                f"Expected 3D anatomical data, got {data.ndim}D: {data.shape}"
            )

        # Check for NaN or Inf
        if np.any(np.isnan(data)):
            raise DataValidationError(f"Data contains NaN values: {file_path}")
        if np.any(np.isinf(data)):
            raise DataValidationError(f"Data contains Inf values: {file_path}")

        # Check data range
        if np.any(data < 0):
            warnings.warn(f"Data contains negative values: {file_path}")

        if np.max(data) > 1e6:
            warnings.warn(
                f"Data contains very large values (max={np.max(data):.2e}). "
                "Check scaling."
            )

        logger.debug(f"Anatomical data validation passed for {file_path}")

    def _extract_nifti_metadata(self, header: nib.nifti1.Nifti1Header) -> Dict:
        """
        Extract metadata from NIfTI header

        Returns
        -------
        dict
            Dictionary containing header metadata
        """
        metadata = {
            'dims': header['dim'][1:4].tolist(),
            'voxel_size': header.get_zooms()[:3],
            'qform_code': int(header['qform_code']),
            'sform_code': int(header['sform_code']),
            'datatype': header.get_data_dtype().name,
            'description': header['descrip'].tobytes().decode('utf-8', errors='ignore').strip('\x00'),
        }

        # Try to extract scanner parameters if available
        try:
            # Repetition time (TR)
            if 'pixdim' in header and len(header['pixdim']) > 4:
                tr = float(header['pixdim'][4])
                if tr > 0:
                    metadata['TR'] = tr
        except Exception:
            pass

        try:
            # Some scanners store additional info in descrip or aux_file
            aux_file = header['aux_file'].tobytes().decode('utf-8', errors='ignore').strip('\x00')
            if aux_file:
                metadata['aux_file'] = aux_file
        except Exception:
            pass

        return metadata


def setup_logging(level: int = logging.INFO):
    """Configure logging for data loader"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


if __name__ == "__main__":
    # Example usage
    setup_logging(logging.DEBUG)

    loader = DataLoader(use_mmap=True, validate=True)

    # Example: Load diffusion data
    # dwi_data = loader.load_diffusion(
    #     nifti_path="path/to/dwi.nii.gz",
    #     bval_path="path/to/dwi.bval",
    #     bvec_path="path/to/dwi.bvec"
    # )

    # Example: Load anatomical data
    # anat_data = loader.load_anatomical(
    #     nifti_path="path/to/T1w.nii.gz",
    #     modality="T1w"
    # )

    print("DataLoader module ready")
