"""
Dataset Ingestion & Validation Layer for NeuroTract 2.0

Enforces strict boundary validation, detects data formats, computes checksums,
validates gradient tables, normalizes metadata safely, and generates authoritative
Dataset Validation Reports.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import hashlib
import json
import logging
import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a scientific or structural data issue"""
    severity: str  # 'ERROR' or 'WARNING' or 'INFO'
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class GradientSummary:
    """Summary of diffusion gradient scheme"""
    n_total: int
    n_b0: int
    n_dwi: int
    unique_bvals: List[float]
    shell_distribution: Dict[str, int]
    is_unit_normalized: bool
    max_norm_deviation: float
    b0_indices: List[int]


@dataclass
class DatasetValidationReport:
    """
    Authoritative Dataset Validation Report
    Documents all properties, provenance, checksums, and validation issues.
    """
    is_valid: bool
    dataset_format: str
    file_paths: Dict[str, str]
    checksums_sha256: Dict[str, str]
    dimensions: Optional[List[int]] = None
    voxel_size_mm: Optional[List[float]] = None
    num_volumes: Optional[int] = None
    coordinate_system: Optional[str] = None
    orientation_codes: Optional[str] = None
    gradient_summary: Optional[GradientSummary] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    transformations_applied: List[Dict[str, Any]] = field(default_factory=list)
    source_info: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization"""
        d = asdict(self)
        if self.gradient_summary:
            d['gradient_summary'] = asdict(self.gradient_summary)
        return d


class DatasetValidator:
    """
    Validates input diffusion MRI datasets at ingestion boundary
    """

    def __init__(self, b0_threshold: float = 50.0, norm_tolerance: float = 0.05):
        self.b0_threshold = b0_threshold
        self.norm_tolerance = norm_tolerance

    @staticmethod
    def compute_sha256(filepath: Union[str, Path]) -> str:
        """Compute SHA-256 hash of a file"""
        p = Path(filepath)
        if not p.exists() or not p.is_file():
            return ""
        sha256 = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def validate_dwi_dataset(
        self,
        dwi_path: Union[str, Path],
        bval_path: Optional[Union[str, Path]] = None,
        bvec_path: Optional[Union[str, Path]] = None,
        json_path: Optional[Union[str, Path]] = None,
        source_name: Optional[str] = None,
        license_info: Optional[str] = None
    ) -> DatasetValidationReport:
        """
        Validate DWI NIfTI volume against bval/bvec gradient tables
        """
        dwi_p = Path(dwi_path)
        file_paths = {"dwi": str(dwi_p)}
        checksums = {}
        errors = []
        warnings = []
        transformations = []

        # Check DWI file existence & non-zero size
        if not dwi_p.exists():
            return DatasetValidationReport(
                is_valid=False,
                dataset_format="DWI_NIFTI",
                file_paths=file_paths,
                checksums_sha256={},
                errors=[f"DWI file not found: {dwi_p}"]
            )

        if dwi_p.stat().st_size == 0:
            return DatasetValidationReport(
                is_valid=False,
                dataset_format="DWI_NIFTI",
                file_paths=file_paths,
                checksums_sha256={str(dwi_p): "00000000"},
                errors=[f"DWI file is empty (0 bytes): {dwi_p}"]
            )

        checksums["dwi"] = self.compute_sha256(dwi_p)

        # Attempt to auto-discover bval/bvec if not provided
        if bval_path is None:
            for cand in [dwi_p.with_suffix('').with_suffix('.bval'), dwi_p.with_suffix('').with_suffix('.bvals'),
                         dwi_p.parent / f"{dwi_p.stem.replace('.nii','')}.bval",
                         dwi_p.parent / f"{dwi_p.stem.replace('.nii','')}.bvals"]:
                if cand.exists():
                    bval_path = cand
                    break

        if bvec_path is None:
            for cand in [dwi_p.with_suffix('').with_suffix('.bvec'), dwi_p.with_suffix('').with_suffix('.bvecs'),
                         dwi_p.parent / f"{dwi_p.stem.replace('.nii','')}.bvec",
                         dwi_p.parent / f"{dwi_p.stem.replace('.nii','')}.bvecs"]:
                if cand.exists():
                    bvec_path = cand
                    break

        if bval_path:
            file_paths["bval"] = str(bval_path)
            checksums["bval"] = self.compute_sha256(bval_path)
        else:
            errors.append("Missing required b-values file (.bval)")

        if bvec_path:
            file_paths["bvec"] = str(bvec_path)
            checksums["bvec"] = self.compute_sha256(bvec_path)
        else:
            errors.append("Missing required b-vectors file (.bvec)")

        # Load NIfTI header and shape
        try:
            nii = nib.load(str(dwi_p))
            hdr = nii.header
            dims = list(nii.shape)
            voxel_size = [float(v) for v in hdr.get_zooms()[:len(dims)]]
            affine = nii.affine

            # Coordinate orientation
            orientation_codes = "".join(nib.aff2axcodes(affine))
            coord_sys = "RAS+" if orientation_codes.startswith("R") else "LAS+" if orientation_codes.startswith("L") else orientation_codes
        except Exception as e:
            return DatasetValidationReport(
                is_valid=False,
                dataset_format="DWI_NIFTI",
                file_paths=file_paths,
                checksums_sha256=checksums,
                errors=[f"Corrupt NIfTI file or invalid header: {e}"]
            )

        if len(dims) != 4:
            errors.append(f"DWI dataset must be 4-dimensional; found {len(dims)} dimensions {dims}")
            num_vols = dims[-1] if dims else 0
        else:
            num_vols = dims[3]

        # Load and validate bvals/bvecs if present
        bvals = None
        bvecs = None
        gradient_summary = None

        if bval_path and Path(bval_path).exists():
            try:
                bval_raw = np.loadtxt(str(bval_path)).flatten()
                bvals = bval_raw.astype(float)
            except Exception as e:
                errors.append(f"Failed to parse bval file: {e}")

        if bvec_path and Path(bvec_path).exists():
            try:
                bvec_raw = np.loadtxt(str(bvec_path))
                if bvec_raw.ndim != 2:
                    errors.append(f"bvec file must be 2D array; got shape {bvec_raw.shape}")
                elif bvec_raw.shape[0] == 3 and bvec_raw.shape[1] != 3:
                    # 3 x N format -> transpose to N x 3
                    bvecs = bvec_raw.T.astype(float)
                    transformations.append({
                        "type": "bvec_transposition",
                        "original_shape": list(bvec_raw.shape),
                        "transformed_shape": list(bvecs.shape),
                        "reason": "Standardized gradient matrix from 3xN to Nx3"
                    })
                elif bvec_raw.shape[1] == 3:
                    bvecs = bvec_raw.astype(float)
                else:
                    errors.append(f"Invalid bvec shape {bvec_raw.shape}; expected (3, N) or (N, 3)")
            except Exception as e:
                errors.append(f"Failed to parse bvec file: {e}")

        # Dimension consistency check
        if bvals is not None and num_vols is not None:
            if len(bvals) != num_vols:
                errors.append(
                    f"Dimension mismatch: DWI volume has {num_vols} volumes, but bval table contains {len(bvals)} entries"
                )

        if bvecs is not None and num_vols is not None:
            if bvecs.shape[0] != num_vols:
                errors.append(
                    f"Dimension mismatch: DWI volume has {num_vols} volumes, but bvec table contains {bvecs.shape[0]} directions"
                )

        # Gradient analysis
        if bvals is not None and bvecs is not None and len(errors) == 0:
            b0_mask = bvals <= self.b0_threshold
            n_b0 = int(np.sum(b0_mask))
            n_dwi = int(np.sum(~b0_mask))
            b0_indices = [int(i) for i in np.where(b0_mask)[0]]

            if n_b0 == 0:
                errors.append(f"No b0 volumes found (b <= {self.b0_threshold}). At least 1 b0 volume is required.")

            # Shell distribution
            rounded_bvals = np.round(bvals / 100) * 100
            unique_shells = sorted(list(set(rounded_bvals)))
            shell_dist = {f"b{int(b)}": int(np.sum(rounded_bvals == b)) for b in unique_shells}

            # Vector normalization check for diffusion directions
            dwi_bvecs = bvecs[~b0_mask]
            if len(dwi_bvecs) > 0:
                norms = np.linalg.norm(dwi_bvecs, axis=1)
                max_dev = float(np.max(np.abs(norms - 1.0)))
                is_unit = bool(max_dev <= self.norm_tolerance)

                if not is_unit:
                    warnings.append(
                        f"Gradient vectors deviate from unit norm by up to {max_dev:.4f}. Normalization will be required."
                    )
            else:
                max_dev = 0.0
                is_unit = True

            gradient_summary = GradientSummary(
                n_total=len(bvals),
                n_b0=n_b0,
                n_dwi=n_dwi,
                unique_bvals=[float(b) for b in sorted(list(set(bvals)))],
                shell_distribution=shell_dist,
                is_unit_normalized=is_unit,
                max_norm_deviation=max_dev,
                b0_indices=b0_indices
            )

        # Check optional JSON sidecar
        metadata = {}
        if json_path and Path(json_path).exists():
            file_paths["json"] = str(json_path)
            checksums["json"] = self.compute_sha256(json_path)
            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                warnings.append(f"Could not read JSON sidecar: {e}")

        # Source / License
        source_info = {}
        if source_name:
            source_info["name"] = source_name
        if license_info:
            source_info["license"] = license_info

        is_valid = len(errors) == 0

        return DatasetValidationReport(
            is_valid=is_valid,
            dataset_format="DWI_NIFTI",
            file_paths=file_paths,
            checksums_sha256=checksums,
            dimensions=dims,
            voxel_size_mm=voxel_size,
            num_volumes=num_vols,
            coordinate_system=coord_sys,
            orientation_codes=orientation_codes,
            gradient_summary=gradient_summary,
            metadata=metadata,
            warnings=warnings,
            errors=errors,
            transformations_applied=transformations,
            source_info=source_info
        )


def validate_dataset(
    dwi_path: Union[str, Path],
    bval_path: Optional[Union[str, Path]] = None,
    bvec_path: Optional[Union[str, Path]] = None,
    json_path: Optional[Union[str, Path]] = None
) -> DatasetValidationReport:
    """Convenience functional interface for dataset validation"""
    validator = DatasetValidator()
    return validator.validate_dwi_dataset(dwi_path, bval_path, bvec_path, json_path)
