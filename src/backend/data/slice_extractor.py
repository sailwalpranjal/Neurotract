"""
Orthogonal MRI Slice Extractor for Synchronized 2D/3D Viewer in NeuroTract 2.0

Extracts real axial, coronal, and sagittal slice planes from subject 3D NIfTI volumes
(DTI FA, MD, B0 baseline, or T1 anatomical). Generates optimized base64-encoded PNG images,
2D scalar intensity matrices for zero-latency client hover telemetry, physical RAS+ scanner
coordinates, and real-time anatomical parcellation sampling.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import io
import base64
import numpy as np
import nibabel as nib
import logging
from PIL import Image

try:
    from ..surfaces.parcellation_mapping import DESIKAN_KILLIANY_89
except ImportError:
    DESIKAN_KILLIANY_89 = {}

logger = logging.getLogger(__name__)

MODALITY_METADATA = {
    "fa": {
        "full_name": "Fractional Anisotropy (DTI)",
        "unit": "unitless [0, 1]",
        "description": "Directional variance of water diffusion along white matter tracts",
        "colormap": "gray",
    },
    "md": {
        "full_name": "Mean Diffusivity (DTI)",
        "unit": "mm²/s",
        "description": "Magnitude of isotropic water diffusion",
        "colormap": "bone",
    },
    "b0": {
        "full_name": "Diffusion B0 Reference",
        "unit": "a.u. (MR signal)",
        "description": "Non-diffusion-weighted T2-like baseline signal",
        "colormap": "gray",
    },
    "t1": {
        "full_name": "T1-Weighted Anatomical",
        "unit": "a.u.",
        "description": "High-contrast structural anatomical scan",
        "colormap": "gray",
    },
    "mask": {
        "full_name": "Brain Extraction Mask",
        "unit": "binary {0, 1}",
        "description": "Binary skull-stripped intracranial brain parenchyma mask",
        "colormap": "gray",
    },
}


class SliceExtractor:
    """
    Extracts orthogonal slice planes and scalar telemetry from 3D neuroimaging volumes.
    """

    def __init__(self, volume_path: Path, modality: str = "fa"):
        self.volume_path = Path(volume_path)
        if not self.volume_path.exists():
            raise FileNotFoundError(f"Volume file not found: {self.volume_path}")

        self.modality = modality.lower()
        img = nib.load(str(self.volume_path))
        raw_data = np.asarray(img.dataobj, dtype=np.float32)
        if raw_data.ndim == 4:
            self.data = raw_data[..., 0]  # Take first volume if 4D (e.g. B0 volume of DWI series)
        else:
            self.data = raw_data

        self.shape = self.data.shape  # (X, Y, Z)
        self.affine = img.affine
        self.zooms = img.header.get_zooms()[:3]

        # Calculate robust data range for intensity windowing (2nd to 98th percentile of non-zero)
        valid_mask = ~np.isnan(self.data) & (self.data > 0)
        if np.any(valid_mask):
            self.p2 = float(np.percentile(self.data[valid_mask], 2))
            self.p98 = float(np.percentile(self.data[valid_mask], 98))
        else:
            self.p2 = float(np.min(self.data))
            self.p98 = float(np.max(self.data)) if float(np.max(self.data)) > 0 else 1.0

    def _normalize_slice_to_png_base64(self, slice_2d: np.ndarray) -> Tuple[str, List[int], np.ndarray]:
        """
        Normalize 2D numpy slice and encode as PNG base64 string.
        Flips vertically to match standard radiological/neurological screen coordinates.
        Returns base64 string, dimensions [height, width], and flipped scalar slice.
        """
        # Window & level
        clipped = np.clip(slice_2d, self.p2, self.p98)
        denom = self.p98 - self.p2
        if denom > 1e-8:
            normalized = ((clipped - self.p2) / denom * 255.0).astype(np.uint8)
        else:
            normalized = np.zeros_like(slice_2d, dtype=np.uint8)

        # Flip vertically to match standard radiological/neurological screen coordinates
        flipped_img = np.flipud(normalized)
        flipped_scalar = np.flipud(slice_2d)

        img = Image.fromarray(flipped_img, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}", list(flipped_img.shape), flipped_scalar

    def get_orthogonal_slices(
        self,
        x_idx: Optional[int] = None,
        y_idx: Optional[int] = None,
        z_idx: Optional[int] = None,
        axial_pct: Optional[float] = None,
        coronal_pct: Optional[float] = None,
        sagittal_pct: Optional[float] = None,
        aparc_data: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Extract orthogonal slices at specified voxel indices or fractional coordinates.
        Includes 2D scalar matrices, scanner physical RAS+ coordinates, and parcellation.
        """
        nx, ny, nz = self.shape

        # Resolve slice indices
        if x_idx is None:
            pct = 0.5 if sagittal_pct is None else sagittal_pct
            x_idx = int(np.clip(round(pct * (nx - 1)), 0, nx - 1))
        else:
            x_idx = int(np.clip(x_idx, 0, nx - 1))

        if y_idx is None:
            pct = 0.5 if coronal_pct is None else coronal_pct
            y_idx = int(np.clip(round(pct * (ny - 1)), 0, ny - 1))
        else:
            y_idx = int(np.clip(y_idx, 0, ny - 1))

        if z_idx is None:
            pct = 0.5 if axial_pct is None else axial_pct
            z_idx = int(np.clip(round(pct * (nz - 1)), 0, nz - 1))
        else:
            z_idx = int(np.clip(z_idx, 0, nz - 1))

        # Axial: slice in Z (XY plane, shape: (X, Y)) -> transpose to (Y, X)
        axial_2d = self.data[:, :, z_idx].T
        axial_b64, axial_dims, axial_scalar_flipped = self._normalize_slice_to_png_base64(axial_2d)

        # Coronal: slice in Y (XZ plane, shape: (X, Z)) -> transpose to (Z, X)
        coronal_2d = self.data[:, y_idx, :].T
        coronal_b64, coronal_dims, coronal_scalar_flipped = self._normalize_slice_to_png_base64(coronal_2d)

        # Sagittal: slice in X (YZ plane, shape: (Y, Z)) -> transpose to (Z, Y)
        sagittal_2d = self.data[x_idx, :, :].T
        sagittal_b64, sagittal_dims, sagittal_scalar_flipped = self._normalize_slice_to_png_base64(sagittal_2d)

        # Calculate scanner physical coordinates (RAS+ in mm) using volume affine
        voxel_homog = np.array([x_idx, y_idx, z_idx, 1.0], dtype=np.float64)
        world_mm = (self.affine @ voxel_homog)[:3]

        # Current voxel scalar value
        current_voxel_val = float(self.data[x_idx, y_idx, z_idx])

        # Parcellation region lookup at current voxel
        parcellation_info = {
            "id": 0,
            "name": "Subcortical White Matter / CSF",
            "abbreviation": "WM/CSF",
            "hemisphere": "bilateral",
            "lobe": "deep",
        }
        if aparc_data is not None:
            try:
                # Check if aparc has identical shape or sample proportionally
                if aparc_data.shape == self.shape:
                    label_id = int(aparc_data[x_idx, y_idx, z_idx])
                else:
                    ax = int(np.clip(round(x_idx / nx * (aparc_data.shape[0] - 1)), 0, aparc_data.shape[0] - 1))
                    ay = int(np.clip(round(y_idx / ny * (aparc_data.shape[1] - 1)), 0, aparc_data.shape[1] - 1))
                    az = int(np.clip(round(z_idx / nz * (aparc_data.shape[2] - 1)), 0, aparc_data.shape[2] - 1))
                    label_id = int(aparc_data[ax, ay, az])

                if label_id in DESIKAN_KILLIANY_89:
                    parcellation_info = {
                        "id": label_id,
                        **DESIKAN_KILLIANY_89[label_id]
                    }
                elif label_id > 0:
                    parcellation_info = {
                        "id": label_id,
                        "name": f"Parcel {label_id}",
                        "abbreviation": f"P{label_id}",
                        "hemisphere": "unknown",
                        "lobe": "cortex",
                    }
            except Exception as e:
                logger.debug(f"Parcellation sampling failed: {e}")

        meta = MODALITY_METADATA.get(self.modality, {
            "full_name": self.modality.upper(),
            "unit": "a.u.",
            "description": "Scalar intensity volume",
            "colormap": "gray",
        })

        # Precision rounding for scalar matrix transfer (4 decimal places for FA/MD, 1 for B0/T1)
        decimals = 1 if self.modality in ("b0", "t1") else 4

        return {
            "volume_shape": list(self.shape),
            "voxel_size_mm": [float(z) for z in self.zooms],
            "intensity_range": [float(self.p2), float(self.p98)],
            "modality": self.modality,
            "modality_name": meta["full_name"],
            "unit": meta["unit"],
            "description": meta["description"],
            "indices": {
                "axial": z_idx,
                "coronal": y_idx,
                "sagittal": x_idx
            },
            "physical_mm": {
                "x": round(float(world_mm[0]), 2),
                "y": round(float(world_mm[1]), 2),
                "z": round(float(world_mm[2]), 2),
            },
            "current_voxel_value": round(current_voxel_val, 4),
            "parcellation": parcellation_info,
            "slices": {
                "axial": {
                    "image": axial_b64,
                    "index": z_idx,
                    "max_index": nz - 1,
                    "percentage": float(z_idx / max(nz - 1, 1)),
                    "plane": "XY (Axial / Transverse)",
                    "dims": axial_dims,
                    "scalar_matrix": np.round(axial_scalar_flipped, decimals).tolist(),
                },
                "coronal": {
                    "image": coronal_b64,
                    "index": y_idx,
                    "max_index": ny - 1,
                    "percentage": float(y_idx / max(ny - 1, 1)),
                    "plane": "XZ (Coronal / Frontal)",
                    "dims": coronal_dims,
                    "scalar_matrix": np.round(coronal_scalar_flipped, decimals).tolist(),
                },
                "sagittal": {
                    "image": sagittal_b64,
                    "index": x_idx,
                    "max_index": nx - 1,
                    "percentage": float(x_idx / max(nx - 1, 1)),
                    "plane": "YZ (Sagittal)",
                    "dims": sagittal_dims,
                    "scalar_matrix": np.round(sagittal_scalar_flipped, decimals).tolist(),
                }
            }
        }


# Cache extractors and parcellations by key to avoid re-reading disk
_extractor_cache: Dict[str, SliceExtractor] = {}
_aparc_cache: Dict[str, Optional[np.ndarray]] = {}


def _get_cached_aparc(subject_id: str) -> Optional[np.ndarray]:
    """Load and cache parcellation data array for a subject"""
    if subject_id in _aparc_cache:
        return _aparc_cache[subject_id]

    candidates = [
        Path("datasets") / "Stanford dataset" / f"{subject_id}_aparc-reduced.nii.gz",
        Path("datasets") / "Stanford dataset" / "SUB1_aparc-reduced.nii.gz",
        Path("output") / subject_id / "aparc-reduced.nii.gz",
    ]

    for cand in candidates:
        if cand.exists():
            try:
                img = nib.load(str(cand))
                data = np.asarray(img.dataobj, dtype=np.uint8)
                _aparc_cache[subject_id] = data
                return data
            except Exception as e:
                logger.warning(f"Could not load aparc from {cand}: {e}")

    _aparc_cache[subject_id] = None
    return None


def get_subject_slices(
    subject_id: str,
    axial_pct: Optional[float] = None,
    coronal_pct: Optional[float] = None,
    sagittal_pct: Optional[float] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    z: Optional[float] = None,
    modality: str = "fa"
) -> Dict[str, Any]:
    """
    Get orthogonal slices for a subject and modality.
    Supports either percentage coordinates or exact voxel indices.
    """
    subject_dir = Path("output") / subject_id
    ds_dir = Path("datasets") / "Stanford dataset"

    mod = modality.lower()
    vol_path: Optional[Path] = None

    if mod == "fa":
        candidates = [
            subject_dir / "dti" / "dti_fa.nii.gz",
            subject_dir / f"{subject_id}_fa.nii.gz",
            subject_dir / "fa.nii.gz",
            Path("output") / "SUB1" / "dti" / "dti_fa.nii.gz",
        ]
        for c in candidates:
            if c.exists():
                vol_path = c
                break

    elif mod == "md":
        candidates = [
            subject_dir / "dti" / "dti_md.nii.gz",
            subject_dir / f"{subject_id}_md.nii.gz",
            subject_dir / "md.nii.gz",
            Path("output") / "SUB1" / "dti" / "dti_md.nii.gz",
        ]
        for c in candidates:
            if c.exists():
                vol_path = c
                break

    elif mod == "b0":
        candidates = [
            subject_dir / "preprocessed" / "preprocessed_dwi.nii.gz",
            subject_dir / f"{subject_id}_b0.nii.gz",
            ds_dir / f"{subject_id}_b1000_1.nii.gz",
            ds_dir / f"{subject_id}_b2000_1.nii.gz",
            ds_dir / "SUB1_b1000_1.nii.gz",
            Path("output") / "SUB1" / "preprocessed" / "preprocessed_dwi.nii.gz",
        ]
        for c in candidates:
            if c.exists():
                vol_path = c
                break

    elif mod == "t1":
        candidates = [
            ds_dir / f"{subject_id}_t1.nii.gz",
            subject_dir / f"{subject_id}_t1.nii.gz",
            ds_dir / "SUB1_t1.nii.gz",
        ]
        for c in candidates:
            if c.exists():
                vol_path = c
                break

    else:  # brain mask or fallback
        candidates = [
            subject_dir / "preprocessed" / "preprocessed_brain_mask.nii.gz",
            subject_dir / "preprocessed" / f"{subject_id}_preproc_brain_mask.nii.gz",
            subject_dir / "preprocessed" / "intermediate" / "brain_mask.nii.gz",
            Path("output") / "SUB1" / "preprocessed" / "preprocessed_brain_mask.nii.gz",
        ]
        for c in candidates:
            if c.exists():
                vol_path = c
                break

    if vol_path is None or not vol_path.exists():
        raise FileNotFoundError(f"No volume found for subject {subject_id} and modality {modality}")

    key = f"{str(vol_path)}_{mod}"
    if key not in _extractor_cache:
        _extractor_cache[key] = SliceExtractor(vol_path, modality=mod)

    extractor = _extractor_cache[key]
    aparc_data = _get_cached_aparc(subject_id)

    # Convert x, y, z to indices if passed as indices (values >= 1.0 or integers)
    x_idx: Optional[int] = None
    y_idx: Optional[int] = None
    z_idx: Optional[int] = None

    if x is not None:
        if x > 1.0:
            x_idx = int(round(x))
        else:
            sagittal_pct = float(x)

    if y is not None:
        if y > 1.0:
            y_idx = int(round(y))
        else:
            coronal_pct = float(y)

    if z is not None:
        if z > 1.0:
            z_idx = int(round(z))
        else:
            axial_pct = float(z)

    return extractor.get_orthogonal_slices(
        x_idx=x_idx,
        y_idx=y_idx,
        z_idx=z_idx,
        axial_pct=axial_pct,
        coronal_pct=coronal_pct,
        sagittal_pct=sagittal_pct,
        aparc_data=aparc_data,
    )

