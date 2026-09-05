"""
Orthogonal MRI Slice Extractor for Synchronized 2D/3D Viewer in NeuroTract 2.0

Extracts real axial, coronal, and sagittal slice planes from subject 3D NIfTI volumes
(DTI FA, MD, or T1 anatomical). Generates optimized base64-encoded PNG images and
intensity matrices with voxel coordinates for synchronized viewport inspection.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import io
import base64
import numpy as np
import nibabel as nib
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class SliceExtractor:
    """
    Extracts orthogonal slice planes from 3D neuroimaging volumes.
    """

    def __init__(self, volume_path: Path):
        self.volume_path = Path(volume_path)
        if not self.volume_path.exists():
            raise FileNotFoundError(f"Volume file not found: {self.volume_path}")

        img = nib.load(str(self.volume_path))
        self.data = np.asarray(img.dataobj, dtype=np.float32)
        if self.data.ndim == 4:
            self.data = self.data[..., 0]  # Take first volume if 4D

        self.shape = self.data.shape  # (X, Y, Z)
        self.affine = img.affine
        self.zooms = img.header.get_zooms()[:3]

        # Calculate robust data range for intensity windowing (2nd to 98th percentile)
        valid_mask = ~np.isnan(self.data) & (self.data != 0)
        if np.any(valid_mask):
            self.p2 = float(np.percentile(self.data[valid_mask], 2))
            self.p98 = float(np.percentile(self.data[valid_mask], 98))
        else:
            self.p2, self.p98 = 0.0, 1.0

    def _normalize_slice_to_png_base64(self, slice_2d: np.ndarray) -> Tuple[str, List[int]]:
        """Normalize 2D numpy slice and encode as PNG base64 string"""
        # Window & level
        clipped = np.clip(slice_2d, self.p2, self.p98)
        denom = self.p98 - self.p2
        if denom > 1e-8:
            normalized = ((clipped - self.p2) / denom * 255.0).astype(np.uint8)
        else:
            normalized = np.zeros_like(slice_2d, dtype=np.uint8)

        # Flip vertically to match standard radiological/neurological screen coords
        flipped = np.flipud(normalized)

        img = Image.fromarray(flipped, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}", list(slice_2d.shape)

    def get_orthogonal_slices(
        self,
        axial_pct: float = 0.5,
        coronal_pct: float = 0.5,
        sagittal_pct: float = 0.5
    ) -> Dict[str, Any]:
        """
        Extract orthogonal slices at specified fractional coordinates [0.0, 1.0].

        Returns
        -------
        Dictionary containing PNG images, coordinates, and voxel dimensions for all 3 planes.
        """
        nx, ny, nz = self.shape

        # Clamped slice indices
        x_idx = int(np.clip(round(sagittal_pct * (nx - 1)), 0, nx - 1))
        y_idx = int(np.clip(round(coronal_pct * (ny - 1)), 0, ny - 1))
        z_idx = int(np.clip(round(axial_pct * (nz - 1)), 0, nz - 1))

        # Axial: slice in Z (XY plane, shape: (X, Y)) -> transpose to (Y, X)
        axial_2d = self.data[:, :, z_idx].T
        axial_b64, axial_dims = self._normalize_slice_to_png_base64(axial_2d)

        # Coronal: slice in Y (XZ plane, shape: (X, Z)) -> transpose to (Z, X)
        coronal_2d = self.data[:, y_idx, :].T
        coronal_b64, coronal_dims = self._normalize_slice_to_png_base64(coronal_2d)

        # Sagittal: slice in X (YZ plane, shape: (Y, Z)) -> transpose to (Z, Y)
        sagittal_2d = self.data[x_idx, :, :].T
        sagittal_b64, sagittal_dims = self._normalize_slice_to_png_base64(sagittal_2d)

        return {
            "volume_shape": list(self.shape),
            "voxel_size_mm": [float(z) for z in self.zooms],
            "intensity_range": [self.p2, self.p98],
            "indices": {
                "axial": z_idx,
                "coronal": y_idx,
                "sagittal": x_idx
            },
            "slices": {
                "axial": {
                    "image": axial_b64,
                    "index": z_idx,
                    "max_index": nz - 1,
                    "percentage": axial_pct,
                    "plane": "XY (Axial / Transverse)",
                    "dims": axial_dims
                },
                "coronal": {
                    "image": coronal_b64,
                    "index": y_idx,
                    "max_index": ny - 1,
                    "percentage": coronal_pct,
                    "plane": "XZ (Coronal / Frontal)",
                    "dims": coronal_dims
                },
                "sagittal": {
                    "image": sagittal_b64,
                    "index": x_idx,
                    "max_index": nx - 1,
                    "percentage": sagittal_pct,
                    "plane": "YZ (Sagittal)",
                    "dims": sagittal_dims
                }
            }
        }


# Cache extractors by path to avoid re-reading disk
_extractor_cache: Dict[str, SliceExtractor] = {}

def get_subject_slices(
    subject_id: str,
    axial_pct: float = 0.5,
    coronal_pct: float = 0.5,
    sagittal_pct: float = 0.5,
    modality: str = "fa"
) -> Dict[str, Any]:
    """Get orthogonal slices for a subject"""
    subject_dir = Path("output") / subject_id
    if modality == "fa":
        vol_path = subject_dir / "dti" / "dti_fa.nii.gz"
    elif modality == "md":
        vol_path = subject_dir / "dti" / "dti_md.nii.gz"
    else:
        vol_path = subject_dir / "preprocessed" / "preprocessed_brain_mask.nii.gz"

    if not vol_path.exists():
        # Fallback to T1 or mask in datasets
        ds_dir = Path("datasets") / "Stanford dataset"
        t1_cand = ds_dir / f"{subject_id}_t1.nii.gz"
        if t1_cand.exists():
            vol_path = t1_cand
        else:
            raise FileNotFoundError(f"No volume found for subject {subject_id} and modality {modality}")

    key = str(vol_path)
    if key not in _extractor_cache:
        _extractor_cache[key] = SliceExtractor(vol_path)

    return _extractor_cache[key].get_orthogonal_slices(axial_pct, coronal_pct, sagittal_pct)
