"""
Brain Mesh Generator

Generates triangulated brain surface meshes from NIfTI brain masks
using the marching cubes algorithm.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def generate_brain_mesh(
    mask_path: str,
    step_size: int = 1,
    smooth_sigma: float = 1.0,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a triangulated brain surface mesh from a NIfTI brain mask.

    Args:
        mask_path: Path to the brain mask NIfTI file
        step_size: Step size for marching cubes (higher = fewer vertices)
        smooth_sigma: Gaussian smoothing sigma before meshing
        cache_dir: Directory to cache the result (optional)

    Returns:
        Dictionary with vertices, faces, normals, and metadata
    """
    import nibabel as nib
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    # Check cache first
    if cache_dir:
        cache_path = Path(cache_dir) / f"brain_mesh_step{step_size}.json"
        if cache_path.exists():
            logger.info(f"Loading cached brain mesh from {cache_path}")
            with open(cache_path) as f:
                return json.load(f)

    logger.info(f"Generating brain mesh from {mask_path}...")

    # Load brain mask
    img = nib.load(mask_path)
    mask_data = img.get_fdata().astype(np.float32)
    affine = img.affine

    logger.info(f"Mask shape: {mask_data.shape}, affine:\n{affine}")

    # Smooth the binary mask for a smoother surface
    if smooth_sigma > 0:
        smoothed = gaussian_filter(mask_data, sigma=smooth_sigma)
    else:
        smoothed = mask_data

    # Get voxel spacing from affine
    spacing = np.abs(np.diag(affine[:3, :3]))
    if np.any(spacing == 0):
        spacing = np.array([1.0, 1.0, 1.0])

    logger.info(f"Voxel spacing: {spacing}")

    # Run marching cubes
    vertices, faces, normals, values = marching_cubes(
        smoothed,
        level=0.5,
        spacing=tuple(spacing),
        step_size=step_size,
    )

    logger.info(f"Marching cubes: {len(vertices)} vertices, {len(faces)} faces")

    # Transform vertices from voxel space to world (RAS) space
    # vertices from marching_cubes are in (index * spacing) space
    # We need to apply the affine translation (origin offset)
    origin = affine[:3, 3]
    vertices_world = vertices + origin

    # Compute bounds
    bounds_min = vertices_world.min(axis=0).tolist()
    bounds_max = vertices_world.max(axis=0).tolist()

    result = {
        "vertices": vertices_world.flatten().tolist(),
        "faces": faces.flatten().tolist(),
        "normals": normals.flatten().tolist(),
        "metadata": {
            "n_vertices": len(vertices_world),
            "n_faces": len(faces),
            "bounds": {
                "min": bounds_min,
                "max": bounds_max,
            },
            "spacing": spacing.tolist(),
            "source": Path(mask_path).name,
            "step_size": step_size,
            "smooth_sigma": smooth_sigma,
        },
    }

    # Cache the result
    if cache_dir:
        cache_path = Path(cache_dir) / f"brain_mesh_step{step_size}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f)
        logger.info(f"Cached brain mesh to {cache_path}")

    return result
