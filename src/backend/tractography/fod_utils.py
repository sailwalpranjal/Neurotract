"""
FOD Peak Finding Utilities
===========================

Utilities for finding peaks in Fiber Orientation Distribution (FOD) functions
represented as spherical harmonics.

"""

import numpy as np
from typing import List, Tuple, Optional
import logging

from dipy.data import get_sphere
from dipy.reconst.shm import real_sym_sh_basis

logger = logging.getLogger(__name__)


class FODPeakFinder:
    """
    Find peaks in FOD represented as spherical harmonics.

    Uses discrete sampling on a sphere followed by non-maximum suppression
    to identify local maxima representing fiber orientations.
    """

    def __init__(
        self,
        sphere_vertices: Optional[np.ndarray] = None,
        relative_peak_threshold: float = 0.5,
        min_separation_angle: float = 25.0,
        max_peaks: int = 5
    ):
        """
        Initialize FOD peak finder.

        Parameters
        ----------
        sphere_vertices : ndarray of shape (n_vertices, 3), optional
            Unit sphere vertices for sampling. If None, uses default 724-vertex sphere.
        relative_peak_threshold : float, default=0.5
            Minimum relative amplitude (as fraction of max) to consider a peak.
        min_separation_angle : float, default=25.0
            Minimum angular separation between peaks in degrees.
        max_peaks : int, default=5
            Maximum number of peaks to extract per voxel.
        """
        self.relative_peak_threshold = relative_peak_threshold
        self.min_separation_angle = np.deg2rad(min_separation_angle)
        self.max_peaks = max_peaks

        # Create or use provided sphere
        if sphere_vertices is None:
            # Use DIPY's deterministic, uniformly-distributed sphere
            sphere = get_sphere(name='repulsion724')
            self.sphere_vertices = sphere.vertices
        else:
            self.sphere_vertices = sphere_vertices

        self.n_vertices = len(self.sphere_vertices)

        # Precompute spherical harmonic basis
        self.sh_basis = None  # Computed on first use

    def _compute_sh_basis(self, sh_order: int) -> np.ndarray:
        """
        Compute spherical harmonic basis matrix for the sphere vertices using DIPY.

        Parameters
        ----------
        sh_order : int
            Maximum SH order (must be even)

        Returns
        -------
        basis : ndarray of shape (n_vertices, n_coefficients)
            SH basis evaluated at each vertex
        """
        if sh_order % 2 != 0:
            raise ValueError("SH order must be even")

        # Convert vertices to spherical coordinates
        x, y, z = self.sphere_vertices.T
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)  # Polar angle from z-axis
        phi = np.arctan2(y, x)    # Azimuthal angle

        # Use DIPY's real symmetric SH basis
        basis, m_values, l_values = real_sym_sh_basis(sh_order, theta, phi)

        return basis

    def find_peaks(
        self,
        fod_coeffs: np.ndarray,
        sh_order: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find peaks in a single-voxel FOD.

        Parameters
        ----------
        fod_coeffs : ndarray of shape (n_coefficients,)
            SH coefficients of the FOD
        sh_order : int, optional
            SH order. If None, inferred from number of coefficients.

        Returns
        -------
        peak_directions : ndarray of shape (n_peaks, 3)
            Unit vectors pointing to peak directions
        peak_values : ndarray of shape (n_peaks,)
            FOD amplitudes at peaks
        """
        # Infer SH order if not provided
        if sh_order is None:
            n_coeffs = len(fod_coeffs)
            sh_order = int((-3 + np.sqrt(1 + 8 * n_coeffs)) / 2)
            if sh_order % 2 != 0:
                sh_order -= 1

        # Compute basis if needed
        if self.sh_basis is None or self.sh_basis.shape[1] != len(fod_coeffs):
            self.sh_basis = self._compute_sh_basis(sh_order)

        # Evaluate FOD on sphere
        fod_on_sphere = self.sh_basis @ fod_coeffs

        # Find local maxima using non-maximum suppression
        peaks = []
        peak_indices = []

        # Sort vertices by FOD amplitude (descending)
        sorted_indices = np.argsort(fod_on_sphere)[::-1]

        max_fod = fod_on_sphere[sorted_indices[0]]
        threshold = max_fod * self.relative_peak_threshold

        for idx in sorted_indices:
            fod_val = fod_on_sphere[idx]

            # Stop if below threshold
            if fod_val < threshold:
                break

            # Check if too close to existing peaks
            vertex = self.sphere_vertices[idx]
            too_close = False

            for peak_idx in peak_indices:
                peak_vertex = self.sphere_vertices[peak_idx]
                # Angular distance
                cos_angle = np.clip(np.dot(vertex, peak_vertex), -1, 1)
                angle = np.arccos(np.abs(cos_angle))  # abs handles antipodal symmetry

                if angle < self.min_separation_angle:
                    too_close = True
                    break

            if not too_close:
                peaks.append((vertex, fod_val))
                peak_indices.append(idx)

                if len(peaks) >= self.max_peaks:
                    break

        if not peaks:
            # No peaks found - return zero direction
            return np.zeros((1, 3)), np.array([0.0])

        # Convert to arrays
        peak_directions = np.array([p[0] for p in peaks])
        peak_values = np.array([p[1] for p in peaks])

        return peak_directions, peak_values

    def precompute_peaks_volume(
        self,
        fod_volume: np.ndarray,
        mask: np.ndarray,
        sh_order: Optional[int] = None,
        batch_size: int = 10000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-compute primary peak directions for all voxels.

        This is much faster than evaluating peaks on-the-fly during tracking,
        as the SH basis evaluation happens once per voxel rather than once
        per tracking step.

        Parameters
        ----------
        fod_volume : ndarray of shape (nx, ny, nz, n_coefficients)
            FOD SH coefficients at each voxel
        mask : ndarray of shape (nx, ny, nz)
            Binary mask of voxels to process
        sh_order : int, optional
            SH order. If None, inferred from number of coefficients.
        batch_size : int, default=10000
            Number of voxels to process per batch (memory control)

        Returns
        -------
        peak_dirs : ndarray of shape (nx, ny, nz, 3)
            Primary peak direction at each voxel
        peak_values : ndarray of shape (nx, ny, nz)
            FOD amplitude at primary peak
        """
        shape = fod_volume.shape[:3]
        n_coeffs = fod_volume.shape[-1]

        if sh_order is None:
            sh_order = int((-3 + np.sqrt(1 + 8 * n_coeffs)) / 2)
            if sh_order % 2 != 0:
                sh_order -= 1

        # Pre-compute SH basis once
        if self.sh_basis is None or self.sh_basis.shape[1] != n_coeffs:
            self.sh_basis = self._compute_sh_basis(sh_order)

        mask_indices = np.where(mask)
        n_voxels = len(mask_indices[0])

        logger.info(f"Pre-computing peak directions for {n_voxels} voxels...")

        peak_dirs = np.zeros(shape + (3,), dtype=np.float32)
        peak_values = np.zeros(shape, dtype=np.float32)

        # Process in batches for memory efficiency
        for start in range(0, n_voxels, batch_size):
            end = min(start + batch_size, n_voxels)
            batch_x = mask_indices[0][start:end]
            batch_y = mask_indices[1][start:end]
            batch_z = mask_indices[2][start:end]

            # Get FOD coefficients for batch
            batch_fod = fod_volume[batch_x, batch_y, batch_z]  # (batch, n_coeffs)

            # Evaluate FOD on sphere: (batch, n_vertices)
            batch_on_sphere = batch_fod @ self.sh_basis.T

            # Find primary peak for each voxel
            max_idx = np.argmax(batch_on_sphere, axis=1)
            max_val = batch_on_sphere[np.arange(len(max_idx)), max_idx]

            # Store valid peaks (vectorized)
            valid = max_val > 0
            valid_x = batch_x[valid]
            valid_y = batch_y[valid]
            valid_z = batch_z[valid]
            peak_dirs[valid_x, valid_y, valid_z] = self.sphere_vertices[max_idx[valid]]
            peak_values[valid_x, valid_y, valid_z] = max_val[valid]

            if end % (batch_size * 5) == 0 or end == n_voxels:
                logger.info(f"  Pre-computed {end}/{n_voxels} voxels...")

        n_valid = np.count_nonzero(peak_values)
        logger.info(f"Peak pre-computation complete: {n_valid} voxels with valid peaks")

        return peak_dirs, peak_values

    def create_direction_getter(
        self,
        fod_volume: np.ndarray,
        sh_order: Optional[int] = None,
        use_primary_only: bool = True
    ):
        """
        Create a direction getter function for tractography.

        Parameters
        ----------
        fod_volume : ndarray of shape (nx, ny, nz, n_coefficients)
            FOD coefficients at each voxel
        sh_order : int, optional
            SH order
        use_primary_only : bool, default=True
            If True, return only the primary (strongest) peak.
            If False, randomly sample from peaks weighted by amplitude.

        Returns
        -------
        direction_getter : callable
            Function that takes position (x, y, z) and returns direction (dx, dy, dz)
        """
        shape = fod_volume.shape[:3]

        def direction_getter(position: np.ndarray) -> np.ndarray:
            """Get tracking direction at a position"""
            # Round to nearest voxel
            x, y, z = np.round(position).astype(int)

            # Check bounds
            if (x < 0 or x >= shape[0] or
                y < 0 or y >= shape[1] or
                z < 0 or z >= shape[2]):
                return np.array([0.0, 0.0, 0.0])

            # Get FOD coefficients at voxel
            fod_coeffs = fod_volume[x, y, z]

            # Check if zero
            if np.all(fod_coeffs == 0):
                return np.array([0.0, 0.0, 0.0])

            # Find peaks
            peak_dirs, peak_vals = self.find_peaks(fod_coeffs, sh_order)

            if len(peak_dirs) == 0 or peak_vals[0] == 0:
                return np.array([0.0, 0.0, 0.0])

            if use_primary_only:
                # Return primary peak
                return peak_dirs[0]
            else:
                # Randomly sample from peaks weighted by amplitude
                weights = peak_vals / np.sum(peak_vals)
                idx = np.random.choice(len(peak_dirs), p=weights)
                return peak_dirs[idx]

        return direction_getter


def create_precomputed_direction_getter(
    peak_dirs_volume: np.ndarray,
    peak_values_volume: Optional[np.ndarray] = None
):
    """
    Create a fast direction getter from pre-computed peak directions.

    This is O(1) per lookup vs O(n_vertices * n_coefficients) for on-the-fly
    computation, resulting in ~100-1000x speedup during tractography.

    Parameters
    ----------
    peak_dirs_volume : ndarray of shape (nx, ny, nz, 3)
        Pre-computed primary peak directions
    peak_values_volume : ndarray of shape (nx, ny, nz), optional
        Peak amplitudes (used for FOD threshold checking)

    Returns
    -------
    direction_getter : callable
        Fast direction getter function for tractography
    """
    shape = peak_dirs_volume.shape[:3]

    def direction_getter(position: np.ndarray) -> np.ndarray:
        """Get tracking direction at a position via pre-computed lookup"""
        x, y, z = np.round(position).astype(int)

        if (x < 0 or x >= shape[0] or
            y < 0 or y >= shape[1] or
            z < 0 or z >= shape[2]):
            return np.array([0.0, 0.0, 0.0])

        return peak_dirs_volume[x, y, z].copy()

    return direction_getter


def create_fod_direction_getter(
    fod_volume: np.ndarray,
    sh_order: Optional[int] = None,
    relative_peak_threshold: float = 0.5,
    min_separation_angle: float = 25.0,
    use_primary_only: bool = True
):
    """
    Create an optimized FOD direction getter with pre-computed peaks.

    Pre-computes peak directions for all voxels upfront, then returns
    a fast lookup-based direction getter for tractography.

    Parameters
    ----------
    fod_volume : ndarray of shape (nx, ny, nz, n_coefficients)
        FOD SH coefficients at each voxel
    sh_order : int, optional
        SH order
    relative_peak_threshold : float, default=0.5
        Relative peak threshold
    min_separation_angle : float, default=25.0
        Minimum angular separation in degrees
    use_primary_only : bool, default=True
        Use only primary peak

    Returns
    -------
    direction_getter : callable
        Direction getter function for tractography
    """
    peak_finder = FODPeakFinder(
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle
    )

    # Pre-compute peaks for all non-zero voxels
    mask = np.any(fod_volume != 0, axis=-1)
    peak_dirs, peak_values = peak_finder.precompute_peaks_volume(
        fod_volume, mask, sh_order
    )

    return create_precomputed_direction_getter(peak_dirs, peak_values)
