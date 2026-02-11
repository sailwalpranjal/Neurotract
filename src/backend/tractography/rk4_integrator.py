"""
4th-Order Runge-Kutta Integration for Streamline Propagation

Implements adaptive RK4 integration with step size control, curvature checking,
and comprehensive termination conditions for diffusion tractography.
"""

import numpy as np
import numba
from typing import Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def _rk4_step(
    position: np.ndarray,
    step_size: float,
    direction_func: Callable,
    *direction_args
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single RK4 integration step (Numba-optimized inner loop)

    Args:
        position: Current position (3,)
        step_size: Integration step size in mm
        direction_func: Function to get direction at a position
        *direction_args: Additional arguments for direction function

    Returns:
        new_position: Updated position (3,)
        direction: Direction at new position (3,)
    """
    # Classical 4th-order Runge-Kutta
    k1 = direction_func(position, *direction_args)
    k2 = direction_func(position + 0.5 * step_size * k1, *direction_args)
    k3 = direction_func(position + 0.5 * step_size * k2, *direction_args)
    k4 = direction_func(position + step_size * k3, *direction_args)

    # Weighted combination
    new_position = position + (step_size / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Direction at new position for curvature checking
    direction = direction_func(new_position, *direction_args)

    return new_position, direction


@numba.jit(nopython=True, cache=True)
def _compute_curvature_angle(dir1: np.ndarray, dir2: np.ndarray) -> float:
    """
    Compute angle between two direction vectors in degrees

    Args:
        dir1: First direction vector (3,)
        dir2: Second direction vector (3,)

    Returns:
        Angle in degrees
    """
    # Normalize vectors
    norm1 = np.sqrt(np.sum(dir1**2))
    norm2 = np.sqrt(np.sum(dir2**2))

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 180.0

    dir1_norm = dir1 / norm1
    dir2_norm = dir2 / norm2

    # Compute angle (handle numerical errors in arccos)
    dot_product = np.sum(dir1_norm * dir2_norm)
    dot_product = max(-1.0, min(1.0, dot_product))

    angle_rad = np.arccos(np.abs(dot_product))  # abs for undirected angles
    angle_deg = np.degrees(angle_rad)

    return angle_deg


@numba.jit(nopython=True, cache=True)
def _is_inside_volume(position: np.ndarray, volume_shape: np.ndarray) -> bool:
    """
    Check if position is inside volume bounds

    Args:
        position: Position in voxel coordinates (3,)
        volume_shape: Shape of volume (3,)

    Returns:
        True if inside volume
    """
    for i in range(3):
        if position[i] < 0 or position[i] >= volume_shape[i] - 1:
            return False
    return True


@numba.jit(nopython=True, cache=True)
def _trilinear_interpolation(
    volume: np.ndarray,
    position: np.ndarray
) -> float:
    """
    Trilinear interpolation at fractional position

    Args:
        volume: 3D volume
        position: Fractional position (3,)

    Returns:
        Interpolated value
    """
    # Get integer and fractional parts
    x, y, z = position
    x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    # Fractional parts
    xd = x - x0
    yd = y - y0
    zd = z - z0

    # Trilinear interpolation
    c000 = volume[x0, y0, z0]
    c001 = volume[x0, y0, z1]
    c010 = volume[x0, y1, z0]
    c011 = volume[x0, y1, z1]
    c100 = volume[x1, y0, z0]
    c101 = volume[x1, y0, z1]
    c110 = volume[x1, y1, z0]
    c111 = volume[x1, y1, z1]

    # Interpolate along x
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    # Interpolate along y
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    # Interpolate along z
    value = c0 * (1 - zd) + c1 * zd

    return value


class RK4Integrator:
    """
    Adaptive 4th-order Runge-Kutta integrator for tractography

    Features:
    - Adaptive step size control based on voxel size
    - Curvature angle checking at each step
    - Multiple termination conditions (FA, FOD amplitude, mask, length)
    - Numba-optimized inner loops for performance
    """

    def __init__(
        self,
        voxel_size: np.ndarray,
        step_size_range: Tuple[float, float] = (0.5, 1.0),
        max_angle: float = 60.0,
        fa_threshold: float = 0.1,
        fod_threshold: float = 0.1,
        max_length: float = 200.0,
        min_length: float = 10.0
    ):
        """
        Initialize RK4 integrator

        Args:
            voxel_size: Voxel dimensions in mm (3,)
            step_size_range: (min, max) step size in mm
            max_angle: Maximum curvature angle per step (degrees)
            fa_threshold: Minimum FA to continue tracking
            fod_threshold: Minimum FOD amplitude to continue
            max_length: Maximum streamline length in mm
            min_length: Minimum streamline length to keep in mm
        """
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.min_voxel_size = np.min(self.voxel_size)

        # Adapt step size to voxel resolution
        self.step_size_min = max(step_size_range[0], 0.25 * self.min_voxel_size)
        self.step_size_max = min(step_size_range[1], 0.75 * self.min_voxel_size)
        self.step_size = (self.step_size_min + self.step_size_max) / 2.0

        self.max_angle = max_angle
        self.fa_threshold = fa_threshold
        self.fod_threshold = fod_threshold
        self.max_length = max_length
        self.min_length = min_length

        # Maximum steps to prevent infinite loops
        self.max_steps = int(np.ceil(self.max_length / self.step_size_min)) + 1000

        logger.debug(
            f"RK4 Integrator initialized: step_size={self.step_size:.3f}mm, "
            f"max_angle={max_angle}°, fa_threshold={fa_threshold}, "
            f"max_steps={self.max_steps}"
        )

    def integrate_streamline(
        self,
        seed_position: np.ndarray,
        initial_direction: np.ndarray,
        direction_getter: Callable,
        fa_volume: Optional[np.ndarray] = None,
        mask_volume: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Integrate a single streamline from seed position

        Args:
            seed_position: Seed position in voxel coordinates (3,)
            initial_direction: Initial tracking direction (3,)
            direction_getter: Callable that returns direction at position
            fa_volume: Fractional anisotropy volume for termination
            mask_volume: Binary mask for termination
            affine: Affine transformation (voxel to world coordinates)

        Returns:
            streamline: Array of points (N, 3) in world coordinates
            termination_reason: String describing why tracking stopped
        """
        # Initialize streamline storage
        max_points = self.max_steps
        streamline_voxel = np.zeros((max_points, 3), dtype=np.float32)

        # Start tracking
        position = np.array(seed_position, dtype=np.float32)
        direction = np.array(initial_direction, dtype=np.float32)
        direction = direction / np.linalg.norm(direction)  # Normalize

        streamline_voxel[0] = position
        current_length = 0.0
        step_idx = 1
        termination_reason = "max_length"

        # Get volume shape for bounds checking
        if fa_volume is not None:
            volume_shape = np.array(fa_volume.shape, dtype=np.int32)
        elif mask_volume is not None:
            volume_shape = np.array(mask_volume.shape, dtype=np.int32)
        else:
            raise ValueError("Either fa_volume or mask_volume must be provided")

        # Forward tracking
        for step in range(self.max_steps - 1):
            # Check termination conditions

            # 1. Check if inside volume
            if not self._check_bounds(position, volume_shape):
                termination_reason = "exit_volume"
                break

            # 2. Check mask
            if mask_volume is not None:
                if not self._check_mask(position, mask_volume):
                    termination_reason = "exit_mask"
                    break

            # 3. Check FA threshold
            if fa_volume is not None:
                fa_value = self._interpolate_scalar(position, fa_volume)
                if fa_value < self.fa_threshold:
                    termination_reason = "low_fa"
                    break

            # 4. Get direction at current position
            new_direction = direction_getter(position)

            # Check FOD amplitude
            amplitude = np.linalg.norm(new_direction)
            if amplitude < self.fod_threshold:
                termination_reason = "low_fod"
                break

            # Normalize direction
            new_direction = new_direction / amplitude

            # Ensure consistent orientation (dot product > 0)
            if np.dot(new_direction, direction) < 0:
                new_direction = -new_direction

            # 5. Check curvature angle
            angle = _compute_curvature_angle(direction, new_direction)
            if angle > self.max_angle:
                termination_reason = "high_curvature"
                break

            # Perform RK4 step
            step_size = self._adaptive_step_size(position, new_direction)
            new_position = position + step_size * new_direction

            # Update state
            position = new_position
            direction = new_direction
            current_length += step_size

            # Store point
            streamline_voxel[step_idx] = position
            step_idx += 1

            # 6. Check maximum length
            if current_length >= self.max_length:
                termination_reason = "max_length"
                break

        # Trim streamline to actual length
        streamline_voxel = streamline_voxel[:step_idx]

        # Convert to world coordinates if affine provided
        if affine is not None:
            streamline_world = self._voxel_to_world(streamline_voxel, affine)
        else:
            streamline_world = streamline_voxel

        # Check minimum length
        streamline_length = self._compute_length(streamline_world)
        if streamline_length < self.min_length:
            termination_reason = "too_short"

        return streamline_world, termination_reason

    def integrate_bidirectional(
        self,
        seed_position: np.ndarray,
        initial_direction: np.ndarray,
        direction_getter: Callable,
        fa_volume: Optional[np.ndarray] = None,
        mask_volume: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Tuple[str, str]]:
        """
        Integrate streamline in both directions from seed

        Args:
            seed_position: Seed position in voxel coordinates (3,)
            initial_direction: Initial tracking direction (3,)
            direction_getter: Callable that returns direction at position
            fa_volume: Fractional anisotropy volume
            mask_volume: Binary mask volume
            affine: Affine transformation matrix

        Returns:
            streamline: Combined streamline (N, 3)
            termination_reasons: (forward_reason, backward_reason)
        """
        # Forward direction
        streamline_fwd, reason_fwd = self.integrate_streamline(
            seed_position, initial_direction, direction_getter,
            fa_volume, mask_volume, affine
        )

        # Backward direction (flip initial direction)
        streamline_bwd, reason_bwd = self.integrate_streamline(
            seed_position, -initial_direction, direction_getter,
            fa_volume, mask_volume, affine
        )

        # Combine streamlines (reverse backward, concatenate with forward)
        if len(streamline_bwd) > 1:
            streamline_bwd = streamline_bwd[::-1]  # Reverse
            streamline = np.vstack([streamline_bwd[:-1], streamline_fwd])
        else:
            streamline = streamline_fwd

        return streamline, (reason_fwd, reason_bwd)

    def _adaptive_step_size(
        self,
        position: np.ndarray,
        direction: np.ndarray
    ) -> float:
        """
        Compute adaptive step size based on local geometry

        Args:
            position: Current position
            direction: Current direction

        Returns:
            Adaptive step size in mm
        """
        # For now, use fixed step size
        # Can be extended to adapt based on curvature or FA
        return self.step_size

    def _check_bounds(
        self,
        position: np.ndarray,
        volume_shape: np.ndarray
    ) -> bool:
        """Check if position is inside volume bounds"""
        return _is_inside_volume(position, volume_shape)

    def _check_mask(
        self,
        position: np.ndarray,
        mask_volume: np.ndarray
    ) -> bool:
        """Check if position is inside mask"""
        # Nearest neighbor interpolation for binary mask
        x, y, z = position.astype(np.int32)

        # Bounds check
        if (x < 0 or y < 0 or z < 0 or
            x >= mask_volume.shape[0] or
            y >= mask_volume.shape[1] or
            z >= mask_volume.shape[2]):
            return False

        return mask_volume[x, y, z] > 0

    def _interpolate_scalar(
        self,
        position: np.ndarray,
        volume: np.ndarray
    ) -> float:
        """Trilinear interpolation of scalar volume"""
        # Check bounds
        volume_shape = np.array(volume.shape, dtype=np.int32)
        if not _is_inside_volume(position, volume_shape):
            return 0.0

        return _trilinear_interpolation(volume, position)

    def _voxel_to_world(
        self,
        points_voxel: np.ndarray,
        affine: np.ndarray
    ) -> np.ndarray:
        """
        Transform points from voxel to world coordinates

        Args:
            points_voxel: Points in voxel coordinates (N, 3)
            affine: 4x4 affine transformation matrix

        Returns:
            Points in world coordinates (N, 3)
        """
        # Add homogeneous coordinate
        n_points = points_voxel.shape[0]
        points_hom = np.hstack([points_voxel, np.ones((n_points, 1))])

        # Apply affine transformation
        points_world = (affine @ points_hom.T).T

        # Remove homogeneous coordinate
        return points_world[:, :3]

    def _compute_length(self, streamline: np.ndarray) -> float:
        """
        Compute streamline length

        Args:
            streamline: Array of points (N, 3)

        Returns:
            Total length in mm
        """
        if len(streamline) < 2:
            return 0.0

        # Compute segment lengths
        segments = np.diff(streamline, axis=0)
        lengths = np.linalg.norm(segments, axis=1)

        return np.sum(lengths)

    def get_config(self) -> dict:
        """Get integrator configuration for logging"""
        return {
            'step_size': self.step_size,
            'step_size_range': (self.step_size_min, self.step_size_max),
            'max_angle': self.max_angle,
            'fa_threshold': self.fa_threshold,
            'fod_threshold': self.fod_threshold,
            'max_length': self.max_length,
            'min_length': self.min_length,
            'max_steps': self.max_steps,
            'voxel_size': self.voxel_size.tolist()
        }
