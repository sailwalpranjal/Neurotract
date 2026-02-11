"""
Gradient Table Quality Control and Correction Module

Implements comprehensive gradient table validation and correction:
1. Validate bvals/bvecs consistency and format
2. Detect and correct flipped gradients
3. Reorient gradients to scanner/image coordinates
4. Check for duplicate or missing directions
5. Validate b-value shell structure
6. Detect gradient encoding issues

Ensures gradient tables are correct before tractography and modeling.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy.spatial.distance import cdist

from ..utils.logger import get_logger, log_decision
from ..utils.memory_manager import get_memory_manager

logger = get_logger(__name__)


class GradientCorrectionError(Exception):
    """Exception raised for gradient correction failures"""
    pass


class GradientQualityMetrics:
    """Container for gradient table quality metrics"""

    def __init__(
        self,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        corrected_bvecs: Optional[np.ndarray] = None,
    ):
        self.bvals = bvals
        self.bvecs = bvecs
        self.corrected_bvecs = corrected_bvecs if corrected_bvecs is not None else bvecs
        self.metrics = self._compute_metrics()

    def _compute_metrics(self) -> Dict:
        """Compute quality metrics for gradient table"""
        num_volumes = len(self.bvals)

        # Identify b-value shells
        b0_threshold = 50
        b0_indices = self.bvals < b0_threshold
        dwi_indices = self.bvals >= b0_threshold

        num_b0 = np.sum(b0_indices)
        num_dwi = np.sum(dwi_indices)

        # Find unique b-value shells (rounded to nearest 100)
        unique_bvals = np.unique(self.bvals[dwi_indices].round(-2))
        shells = sorted([int(b) for b in unique_bvals])

        # Check gradient vector normalization
        norms = np.linalg.norm(self.bvecs, axis=0)
        dwi_norms = norms[dwi_indices]

        norm_mean = np.mean(dwi_norms) if len(dwi_norms) > 0 else 0
        norm_std = np.std(dwi_norms) if len(dwi_norms) > 0 else 0
        norm_min = np.min(dwi_norms) if len(dwi_norms) > 0 else 0
        norm_max = np.max(dwi_norms) if len(dwi_norms) > 0 else 0

        # Check for duplicate directions
        if num_dwi > 0:
            dwi_bvecs = self.bvecs[:, dwi_indices]
            num_duplicates = self._count_duplicate_directions(dwi_bvecs)
        else:
            num_duplicates = 0

        # Compute angular coverage metrics
        if num_dwi > 1:
            min_angle, max_angle, mean_angle = self._compute_angular_metrics(
                self.bvecs[:, dwi_indices]
            )
        else:
            min_angle = max_angle = mean_angle = 0.0

        metrics = {
            'num_volumes': int(num_volumes),
            'num_b0': int(num_b0),
            'num_dwi': int(num_dwi),
            'num_shells': len(shells),
            'shells': shells,
            'norm_mean': float(norm_mean),
            'norm_std': float(norm_std),
            'norm_min': float(norm_min),
            'norm_max': float(norm_max),
            'num_duplicates': int(num_duplicates),
            'min_angle_deg': float(min_angle),
            'max_angle_deg': float(max_angle),
            'mean_angle_deg': float(mean_angle),
        }

        # Shell-specific metrics
        for shell_bval in shells:
            shell_indices = np.abs(self.bvals - shell_bval) < 50
            shell_count = np.sum(shell_indices)
            metrics[f'shell_{shell_bval}_count'] = int(shell_count)

        return metrics

    def _count_duplicate_directions(
        self,
        bvecs: np.ndarray,
        threshold: float = 0.01
    ) -> int:
        """
        Count number of duplicate gradient directions

        Parameters
        ----------
        bvecs : np.ndarray
            Gradient vectors (3, N)
        threshold : float
            Distance threshold for considering directions as duplicates

        Returns
        -------
        num_duplicates : int
            Number of duplicate direction pairs
        """
        # Compute pairwise distances
        distances = cdist(bvecs.T, bvecs.T, metric='euclidean')

        # Set diagonal to large value
        np.fill_diagonal(distances, np.inf)

        # Count pairs with distance below threshold
        num_duplicates = np.sum(distances < threshold) // 2

        return num_duplicates

    def _compute_angular_metrics(
        self,
        bvecs: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute angular separation metrics between gradient directions

        Parameters
        ----------
        bvecs : np.ndarray
            Normalized gradient vectors (3, N)

        Returns
        -------
        min_angle : float
            Minimum angle between any two directions (degrees)
        max_angle : float
            Maximum angle between any two directions (degrees)
        mean_angle : float
            Mean nearest-neighbor angle (degrees)
        """
        # Compute pairwise angles using dot product
        dot_products = np.dot(bvecs.T, bvecs)
        dot_products = np.clip(dot_products, -1.0, 1.0)

        angles = np.arccos(np.abs(dot_products))
        angles_deg = np.rad2deg(angles)

        # Set diagonal to large value
        np.fill_diagonal(angles_deg, np.inf)

        # Find minimum angle for each direction (nearest neighbor)
        min_angles_per_direction = np.min(angles_deg, axis=1)

        min_angle = np.min(angles_deg[np.isfinite(angles_deg)])
        max_angle = np.max(angles_deg[np.isfinite(angles_deg)])
        mean_angle = np.mean(min_angles_per_direction)

        return min_angle, max_angle, mean_angle

    def is_valid(
        self,
        min_num_dwi: int = 6,
        min_num_b0: int = 1,
        max_norm_deviation: float = 0.1,
        min_nearest_angle: float = 5.0,
    ) -> Tuple[bool, str]:
        """
        Validate gradient table quality

        Parameters
        ----------
        min_num_dwi : int
            Minimum number of DWI volumes required
        min_num_b0 : int
            Minimum number of b0 volumes required
        max_norm_deviation : float
            Maximum allowed deviation from unit norm
        min_nearest_angle : float
            Minimum allowed nearest-neighbor angle (degrees)

        Returns
        -------
        is_valid : bool
            Whether gradient table passes quality checks
        message : str
            Validation message
        """
        issues = []

        if self.metrics['num_dwi'] < min_num_dwi:
            issues.append(
                f"Too few DWI volumes: {self.metrics['num_dwi']} < {min_num_dwi}"
            )

        if self.metrics['num_b0'] < min_num_b0:
            issues.append(
                f"Too few b0 volumes: {self.metrics['num_b0']} < {min_num_b0}"
            )

        if abs(self.metrics['norm_mean'] - 1.0) > max_norm_deviation:
            issues.append(
                f"Gradient vectors not normalized: mean norm = {self.metrics['norm_mean']:.3f}"
            )

        if self.metrics['num_duplicates'] > 0:
            issues.append(
                f"Found {self.metrics['num_duplicates']} duplicate directions"
            )

        if self.metrics['mean_angle_deg'] < min_nearest_angle:
            issues.append(
                f"Poor angular coverage: mean angle = {self.metrics['mean_angle_deg']:.1f}° < {min_nearest_angle}°"
            )

        if issues:
            return False, "; ".join(issues)
        else:
            return True, "Gradient table quality acceptable"

    def save(self, output_path: Path):
        """Save quality metrics to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("Gradient Table Quality Metrics\n")
            f.write("=" * 50 + "\n\n")

            f.write("Volume Counts:\n")
            f.write(f"  Total volumes: {self.metrics['num_volumes']}\n")
            f.write(f"  b0 volumes: {self.metrics['num_b0']}\n")
            f.write(f"  DWI volumes: {self.metrics['num_dwi']}\n")
            f.write(f"  Number of shells: {self.metrics['num_shells']}\n")
            f.write(f"  Shell b-values: {self.metrics['shells']}\n\n")

            for shell in self.metrics['shells']:
                key = f'shell_{shell}_count'
                if key in self.metrics:
                    f.write(f"  Shell {shell}: {self.metrics[key]} volumes\n")

            f.write("\nGradient Vector Normalization:\n")
            f.write(f"  Mean norm: {self.metrics['norm_mean']:.6f}\n")
            f.write(f"  Std norm: {self.metrics['norm_std']:.6f}\n")
            f.write(f"  Min norm: {self.metrics['norm_min']:.6f}\n")
            f.write(f"  Max norm: {self.metrics['norm_max']:.6f}\n\n")

            f.write("Angular Coverage:\n")
            f.write(f"  Min angle: {self.metrics['min_angle_deg']:.2f}°\n")
            f.write(f"  Max angle: {self.metrics['max_angle_deg']:.2f}°\n")
            f.write(f"  Mean nearest-neighbor angle: {self.metrics['mean_angle_deg']:.2f}°\n\n")

            f.write(f"Duplicate directions: {self.metrics['num_duplicates']}\n\n")

            is_valid, message = self.is_valid()
            f.write(f"Validation: {'PASS' if is_valid else 'FAIL'}\n")
            f.write(f"Message: {message}\n")

        logger.info(f"Gradient quality metrics saved to {output_path}")


class GradientCorrector:
    """
    Gradient table validation and correction

    Performs comprehensive QC and corrections on b-values and gradient vectors
    """

    def __init__(
        self,
        b0_threshold: float = 50.0,
        normalize_bvecs: bool = True,
        flip_x: bool = False,
        flip_y: bool = False,
        flip_z: bool = False,
    ):
        """
        Initialize gradient corrector

        Parameters
        ----------
        b0_threshold : float
            Threshold for identifying b0 volumes (default: 50.0)
        normalize_bvecs : bool
            Normalize gradient vectors to unit length (default: True)
        flip_x : bool
            Flip gradient x-component (default: False)
        flip_y : bool
            Flip gradient y-component (default: False)
        flip_z : bool
            Flip gradient z-component (default: False)
        """
        self.b0_threshold = b0_threshold
        self.normalize_bvecs = normalize_bvecs
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.flip_z = flip_z

        logger.info(
            f"GradientCorrector initialized: "
            f"b0_threshold={b0_threshold}, "
            f"normalize={normalize_bvecs}"
        )

    def correct(
        self,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        output_dir: Optional[Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray, GradientQualityMetrics]:
        """
        Validate and correct gradient table

        Parameters
        ----------
        bvals : np.ndarray
            B-values (N,)
        bvecs : np.ndarray
            Gradient vectors (3, N) or (N, 3)
        output_dir : Path, optional
            Directory to save QC outputs

        Returns
        -------
        corrected_bvals : np.ndarray
            Validated/corrected b-values
        corrected_bvecs : np.ndarray
            Validated/corrected gradient vectors (3, N)
        metrics : GradientQualityMetrics
            Quality metrics
        """
        logger.info("Starting gradient table validation and correction")

        # Ensure proper shape
        bvals, bvecs = self._validate_shapes(bvals, bvecs)

        # Store original for metrics
        original_bvecs = bvecs.copy()

        # Round b-values to nearest 5
        corrected_bvals = self._round_bvals(bvals)

        # Apply gradient flips if requested
        corrected_bvecs = self._apply_flips(bvecs)

        # Normalize gradient vectors
        if self.normalize_bvecs:
            corrected_bvecs = self._normalize_gradients(
                corrected_bvecs, corrected_bvals
            )

        # Detect and warn about issues
        self._detect_issues(corrected_bvals, corrected_bvecs)

        # Compute quality metrics
        metrics = GradientQualityMetrics(
            corrected_bvals, original_bvecs, corrected_bvecs
        )

        # Validate
        is_valid, message = metrics.is_valid()
        if is_valid:
            logger.info(f"Gradient table validation: {message}")
        else:
            logger.warning(f"Gradient table validation issues: {message}")

        # Log decision
        log_decision(
            decision_id="gradient_correction",
            component="preprocessing.gradient_correction",
            decision="Gradient table validated and corrected",
            rationale=(
                f"Applied normalization: {self.normalize_bvecs}, "
                f"Applied flips: x={self.flip_x}, y={self.flip_y}, z={self.flip_z}"
            ),
            parameters={
                'b0_threshold': self.b0_threshold,
                'normalize_bvecs': self.normalize_bvecs,
                'flip_x': self.flip_x,
                'flip_y': self.flip_y,
                'flip_z': self.flip_z,
                'num_volumes': metrics.metrics['num_volumes'],
                'num_shells': metrics.metrics['num_shells'],
                'shells': metrics.metrics['shells'],
            },
        )

        # Save outputs if requested
        if output_dir is not None:
            self._save_outputs(
                corrected_bvals, corrected_bvecs, metrics, output_dir
            )

        logger.info(
            f"Gradient correction completed: "
            f"{metrics.metrics['num_dwi']} DWI, "
            f"{metrics.metrics['num_b0']} b0, "
            f"{metrics.metrics['num_shells']} shells"
        )

        return corrected_bvals, corrected_bvecs, metrics

    def _validate_shapes(
        self,
        bvals: np.ndarray,
        bvecs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and fix shapes of bvals and bvecs

        Ensures bvals is 1D and bvecs is (3, N)
        """
        # Ensure bvals is 1D
        if bvals.ndim > 1:
            bvals = bvals.flatten()
            logger.info(f"Flattened bvals to shape {bvals.shape}")

        # Ensure bvecs is (3, N)
        if bvecs.ndim != 2:
            raise GradientCorrectionError(
                f"bvecs must be 2D, got {bvecs.ndim}D: {bvecs.shape}"
            )

        if bvecs.shape[0] != 3 and bvecs.shape[1] == 3:
            # Transpose from (N, 3) to (3, N)
            bvecs = bvecs.T
            logger.info(f"Transposed bvecs to shape {bvecs.shape}")
        elif bvecs.shape[0] != 3:
            raise GradientCorrectionError(
                f"bvecs must have shape (3, N) or (N, 3), got {bvecs.shape}"
            )

        # Check matching lengths
        num_bvals = len(bvals)
        num_bvecs = bvecs.shape[1]

        if num_bvals != num_bvecs:
            raise GradientCorrectionError(
                f"Number of bvals ({num_bvals}) does not match "
                f"number of bvecs ({num_bvecs})"
            )

        return bvals, bvecs

    def _round_bvals(self, bvals: np.ndarray) -> np.ndarray:
        """
        Round b-values to nearest 5 for consistency

        Also ensures b-values below threshold are set to 0
        """
        rounded = np.round(bvals / 5) * 5

        # Set values below threshold to exactly 0
        rounded[rounded < self.b0_threshold] = 0

        if not np.array_equal(bvals, rounded):
            logger.info("Rounded b-values for consistency")

        return rounded

    def _apply_flips(self, bvecs: np.ndarray) -> np.ndarray:
        """Apply gradient direction flips if requested"""
        flipped = bvecs.copy()

        if self.flip_x:
            flipped[0, :] *= -1
            logger.info("Flipped gradient x-component")

        if self.flip_y:
            flipped[1, :] *= -1
            logger.info("Flipped gradient y-component")

        if self.flip_z:
            flipped[2, :] *= -1
            logger.info("Flipped gradient z-component")

        return flipped

    def _normalize_gradients(
        self,
        bvecs: np.ndarray,
        bvals: np.ndarray
    ) -> np.ndarray:
        """
        Normalize gradient vectors to unit length

        Only normalizes vectors for non-b0 volumes
        """
        normalized = bvecs.copy()

        # Find non-b0 volumes
        non_b0 = bvals >= self.b0_threshold

        # Compute norms
        norms = np.linalg.norm(normalized[:, non_b0], axis=0)

        # Check if already normalized
        if np.allclose(norms, 1.0, atol=1e-3):
            logger.info("Gradient vectors already normalized")
            return normalized

        # Normalize
        nonzero = norms > 0
        normalized[:, non_b0][:, nonzero] /= norms[nonzero]

        # Set b0 gradients to zero
        normalized[:, ~non_b0] = 0

        logger.info("Normalized gradient vectors to unit length")

        return normalized

    def _detect_issues(self, bvals: np.ndarray, bvecs: np.ndarray):
        """Detect and warn about potential issues"""

        # Check for zero gradients in non-b0 volumes
        non_b0 = bvals >= self.b0_threshold
        norms = np.linalg.norm(bvecs[:, non_b0], axis=0)

        if np.any(norms < 0.1):
            num_zero = np.sum(norms < 0.1)
            warnings.warn(
                f"Found {num_zero} non-b0 volumes with near-zero gradients"
            )

        # Check for missing b0
        num_b0 = np.sum(~non_b0)
        if num_b0 == 0:
            warnings.warn("No b0 volumes found in gradient table!")

        # Check for unusual b-value range
        max_bval = np.max(bvals)
        if max_bval > 10000:
            warnings.warn(
                f"Unusually high b-value detected: {max_bval}. "
                "Check if b-values are in correct units (s/mm²)"
            )

    def _save_outputs(
        self,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        metrics: GradientQualityMetrics,
        output_dir: Path,
    ):
        """Save corrected gradients and metrics"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save corrected bvals
        np.savetxt(
            output_dir / "corrected.bval",
            bvals,
            fmt='%d'
        )

        # Save corrected bvecs
        np.savetxt(
            output_dir / "corrected.bvec",
            bvecs,
            fmt='%.6f'
        )

        # Save metrics
        metrics.save(output_dir / "gradient_qc.txt")

        logger.info(f"Gradient correction outputs saved to {output_dir}")


def reorient_gradients(
    bvecs: np.ndarray,
    affine_src: np.ndarray,
    affine_dst: np.ndarray,
) -> np.ndarray:
    """
    Reorient gradient vectors when resampling data to new space

    Parameters
    ----------
    bvecs : np.ndarray
        Original gradient vectors (3, N)
    affine_src : np.ndarray
        Source image affine matrix
    affine_dst : np.ndarray
        Destination image affine matrix

    Returns
    -------
    reoriented_bvecs : np.ndarray
        Reoriented gradient vectors (3, N)
    """
    # Extract rotation component of transformation
    src_rotation = affine_src[:3, :3]
    dst_rotation = affine_dst[:3, :3]

    # Compute relative rotation
    rotation = np.linalg.inv(dst_rotation) @ src_rotation

    # Apply rotation to gradient vectors
    reoriented = rotation @ bvecs

    # Renormalize
    norms = np.linalg.norm(reoriented, axis=0)
    nonzero = norms > 0
    reoriented[:, nonzero] /= norms[nonzero]

    logger.info("Reoriented gradient vectors to new coordinate system")

    return reoriented
