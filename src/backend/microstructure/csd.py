"""
Constrained Spherical Deconvolution (CSD) Module

Implements spherical harmonic decomposition and fiber orientation distribution (FOD)
estimation using multi-tissue response functions.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import logging
from dipy.reconst.shm import real_sym_sh_basis

logger = logging.getLogger(__name__)


class SphericalHarmonics:
    """Spherical harmonic basis functions using DIPY"""

    def __init__(self, max_order: int = 8):
        """
        Initialize SH basis

        Args:
            max_order: Maximum SH order (must be even, typically 6-12)
        """
        if max_order % 2 != 0:
            raise ValueError("max_order must be even")

        self.max_order = max_order
        self.n_coeffs = int((max_order + 1) * (max_order + 2) / 2)

        logger.info(f"Initialized SH basis: lmax={max_order}, n_coeffs={self.n_coeffs}")

    def get_order_from_ncoeffs(self, n_coeffs: int) -> int:
        """Get max order from number of coefficients"""
        return int((-3 + np.sqrt(1 + 8 * n_coeffs)) / 2)

    def sh_matrix(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Build real spherical harmonic matrix for given directions using DIPY

        Args:
            theta: Elevation angles (N,) in radians [0, pi]
            phi: Azimuth angles (N,) in radians [0, 2*pi]

        Returns:
            SH matrix of shape (N, n_coeffs)
        """
        B_real, m_values, l_values = real_sym_sh_basis(self.max_order, theta, phi)
        return B_real


class ResponseFunction:
    """
    Multi-tissue response function estimation

    Uses Dhollander approach for WM/GM/CSF separation or single-tissue for simpler data.
    """

    def __init__(self, method: str = "dhollander"):
        """
        Initialize response function estimator

        Args:
            method: 'dhollander' for multi-tissue or 'tournier' for single-tissue
        """
        self.method = method
        logger.info(f"Response function method: {method}")

    def estimate(
        self,
        dwi_data: np.ndarray,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        fa_map: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Estimate response functions for different tissue types

        Args:
            dwi_data: 4D diffusion data (x, y, z, volumes)
            bvals: b-values
            bvecs: gradient directions
            fa_map: FA map for tissue classification (computed if None)
            mask: Brain mask

        Returns:
            Dictionary with response functions for WM, GM, CSF
        """
        logger.info("Estimating response functions...")

        if mask is None:
            mask = np.ones(dwi_data.shape[:3], dtype=bool)

        # Simple single-shell response for now
        # In production, this would implement full Dhollander algorithm
        if self.method == "dhollander":
            return self._estimate_dhollander(dwi_data, bvals, bvecs, fa_map, mask)
        else:
            return self._estimate_single_tissue(dwi_data, bvals, bvecs, fa_map, mask)

    def _estimate_single_tissue(
        self,
        dwi_data: np.ndarray,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        fa_map: Optional[np.ndarray],
        mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Estimate single-tissue (WM) response"""

        # Compute FA map if not provided
        if fa_map is None:
            from .dti import DTIModel
            dti = DTIModel(bvals, bvecs)
            dti_results = dti.fit(dwi_data, mask)
            fa_map = dti_results['fa']

        # Try progressively lower FA thresholds to find WM voxels
        fa_thresholds = [0.7, 0.5, 0.3, 0.2, 0.1]
        n_wm = 0
        used_threshold = 0.1

        for threshold in fa_thresholds:
            wm_mask = (fa_map > threshold) & mask
            n_wm = int(np.sum(wm_mask))
            if n_wm >= 10:
                used_threshold = threshold
                break
            logger.warning(f"Only {n_wm} voxels with FA > {threshold}, trying lower threshold")

        # If still not enough, use top 1% FA voxels
        if n_wm < 10:
            fa_values = fa_map[mask]
            if len(fa_values) > 0:
                percentile_threshold = np.percentile(fa_values[fa_values > 0], 99)
                wm_mask = (fa_map > percentile_threshold) & mask
                n_wm = int(np.sum(wm_mask))
                used_threshold = float(percentile_threshold)
                logger.info(f"Using top 1% FA voxels (threshold={used_threshold:.3f})")

        logger.info(f"Selected {n_wm} WM voxels for response estimation (FA > {used_threshold:.3f})")

        if n_wm == 0:
            logger.warning("No WM voxels found, using mean signal from all masked voxels")
            wm_signals = dwi_data[mask]
        else:
            wm_signals = dwi_data[wm_mask]

        response_wm = np.median(wm_signals, axis=0)

        # Handle NaN values
        response_wm = np.nan_to_num(response_wm, nan=0.0)

        return {
            'wm': response_wm,
            'n_voxels': int(n_wm),
            'fa_threshold': used_threshold
        }

    def _estimate_dhollander(
        self,
        dwi_data: np.ndarray,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        fa_map: Optional[np.ndarray],
        mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Estimate multi-tissue response (WM/GM/CSF)

        Simplified implementation - production would use full Dhollander algorithm
        """
        # For now, fall back to single tissue
        logger.info("Multi-tissue response: using simplified single-tissue approach")
        return self._estimate_single_tissue(dwi_data, bvals, bvecs, fa_map, mask)


class CSDModel:
    """
    Constrained Spherical Deconvolution for FOD estimation
    """

    def __init__(
        self,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        sh_order: Optional[int] = None,
        lambda_: float = 1.0,
        tau: float = 0.1
    ):
        """
        Initialize CSD model

        Args:
            bvals: b-values (N,)
            bvecs: gradient directions (N, 3)
            sh_order: Maximum SH order (auto-determined from bmax if None)
            lambda_: Regularization parameter
            tau: Constraint threshold for negative lobes
        """
        self.bvals = np.array(bvals)
        self.bvecs = np.array(bvecs)

        # Ensure bvecs is (N, 3) not (3, N)
        if self.bvecs.shape[0] == 3 and self.bvecs.shape[1] != 3:
            self.bvecs = self.bvecs.T

        self.lambda_ = lambda_
        self.tau = tau

        # Auto-determine SH order from bmax
        if sh_order is None:
            bmax = np.max(bvals)
            # Rule: lmax ≈ (bmax / 1000) * 2, capped at 12
            sh_order = int(min(12, 2 * (bmax / 1000.0)))
            # Ensure even
            if sh_order % 2 != 0:
                sh_order -= 1
            logger.info(f"Auto-selected SH order {sh_order} based on bmax={bmax}")

        self.sh_order = sh_order
        self.sh = SphericalHarmonics(sh_order)

        # Convert gradients to spherical coordinates (use transposed self.bvecs)
        self.theta, self.phi = self._cart_to_sphere(self.bvecs)

        # Build spherical harmonic matrix
        self.B = self.sh.sh_matrix(self.theta, self.phi)

        logger.info(f"CSD model initialized: lmax={sh_order}, lambda={lambda_}, tau={tau}")

    def _cart_to_sphere(self, vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Cartesian to spherical coordinates"""
        x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]

        # Theta: elevation angle from z-axis [0, π]
        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.maximum(r, 1e-10)
        theta = np.arccos(np.clip(z / r, -1, 1))

        # Phi: azimuth angle in xy-plane [0, 2π]
        phi = np.arctan2(y, x)
        phi = np.where(phi < 0, phi + 2 * np.pi, phi)

        return theta, phi

    def fit(
        self,
        dwi_data: np.ndarray,
        response: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit CSD model and compute FOD

        Args:
            dwi_data: 4D diffusion data (x, y, z, volumes)
            response: Response function (n_volumes,)
            mask: Brain mask

        Returns:
            FOD coefficients (x, y, z, n_sh_coeffs)
        """
        logger.info("Computing FOD using CSD...")

        shape = dwi_data.shape[:3]
        if mask is None:
            mask = np.ones(shape, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.ndim > 3:
                mask = mask[..., 0]

        # Initialize FOD coefficients
        fod = np.zeros(shape + (self.sh.n_coeffs,), dtype=np.float32)

        # Build response convolution matrix
        R = self._build_response_matrix(response)

        # Compute pseudo-inverse with regularization
        A = self.B @ R  # Combined forward model
        # Clean NaN/Inf values
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        AtA = A.T @ A
        AtA_reg = AtA + self.lambda_ * np.eye(AtA.shape[0])

        try:
            from scipy.linalg import solve
            # Pre-compute (A^T A + λI)^-1 A^T for efficiency
            inv_term = solve(AtA_reg, A.T, assume_a='pos')
        except:
            inv_term = np.linalg.lstsq(AtA_reg, A.T, rcond=None)[0]

        # Fit voxel-by-voxel
        mask_indices = np.where(mask)
        n_voxels = len(mask_indices[0])

        logger.info(f"Fitting CSD for {n_voxels} voxels...")

        for i in range(n_voxels):
            x, y, z = int(mask_indices[0][i]), int(mask_indices[1][i]), int(mask_indices[2][i])

            signal = dwi_data[x, y, z]

            # Fit: fod_coeffs = (A^T A + λI)^-1 A^T signal
            coeffs = inv_term @ signal

            # Apply non-negativity constraint (simplified)
            # Full implementation would use iterative constraint optimization
            coeffs = np.maximum(coeffs, 0)

            fod[x, y, z] = coeffs

        logger.info("CSD fitting complete")
        return fod

    def _build_response_matrix(self, response: np.ndarray) -> np.ndarray:
        """
        Build response convolution matrix in SH space

        Args:
            response: Response function (n_volumes,)

        Returns:
            Convolution matrix (n_sh_coeffs, n_sh_coeffs)
        """
        # Simplified: diagonal matrix for now
        # Full implementation would compute proper SH convolution
        n_coeffs = self.sh.n_coeffs
        R = np.eye(n_coeffs, dtype=np.float64)

        # Scale by response magnitude
        scale = np.mean(response[self.bvals > 50])
        R *= scale

        return R


def auto_select_sh_order(bmax: float, snr: Optional[float] = None) -> int:
    """
    Automatically select appropriate SH order based on data quality

    Args:
        bmax: Maximum b-value
        snr: Signal-to-noise ratio (if known)

    Returns:
        Recommended SH order (even integer)
    """
    # Base selection on b-value
    if bmax < 1500:
        base_order = 6
    elif bmax < 2500:
        base_order = 8
    elif bmax < 4500:
        base_order = 10
    else:
        base_order = 12

    # Adjust for SNR if provided
    if snr is not None:
        if snr < 15:
            base_order = max(4, base_order - 2)
        elif snr < 20:
            base_order = max(6, base_order - 2)

    return base_order
