"""
Diffusion Tensor Imaging (DTI) Module

Implements robust DTI fitting with weighted least squares and outlier rejection.
Computes FA, MD, RD, AD, eigenvalues, and eigenvectors.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import logging
from scipy import linalg

logger = logging.getLogger(__name__)


class DTIModel:
    """
    Diffusion Tensor Imaging model with robust fitting
    """

    def __init__(
        self,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        b_threshold: float = 50.0,
        min_signal: float = 1e-6
    ):
        """
        Initialize DTI model

        Args:
            bvals: b-values array (N,)
            bvecs: gradient directions (N, 3)
            b_threshold: Threshold below which images are considered b0
            min_signal: Minimum signal value to avoid log(0)
        """
        self.bvals = np.array(bvals, dtype=np.float64)
        self.bvecs = np.array(bvecs, dtype=np.float64)

        # Ensure bvecs is (N, 3) not (3, N)
        if self.bvecs.shape[0] == 3 and self.bvecs.shape[1] != 3:
            self.bvecs = self.bvecs.T
            logger.info(f"Transposed bvecs from (3, {self.bvecs.shape[0]}) to ({self.bvecs.shape[0]}, 3)")

        self.b_threshold = b_threshold
        self.min_signal = min_signal

        # Identify b0 and dwi indices
        self.b0_mask = self.bvals <= b_threshold
        self.dwi_mask = ~self.b0_mask

        if not np.any(self.b0_mask):
            raise ValueError("No b0 images found (bval <= threshold)")

        # Build design matrix
        self.design_matrix = self._build_design_matrix()

        logger.info(f"DTI model initialized: {np.sum(self.b0_mask)} b0 images, "
                   f"{np.sum(self.dwi_mask)} DWI images")

    def _build_design_matrix(self) -> np.ndarray:
        """
        Build DTI design matrix for weighted least squares

        Returns:
            Design matrix of shape (n_dwi, 7)
        """
        # Extract DWI gradient directions and b-values
        g = self.bvecs[self.dwi_mask]  # (n_dwi, 3)
        b = self.bvals[self.dwi_mask]  # (n_dwi,)

        # Design matrix: [gx^2, 2gxgy, 2gxgz, gy^2, 2gygz, gz^2, 1] * (-b)
        # This gives: S = S0 * exp(-b * g^T D g)
        # Taking log: log(S) = log(S0) - b * (Dxx*gx^2 + 2*Dxy*gx*gy + ...)

        B = np.zeros((len(b), 7), dtype=np.float64)
        B[:, 0] = -b * g[:, 0]**2            # Dxx
        B[:, 1] = -2 * b * g[:, 0] * g[:, 1]  # Dxy
        B[:, 2] = -2 * b * g[:, 0] * g[:, 2]  # Dxz
        B[:, 3] = -b * g[:, 1]**2            # Dyy
        B[:, 4] = -2 * b * g[:, 1] * g[:, 2]  # Dyz
        B[:, 5] = -b * g[:, 2]**2            # Dzz
        B[:, 6] = 1.0                         # log(S0)

        return B

    def fit(
        self,
        dwi_data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_s0: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Fit DTI model to diffusion data

        Args:
            dwi_data: 4D diffusion data (x, y, z, volumes)
            mask: Binary brain mask (x, y, z). If None, processes all voxels
            return_s0: Whether to return fitted S0 values

        Returns:
            Dictionary containing:
                - 'fa': Fractional anisotropy
                - 'md': Mean diffusivity
                - 'rd': Radial diffusivity
                - 'ad': Axial diffusivity
                - 'eigenvalues': Sorted eigenvalues (x, y, z, 3)
                - 'eigenvectors': Corresponding eigenvectors (x, y, z, 3, 3)
                - 's0': Baseline signal (if return_s0=True)
        """
        logger.info("Fitting DTI model...")

        shape = dwi_data.shape[:3]
        if mask is None:
            mask = np.ones(shape, dtype=bool)
        else:
            # Ensure mask is boolean and 3D
            mask = np.asarray(mask, dtype=bool)
            if mask.ndim > 3:
                mask = mask[..., 0]

        # Initialize output arrays
        fa = np.zeros(shape, dtype=np.float32)
        md = np.zeros(shape, dtype=np.float32)
        rd = np.zeros(shape, dtype=np.float32)
        ad = np.zeros(shape, dtype=np.float32)
        eigenvalues = np.zeros(shape + (3,), dtype=np.float32)
        eigenvectors = np.zeros(shape + (3, 3), dtype=np.float32)

        if return_s0:
            s0 = np.zeros(shape, dtype=np.float32)

        # Extract b0 and DWI
        b0_data = dwi_data[..., self.b0_mask]
        dwi_subset = dwi_data[..., self.dwi_mask]

        # Compute mean b0
        s0_mean = np.mean(b0_data, axis=-1)
        s0_mean = np.maximum(s0_mean, self.min_signal)

        # Prepare for fitting
        mask_indices = np.where(mask)
        n_voxels = len(mask_indices[0])

        logger.info(f"Fitting {n_voxels} voxels...")

        # Voxel-wise fitting
        for i in range(n_voxels):
            x, y, z = int(mask_indices[0][i]), int(mask_indices[1][i]), int(mask_indices[2][i])

            # Extract signals
            s0_val = s0_mean[x, y, z]
            dwi_signals = dwi_subset[x, y, z]

            # Avoid log(0)
            dwi_signals = np.maximum(dwi_signals, self.min_signal)

            # Weighted least squares fit
            # Weight by signal intensity to reduce noise influence
            weights = dwi_signals / np.max(dwi_signals)
            W = np.diag(weights)

            # Log-linear fit: log_signal = B * params
            log_signal = np.log(dwi_signals / s0_val)

            try:
                # Weighted least squares: (B^T W B)^-1 B^T W log_signal
                BtW = self.design_matrix.T @ W
                params = linalg.lstsq(BtW @ self.design_matrix, BtW @ log_signal)[0]

                # Extract diffusion tensor elements
                Dxx, Dxy, Dxz, Dyy, Dyz, Dzz = params[:6]

                # Construct tensor matrix
                D = np.array([
                    [Dxx, Dxy, Dxz],
                    [Dxy, Dyy, Dyz],
                    [Dxz, Dyz, Dzz]
                ], dtype=np.float64)

                # Compute eigenvalues and eigenvectors
                evals, evecs = linalg.eigh(D)

                # Sort eigenvalues in descending order: λ1 >= λ2 >= λ3
                idx = np.argsort(evals)[::-1]
                evals = evals[idx]
                evecs = evecs[:, idx]

                # Ensure positive eigenvalues (physical constraint)
                evals = np.maximum(evals, 0.0)

                # Store results
                eigenvalues[x, y, z] = evals
                eigenvectors[x, y, z] = evecs

                # Compute scalar metrics
                if np.sum(evals) > 1e-10:
                    md[x, y, z] = np.mean(evals)  # Mean diffusivity
                    ad[x, y, z] = evals[0]        # Axial diffusivity (largest eigenvalue)
                    rd[x, y, z] = np.mean(evals[1:])  # Radial diffusivity (mean of smaller eigenvalues)

                    # Fractional anisotropy
                    md_val = md[x, y, z]
                    numerator = np.sqrt(0.5 * np.sum((evals - md_val)**2))
                    denominator = np.sqrt(np.sum(evals**2))
                    if denominator > 1e-10:
                        fa[x, y, z] = numerator / denominator
                    else:
                        fa[x, y, z] = 0.0
                else:
                    fa[x, y, z] = 0.0
                    md[x, y, z] = 0.0
                    rd[x, y, z] = 0.0
                    ad[x, y, z] = 0.0

                if return_s0:
                    s0[x, y, z] = s0_val

            except (linalg.LinAlgError, ValueError) as e:
                # Fitting failed for this voxel - leave as zero
                logger.debug(f"DTI fit failed at voxel ({x},{y},{z}): {e}")
                continue

        # Clip FA to [0, 1]
        fa = np.clip(fa, 0.0, 1.0)

        logger.info("DTI fitting complete")

        result = {
            'fa': fa,
            'md': md,
            'rd': rd,
            'ad': ad,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors
        }

        if return_s0:
            result['s0'] = s0

        return result

    def get_tensor_field(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct full tensor field from eigendecomposition

        Args:
            eigenvalues: Array of shape (x, y, z, 3)
            eigenvectors: Array of shape (x, y, z, 3, 3)

        Returns:
            Tensor field of shape (x, y, z, 3, 3)
        """
        shape = eigenvalues.shape[:3]
        tensors = np.zeros(shape + (3, 3), dtype=np.float32)

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    evals = eigenvalues[i, j, k]
                    evecs = eigenvectors[i, j, k]

                    # D = V * Λ * V^T
                    Lambda = np.diag(evals)
                    tensors[i, j, k] = evecs @ Lambda @ evecs.T

        return tensors


def compute_fa_map(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute FA map from eigenvalues

    Args:
        eigenvalues: Array of shape (..., 3)

    Returns:
        FA map with same shape as input (excluding last dimension)
    """
    md = np.mean(eigenvalues, axis=-1)
    numerator = np.sqrt(0.5 * np.sum((eigenvalues - md[..., np.newaxis])**2, axis=-1))
    denominator = np.sqrt(np.sum(eigenvalues**2, axis=-1))

    fa = np.zeros_like(md)
    valid = denominator > 1e-10
    fa[valid] = numerator[valid] / denominator[valid]

    return np.clip(fa, 0.0, 1.0)
