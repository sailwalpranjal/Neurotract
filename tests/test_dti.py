"""
Unit tests for DTI model
"""

import pytest
import numpy as np
from src.backend.microstructure.dti import DTIModel, compute_fa_map


class TestDTIModel:
    """Test DTI model fitting"""

    @pytest.fixture
    def simple_gradients(self):
        """Create simple gradient scheme"""
        # 1 b0 + 6 directions
        bvals = np.array([0, 1000, 1000, 1000, 1000, 1000, 1000])
        bvecs = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1]
        ], dtype=np.float64)
        # Normalize
        bvecs[1:] = bvecs[1:] / np.linalg.norm(bvecs[1:], axis=1, keepdims=True)
        return bvals, bvecs

    @pytest.fixture
    def synthetic_dwi(self, simple_gradients):
        """Create synthetic DWI data"""
        bvals, bvecs = simple_gradients

        # Create 3x3x3 volume
        dwi = np.zeros((3, 3, 3, len(bvals)))

        # Generate synthetic signals with known tensor
        D = np.array([[1.5e-3, 0, 0],
                     [0, 0.4e-3, 0],
                     [0, 0, 0.4e-3]])  # Simple cylindrical tensor

        for vol_idx in range(len(bvals)):
            if bvals[vol_idx] == 0:
                dwi[..., vol_idx] = 1000  # b0 signal
            else:
                g = bvecs[vol_idx]
                b = bvals[vol_idx]
                # Signal: S = S0 * exp(-b * g^T D g)
                signal = 1000 * np.exp(-b * g @ D @ g)
                dwi[..., vol_idx] = signal

        return dwi

    def test_dti_initialization(self, simple_gradients):
        """Test DTI model initialization"""
        bvals, bvecs = simple_gradients
        model = DTIModel(bvals, bvecs)

        assert model.bvals.shape == (7,)
        assert model.bvecs.shape == (7, 3)
        assert np.sum(model.b0_mask) == 1
        assert np.sum(model.dwi_mask) == 6

    def test_dti_fitting(self, simple_gradients, synthetic_dwi):
        """Test DTI fitting on synthetic data"""
        bvals, bvecs = simple_gradients
        model = DTIModel(bvals, bvecs)

        # Fit center voxel
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = True

        results = model.fit(synthetic_dwi, mask=mask)

        # Check outputs
        assert 'fa' in results
        assert 'md' in results
        assert 'rd' in results
        assert 'ad' in results
        assert 'eigenvalues' in results
        assert 'eigenvectors' in results

        # FA should be high for cylindrical tensor
        fa_value = results['fa'][1, 1, 1]
        assert 0.5 < fa_value < 1.0, f"FA={fa_value} out of expected range"

        # Check eigenvalue ordering
        evals = results['eigenvalues'][1, 1, 1]
        assert evals[0] >= evals[1] >= evals[2], "Eigenvalues not sorted"

    def test_fa_computation(self):
        """Test FA map computation"""
        # Create eigenvalues
        eigenvalues = np.array([
            [[1.5e-3, 0.4e-3, 0.4e-3]],  # High FA
            [[0.8e-3, 0.7e-3, 0.7e-3]],  # Low FA
            [[0.6e-3, 0.6e-3, 0.6e-3]]   # Isotropic (FA=0)
        ])

        fa = compute_fa_map(eigenvalues)

        assert fa.shape == (3, 1)
        assert 0.6 < fa[0, 0] < 1.0, "High FA case failed"
        assert 0.0 < fa[1, 0] < 0.3, "Low FA case failed"
        assert fa[2, 0] < 0.1, "Isotropic case failed"

    def test_dti_robustness(self, simple_gradients):
        """Test DTI robustness to noisy data"""
        bvals, bvecs = simple_gradients
        model = DTIModel(bvals, bvecs)

        # Create noisy data
        dwi_noisy = np.random.rand(3, 3, 3, len(bvals)) * 100 + 500

        # Should not crash
        results = model.fit(dwi_noisy)

        assert results['fa'].shape == (3, 3, 3)
        assert np.all(results['fa'] >= 0)
        assert np.all(results['fa'] <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
