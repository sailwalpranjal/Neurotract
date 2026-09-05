"""
Unit tests for DatasetValidator boundary checks
"""

import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
from src.backend.data.validator import DatasetValidator, validate_dataset


class TestDatasetValidator:
    """Test boundary validation and gradient verification"""

    def test_valid_stanford_dataset(self):
        """Test validation on real Stanford HARDI dataset"""
        dwi_path = Path("datasets/Stanford dataset/SUB1_b1000_1.nii.gz")
        if not dwi_path.exists():
            pytest.skip("Stanford dataset not present locally")

        validator = DatasetValidator()
        report = validator.validate_dwi_dataset(dwi_path)

        assert report.is_valid is True
        assert report.dimensions == [81, 106, 76, 160]
        assert report.num_volumes == 160
        assert report.coordinate_system.startswith("RAS")
        assert report.gradient_summary is not None
        assert report.gradient_summary.n_b0 == 10
        assert report.gradient_summary.n_dwi == 150
        assert report.gradient_summary.is_unit_normalized is True
        assert len(report.errors) == 0

    def test_missing_file_rejected(self, tmp_path):
        """Test non-existent file rejection"""
        non_existent = tmp_path / "does_not_exist.nii.gz"
        validator = DatasetValidator()
        report = validator.validate_dwi_dataset(non_existent)

        assert report.is_valid is False
        assert len(report.errors) > 0
        assert "not found" in report.errors[0]

    def test_empty_zero_byte_file_rejected(self, tmp_path):
        """Test zero-byte file rejection"""
        empty_file = tmp_path / "empty_scan.nii.gz"
        empty_file.touch()

        validator = DatasetValidator()
        report = validator.validate_dwi_dataset(empty_file)

        assert report.is_valid is False
        assert any("empty" in e.lower() for e in report.errors)

    def test_dimension_mismatch_rejected(self, tmp_path):
        """Test mismatch between 4D DWI volume count and bval entries"""
        # Create synthetic 4D NIfTI with 10 volumes
        data = np.zeros((10, 10, 10, 10), dtype=np.float32)
        affine = np.eye(4)
        nii = nib.Nifti1Image(data, affine)
        nii_path = tmp_path / "test_dwi.nii.gz"
        nib.save(nii, str(nii_path))

        # Create bval with 5 entries (mismatch: 10 vs 5)
        bval_path = tmp_path / "test_dwi.bval"
        np.savetxt(str(bval_path), np.array([0, 1000, 1000, 1000, 1000]))

        # Create bvec with 5 entries
        bvec_path = tmp_path / "test_dwi.bvec"
        np.savetxt(str(bvec_path), np.zeros((5, 3)))

        validator = DatasetValidator()
        report = validator.validate_dwi_dataset(nii_path, bval_path, bvec_path)

        assert report.is_valid is False
        assert any("mismatch" in e.lower() for e in report.errors)
