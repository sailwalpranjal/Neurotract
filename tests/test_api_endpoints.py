"""
Integration tests for FastAPI endpoints
"""

import pytest
import io
import shutil
import uuid
import numpy as np
import nibabel as nib
from fastapi.testclient import TestClient
from src.backend.api.server import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test REST API and real-time endpoints"""

    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "neurotract" in data["software_versions"]

    def test_version(self):
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert data["neurotract_version"] == "2.0.0"

    def test_results_available(self):
        response = client.get("/results/available")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "subject_id" in data[0]

    def test_metric_definitions(self):
        response = client.get("/api/provenance/metric-definitions")
        assert response.status_code == 200
        data = response.json()
        assert "global_efficiency" in data
        assert "formula" in data["global_efficiency"]

    def test_orthogonal_slices(self):
        response = client.get("/results/SUB1/slices?axial=0.5&coronal=0.5&sagittal=0.5")
        if response.status_code == 404:
            pytest.skip("SUB1 volume not available for test")
        assert response.status_code == 200
        data = response.json()
        assert "slices" in data
        assert "axial" in data["slices"]
        assert data["slices"]["axial"]["image"].startswith("data:image/png;base64,")

    def test_validation_report_endpoint(self):
        response = client.get("/api/datasets/report/SUB1")
        if response.status_code == 404:
            pytest.skip("SUB1 not available for test")
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert data["num_volumes"] == 160

    def test_uploaded_dataset_is_validated_before_submission(self, tmp_path):
        """A browser upload session keeps its DWI and gradient files together."""
        dwi_path = tmp_path / "dwi.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 2)), np.eye(4)), dwi_path)
        session_id = str(uuid.uuid4())
        try:
            with open(dwi_path, "rb") as dwi:
                response = client.post("/upload", data={"upload_id": session_id}, files={"file": ("dwi.nii.gz", dwi, "application/gzip")})
            assert response.status_code == 200
            response = client.post("/upload", data={"upload_id": session_id}, files={"file": ("dwi.bval", io.BytesIO(b"0 1000\n"), "text/plain")})
            assert response.status_code == 200
            response = client.post("/upload", data={"upload_id": session_id}, files={"file": ("dwi.bvec", io.BytesIO(b"0 1\n0 0\n0 0\n"), "text/plain")})
            assert response.status_code == 200

            report = client.post("/api/uploads/validate", json={"upload_id": session_id})
            assert report.status_code == 200
            assert report.json()["ready_for_pipeline"] is True
            discovery = client.post("/api/uploads/discover", json={"upload_id": session_id})
            assert discovery.status_code == 200
            assert discovery.json()["summary"]["compatible"] == 1
            assert discovery.json()["datasets"][0]["id"] == "dwi.nii.gz"
        finally:
            shutil.rmtree("uploads/" + session_id, ignore_errors=True)

    def test_provenance_metric_endpoint(self):
        response = client.get("/api/provenance/global_efficiency")
        assert response.status_code == 200
        data = response.json()
        assert data["metric_id"] == "global_efficiency"
        assert "E_glob" in data["formula"]
        assert "Latora" in data["reference_citation"]

    def test_sensitivity_run_endpoint(self):
        response = client.post("/api/sensitivity/run", json={"subject_id": "SUB1", "thresholds": [0, 2, 5]})
        if response.status_code == 404:
            pytest.skip("SUB1 connectome not found")
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "stability_ratio" in data["summary"]

    def test_report_export_endpoint(self):
        response = client.get("/api/report/SUB1/export")
        if response.status_code == 404:
            pytest.skip("SUB1 not available for report export")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "NeuroTract 2.0" in response.text
