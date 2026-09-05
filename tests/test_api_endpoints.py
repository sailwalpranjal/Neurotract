"""
Integration tests for FastAPI endpoints
"""

import pytest
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
