"""
Unit tests for Scientific Provenance Tracking
"""

import pytest
from src.backend.provenance.tracker import (
    ProvenanceTracker,
    get_software_versions,
    METRIC_REGISTRY
)


class TestProvenanceTracker:
    """Test scientific provenance recording and metric inspections"""

    def test_software_versions_collected(self):
        versions = get_software_versions()
        assert "neurotract" in versions
        assert "dipy" in versions
        assert "networkx" in versions
        assert "nibabel" in versions
        assert "scipy" in versions

    def test_metric_registry_definitions(self):
        assert "global_efficiency" in METRIC_REGISTRY
        assert "clustering_coefficient" in METRIC_REGISTRY
        assert "fa" in METRIC_REGISTRY
        assert "formula" in METRIC_REGISTRY["global_efficiency"]
        assert "units" in METRIC_REGISTRY["global_efficiency"]

    def test_create_and_query_execution_provenance(self, tmp_path):
        tracker = ProvenanceTracker(storage_dir=tmp_path)
        exec_id = "test-exec-123"

        rec = tracker.create_execution(
            execution_id=exec_id,
            dataset_name="TestDataset",
            dataset_checksums={"dwi": "abc123sha"},
            rng_seed=42,
            tractography_params={"step_size": 0.5}
        )

        assert rec.execution_id == exec_id
        assert rec.rng_seed == 42
        assert rec.status == "running"

        # Create metric provenance
        prov = tracker.create_metric_provenance(
            execution_id=exec_id,
            metric_key="global_efficiency",
            value=0.489,
            input_properties={"node_count": 89, "edge_count": 788}
        )
        assert prov.value == 0.489
        assert prov.units == "dimensionless [0, 1]"

        # Complete execution
        tracker.complete_execution(exec_id, status="completed")

        # Query back
        queried = tracker.get_record(exec_id)
        assert queried is not None
        assert queried["status"] == "completed"
        assert "global_efficiency" in queried["metric_provenance_records"]
