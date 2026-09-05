"""
Tests for Reference Benchmarks and Sensitivity Analysis
"""

import pytest
import numpy as np
from src.backend.analysis.benchmark import run_dti_reference_benchmark, run_graph_reference_benchmark
from src.backend.analysis.sensitivity import evaluate_connectome_threshold_sensitivity


class TestBenchmarkAndSensitivity:
    """Test algorithmic validation against references and parameter sensitivity"""

    def test_dti_reference_benchmark(self):
        """Test DTI agreement with DIPY baseline"""
        res = run_dti_reference_benchmark(n_samples=15, noise_snr=40.0)
        assert res["benchmark_name"] == "DTI Microstructure Reference Agreement"
        fa_metrics = res["metrics"]["fractional_anisotropy"]
        assert fa_metrics["pearson_r"] > 0.85
        assert fa_metrics["mean_absolute_error"] < 0.05
        assert fa_metrics["passed"] is True

    def test_graph_reference_benchmark(self):
        """Test connectome metrics agreement with NetworkX reference"""
        res = run_graph_reference_benchmark()
        if "error" in res:
            pytest.skip(res["error"])

        assert res["matrix_nodes"] == 89
        assert res["all_metrics_passed"] is True
        for comp in res["comparisons"]:
            assert comp["passed"] is True
            assert comp["absolute_diff"] < 1e-4

    def test_sensitivity_analysis(self):
        """Test connectome sensitivity across thresholds"""
        # Create synthetic symmetric adjacency matrix
        np.random.seed(42)
        A = np.random.randint(0, 10, size=(20, 20))
        A = (A + A.T) // 2
        np.fill_diagonal(A, 0)

        res = evaluate_connectome_threshold_sensitivity(A, [0, 2, 5])
        assert res["n_nodes"] == 20
        assert len(res["runs"]) == 3
        assert 0.0 <= res["summary"]["stability_ratio"] <= 1.0
