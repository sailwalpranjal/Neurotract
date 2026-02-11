"""
Unit tests for graph theory metrics
"""

import pytest
import numpy as np
from src.backend.connectome.graph_metrics import ConnectomeMetrics


class TestGraphMetrics:
    """Test graph theory metric computations"""

    @pytest.fixture
    def simple_network(self):
        """Create simple test network"""
        # 5-node network
        adj = np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 0, 1],
            [0, 0, 1, 1, 0]
        ], dtype=np.float64)
        return adj

    @pytest.fixture
    def complete_graph(self):
        """Create complete graph (fully connected)"""
        n = 5
        adj = np.ones((n, n)) - np.eye(n)
        return adj

    def test_metrics_initialization(self, simple_network):
        """Test metrics calculator initialization"""
        metrics = ConnectomeMetrics(simple_network)

        assert metrics.n_nodes == 5
        assert metrics.graph.number_of_nodes() == 5
        assert metrics.graph.number_of_edges() > 0

    def test_global_metrics(self, simple_network):
        """Test global network metrics"""
        metrics = ConnectomeMetrics(simple_network)
        global_metrics = metrics._compute_global_metrics()

        assert 'density' in global_metrics
        assert 'global_efficiency' in global_metrics
        assert 'characteristic_path_length' in global_metrics
        assert 'clustering_coefficient' in global_metrics

        # Check value ranges
        assert 0 <= global_metrics['density'] <= 1
        assert 0 <= global_metrics['global_efficiency'] <= 1
        assert global_metrics['characteristic_path_length'] > 0

    def test_degree_metrics(self, simple_network):
        """Test node degree computation"""
        metrics = ConnectomeMetrics(simple_network)
        degree = metrics._compute_degree()

        assert len(degree) == 5
        assert np.all(degree >= 0)

        # Check specific values for simple network
        # Node 2 (center) should have highest degree
        assert degree[2] == np.max(degree)

    def test_centrality_metrics(self, simple_network):
        """Test centrality measures"""
        metrics = ConnectomeMetrics(simple_network)

        betweenness = metrics._compute_betweenness()
        assert len(betweenness) == 5
        assert np.all(betweenness >= 0)

        closeness = metrics._compute_closeness()
        assert len(closeness) == 5
        assert np.all(closeness >= 0)

    def test_complete_graph_properties(self, complete_graph):
        """Test metrics on complete graph"""
        metrics = ConnectomeMetrics(complete_graph)
        global_metrics = metrics._compute_global_metrics()

        # Complete graph has density = 1
        assert global_metrics['density'] == 1.0

        # All nodes have same degree
        degree = metrics._compute_degree()
        assert np.all(degree == degree[0])

    def test_small_world_metrics(self, simple_network):
        """Test small-worldness computation"""
        metrics = ConnectomeMetrics(simple_network)
        global_metrics = metrics._compute_global_metrics()

        assert 'small_world_sigma' in global_metrics
        # Sigma should be positive
        assert global_metrics['small_world_sigma'] >= 0

    def test_all_metrics(self, simple_network):
        """Test comprehensive metrics computation"""
        metrics = ConnectomeMetrics(simple_network)
        all_metrics = metrics.compute_all_metrics()

        # Check all expected keys
        assert 'global' in all_metrics
        assert 'node_degree' in all_metrics
        assert 'node_strength' in all_metrics
        assert 'betweenness_centrality' in all_metrics
        assert 'communities' in all_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
