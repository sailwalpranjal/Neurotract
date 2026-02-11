"""
Graph Theory Metrics for Connectome Analysis

Implements comprehensive graph metrics including:
- Global: efficiency, path length, small-worldness, modularity
- Node-level: degree, centrality measures
- Community detection: Louvain, Leiden
"""

import numpy as np
import networkx as nx
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class ConnectomeMetrics:
    """Compute graph theory metrics on structural connectomes"""

    def __init__(self, adjacency_matrix: np.ndarray, node_labels: Optional[List[str]] = None):
        """
        Initialize connectome metrics calculator

        Args:
            adjacency_matrix: Weighted adjacency matrix (n_nodes, n_nodes)
            node_labels: Optional labels for nodes
        """
        self.adj = adjacency_matrix
        self.n_nodes = adjacency_matrix.shape[0]
        self.node_labels = node_labels or [f"node_{i}" for i in range(self.n_nodes)]

        # Create NetworkX graph
        self.graph = nx.from_numpy_array(self.adj)

        # Add node labels
        nx.set_node_attributes(
            self.graph,
            {i: label for i, label in enumerate(self.node_labels)},
            'label'
        )

        logger.info(f"Initialized connectome: {self.n_nodes} nodes, "
                   f"{self.graph.number_of_edges()} edges")

    def compute_all_metrics(self) -> Dict[str, any]:
        """
        Compute comprehensive set of graph metrics

        Returns:
            Dictionary containing all computed metrics
        """
        logger.info("Computing comprehensive graph metrics...")

        metrics = {}

        # Global metrics
        metrics['global'] = self._compute_global_metrics()

        # Node-level metrics
        metrics['node_degree'] = self._compute_degree()
        metrics['node_strength'] = self._compute_strength()
        metrics['betweenness_centrality'] = self._compute_betweenness()
        metrics['eigenvector_centrality'] = self._compute_eigenvector_centrality()
        metrics['closeness_centrality'] = self._compute_closeness()

        # Community detection
        metrics['communities'] = self._detect_communities()

        # Rich club
        metrics['rich_club'] = self._compute_rich_club()

        logger.info("All metrics computed")
        return metrics

    def _compute_global_metrics(self) -> Dict[str, float]:
        """Compute global network metrics"""
        logger.info("Computing global metrics...")

        metrics = {}

        # Density
        metrics['density'] = nx.density(self.graph)

        # Global efficiency
        metrics['global_efficiency'] = nx.global_efficiency(self.graph)

        # Characteristic path length (on connected component)
        if nx.is_connected(self.graph):
            metrics['characteristic_path_length'] = nx.average_shortest_path_length(
                self.graph, weight='weight'
            )
        else:
            # Compute on largest connected component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            metrics['characteristic_path_length'] = nx.average_shortest_path_length(
                subgraph, weight='weight'
            )
            logger.warning(f"Graph not connected, using largest component "
                         f"({len(largest_cc)}/{self.n_nodes} nodes)")

        # Clustering coefficient
        metrics['clustering_coefficient'] = nx.average_clustering(
            self.graph, weight='weight'
        )

        # Transitivity
        metrics['transitivity'] = nx.transitivity(self.graph)

        # Assortativity
        try:
            metrics['assortativity'] = nx.degree_assortativity_coefficient(self.graph)
        except:
            metrics['assortativity'] = np.nan

        # Small-worldness (sigma and omega)
        metrics.update(self._compute_small_worldness())

        return metrics

    def _compute_small_worldness(self) -> Dict[str, float]:
        """
        Compute small-world metrics (Watts-Strogatz)

        Returns:
            sigma and omega values
        """
        logger.info("Computing small-worldness...")

        # Compute network metrics
        C = nx.average_clustering(self.graph, weight='weight')

        if nx.is_connected(self.graph):
            L = nx.average_shortest_path_length(self.graph, weight='weight')
        else:
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            L = nx.average_shortest_path_length(subgraph, weight='weight')

        # Generate random graph for comparison
        n_edges = self.graph.number_of_edges()
        random_graph = nx.gnm_random_graph(self.n_nodes, n_edges)
        C_rand = nx.average_clustering(random_graph)

        if nx.is_connected(random_graph):
            L_rand = nx.average_shortest_path_length(random_graph)
        else:
            largest_cc = max(nx.connected_components(random_graph), key=len)
            subgraph = random_graph.subgraph(largest_cc)
            L_rand = nx.average_shortest_path_length(subgraph)

        # Compute small-world coefficient sigma
        # sigma = (C / C_rand) / (L / L_rand)
        # sigma > 1 indicates small-world properties
        gamma = C / C_rand if C_rand > 0 else 0
        lambda_ = L / L_rand if L_rand > 0 else 0
        sigma = gamma / lambda_ if lambda_ > 0 else 0

        return {
            'small_world_sigma': sigma,
            'clustering_ratio': gamma,
            'path_length_ratio': lambda_
        }

    def _compute_degree(self) -> np.ndarray:
        """Compute node degree"""
        degree = np.array([self.graph.degree(i) for i in range(self.n_nodes)])
        return degree

    def _compute_strength(self) -> np.ndarray:
        """Compute node strength (weighted degree)"""
        strength = np.array([
            sum(self.adj[i, :]) for i in range(self.n_nodes)
        ])
        return strength

    def _compute_betweenness(self) -> np.ndarray:
        """Compute betweenness centrality using Brandes' algorithm"""
        logger.info("Computing betweenness centrality...")

        betweenness_dict = nx.betweenness_centrality(
            self.graph, weight='weight', normalized=True
        )
        betweenness = np.array([betweenness_dict[i] for i in range(self.n_nodes)])

        return betweenness

    def _compute_eigenvector_centrality(self) -> np.ndarray:
        """Compute eigenvector centrality with power iteration"""
        logger.info("Computing eigenvector centrality...")

        try:
            centrality_dict = nx.eigenvector_centrality(
                self.graph, weight='weight', max_iter=1000, tol=1e-6
            )
            centrality = np.array([centrality_dict[i] for i in range(self.n_nodes)])
        except nx.PowerIterationFailedConvergence:
            logger.warning("Eigenvector centrality did not converge, using zeros")
            centrality = np.zeros(self.n_nodes)

        return centrality

    def _compute_closeness(self) -> np.ndarray:
        """Compute closeness centrality"""
        logger.info("Computing closeness centrality...")

        closeness_dict = nx.closeness_centrality(self.graph, distance='weight')
        closeness = np.array([closeness_dict[i] for i in range(self.n_nodes)])

        return closeness

    def _detect_communities(self) -> Dict[str, any]:
        """
        Detect communities using Louvain and Leiden algorithms

        Returns:
            Dictionary with community assignments and modularity
        """
        logger.info("Detecting communities...")

        communities = {}

        # Louvain algorithm
        try:
            from community import community_louvain
            partition_louvain = community_louvain.best_partition(self.graph, weight='weight')
            communities['louvain_partition'] = np.array([
                partition_louvain[i] for i in range(self.n_nodes)
            ])
            communities['louvain_modularity'] = community_louvain.modularity(
                partition_louvain, self.graph, weight='weight'
            )
            logger.info(f"Louvain: {len(set(partition_louvain.values()))} communities, "
                       f"Q={communities['louvain_modularity']:.3f}")
        except ImportError:
            logger.warning("python-louvain not available, skipping Louvain algorithm")

        # Leiden algorithm (if available)
        try:
            import igraph as ig
            import leidenalg

            # Convert to igraph
            edges = [(u, v, self.adj[u, v]) for u, v in self.graph.edges()]
            g = ig.Graph(n=self.n_nodes, edges=[(u, v) for u, v, _ in edges])
            g.es['weight'] = [w for _, _, w in edges]

            # Run Leiden
            partition_leiden = leidenalg.find_partition(
                g, leidenalg.ModularityVertexPartition, weights='weight'
            )

            communities['leiden_partition'] = np.array(partition_leiden.membership)
            communities['leiden_modularity'] = partition_leiden.modularity

            logger.info(f"Leiden: {len(set(partition_leiden.membership))} communities, "
                       f"Q={partition_leiden.modularity:.3f}")

        except (ImportError, Exception) as e:
            logger.warning(f"Leiden algorithm not available: {e}")

        return communities

    def _compute_rich_club(self, k_max: Optional[int] = None) -> Dict[int, float]:
        """
        Compute rich-club coefficient

        Args:
            k_max: Maximum degree to compute (auto if None)

        Returns:
            Dictionary mapping degree to rich-club coefficient
        """
        logger.info("Computing rich-club coefficient...")

        if k_max is None:
            degrees = [self.graph.degree(i) for i in range(self.n_nodes)]
            k_max = int(np.max(degrees))

        rich_club = {}
        for k in range(1, min(k_max, self.n_nodes)):
            try:
                phi_k = nx.rich_club_coefficient(self.graph, k, normalized=False)
                if k in phi_k:
                    rich_club[k] = phi_k[k]
            except:
                break

        return rich_club

    def get_shortest_paths(
        self,
        source: int,
        target: Optional[int] = None
    ) -> Dict:
        """
        Compute shortest paths using Dijkstra's algorithm

        Args:
            source: Source node index
            target: Target node index (all targets if None)

        Returns:
            Dictionary of shortest paths
        """
        if target is None:
            # All pairs from source
            paths = nx.single_source_dijkstra_path(
                self.graph, source, weight='weight'
            )
            lengths = nx.single_source_dijkstra_path_length(
                self.graph, source, weight='weight'
            )
        else:
            # Single path
            path = nx.dijkstra_path(self.graph, source, target, weight='weight')
            length = nx.dijkstra_path_length(self.graph, source, target, weight='weight')
            paths = {target: path}
            lengths = {target: length}

        return {'paths': paths, 'lengths': lengths}


def compute_network_resilience(
    adjacency: np.ndarray,
    attack_type: str = 'random',
    n_steps: int = 10
) -> Dict[str, np.ndarray]:
    """
    Simulate network resilience to node attacks

    Args:
        adjacency: Adjacency matrix
        attack_type: 'random' or 'targeted' (by degree)
        n_steps: Number of attack steps

    Returns:
        Dictionary with efficiency at each step
    """
    logger.info(f"Computing network resilience ({attack_type} attack)...")

    n_nodes = adjacency.shape[0]
    nodes_per_step = max(1, n_nodes // n_steps)

    efficiency = []
    size = []

    # Original network
    G = nx.from_numpy_array(adjacency)
    efficiency.append(nx.global_efficiency(G))
    size.append(G.number_of_nodes())

    # Determine attack order
    if attack_type == 'targeted':
        # Target high-degree nodes
        degrees = dict(G.degree())
        attack_order = sorted(degrees, key=degrees.get, reverse=True)
    else:
        # Random order
        attack_order = np.random.permutation(n_nodes).tolist()

    # Simulate attacks
    for step in range(1, n_steps + 1):
        # Remove nodes
        nodes_to_remove = attack_order[:step * nodes_per_step]
        G_attacked = G.copy()
        G_attacked.remove_nodes_from(nodes_to_remove)

        # Compute metrics
        if G_attacked.number_of_nodes() > 0:
            efficiency.append(nx.global_efficiency(G_attacked))
            size.append(G_attacked.number_of_nodes())
        else:
            efficiency.append(0)
            size.append(0)

    return {
        'efficiency': np.array(efficiency),
        'size': np.array(size),
        'attack_order': attack_order
    }
