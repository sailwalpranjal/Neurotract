"""
Parameter Sensitivity & Stability Analysis Engine for NeuroTract 2.0

Enables rigorous parameter-sensitivity exploration across tractography and connectome thresholds:
- Curvature angle constraints (max_angle)
- Anisotropy stopping thresholds (fa_threshold)
- Edge thresholding / pruning (streamline count cutoffs)
Computes edge stability, Jaccard overlap, metric divergence, and runtime impacts.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
import logging
from ..connectome.graph_metrics import ConnectomeMetrics

logger = logging.getLogger(__name__)


def compute_edge_stability(
    matrices: List[np.ndarray],
    thresholds: List[float],
    param_name: str
) -> Dict[str, Any]:
    """
    Compute edge stability across a spectrum of parameter variations.

    Parameters
    ----------
    matrices : List of 2D numpy arrays (adjacency matrices)
    thresholds : Parameter values corresponding to each matrix
    param_name : Name of the parameter varied (e.g. 'edge_threshold' or 'fa_threshold')

    Returns
    -------
    Dictionary with edge stability, Jaccard overlap, and metric divergence.
    """
    n_runs = len(matrices)
    if n_runs < 2:
        raise ValueError("At least 2 matrices required for sensitivity comparison")

    n_nodes = matrices[0].shape[0]
    calc = ConnectomeMetrics()

    # Binarized presence masks
    bin_masks = [m > 0 for m in matrices]

    # Consensus matrix: how many runs contain each edge
    consensus = np.zeros((n_nodes, n_nodes), dtype=int)
    for mask in bin_masks:
        consensus += mask.astype(int)

    # Upper triangle indices (undirected network)
    triu_idx = np.triu_indices(n_nodes, k=1)
    total_possible_edges = len(triu_idx[0])

    edge_presence_counts = consensus[triu_idx]

    # Stable edges: present in 100% of tested runs
    stable_edges_mask = edge_presence_counts == n_runs
    n_stable_edges = int(np.sum(stable_edges_mask))

    # Variable edges: present in at least one, but not all runs
    any_present_mask = edge_presence_counts > 0
    n_union_edges = int(np.sum(any_present_mask))
    n_variable_edges = n_union_edges - n_stable_edges

    # Overall Stability Ratio: stable edges / union edges
    stability_ratio = float(n_stable_edges / n_union_edges) if n_union_edges > 0 else 1.0

    # Pairwise Jaccard overlap matrix
    jaccard_matrix = np.ones((n_runs, n_runs), dtype=float)
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            m1 = bin_masks[i][triu_idx]
            m2 = bin_masks[j][triu_idx]
            intersection = np.sum(m1 & m2)
            union = np.sum(m1 | m2)
            jacc = float(intersection / union) if union > 0 else 1.0
            jaccard_matrix[i, j] = jacc
            jaccard_matrix[j, i] = jacc

    # Per-run metrics
    run_details = []
    for idx, (mat, val) in enumerate(zip(matrices, thresholds)):
        metrics = calc.compute_all(mat)
        g = metrics["global"]
        n_edges = int(np.sum(mat[triu_idx] > 0))

        # Difference against baseline (first run)
        if idx == 0:
            gained = 0
            lost = 0
        else:
            base_edges = bin_masks[0][triu_idx]
            curr_edges = bin_masks[idx][triu_idx]
            gained = int(np.sum(curr_edges & ~base_edges))
            lost = int(np.sum(base_edges & ~curr_edges))

        run_details.append({
            "run_index": idx,
            "parameter_value": val,
            "edge_count": n_edges,
            "density": round(float(g["density"]), 4),
            "global_efficiency": round(float(g["global_efficiency"]), 4),
            "clustering_coefficient": round(float(g["clustering_coefficient"]), 4),
            "characteristic_path_length": round(float(g["characteristic_path_length"]), 4),
            "modularity": round(float(metrics.get("communities", {}).get("louvain_modularity", 0)), 4),
            "gained_edges_vs_baseline": gained,
            "lost_edges_vs_baseline": lost,
        })

    return {
        "analysis_type": "Parameter Sensitivity & Edge Stability",
        "parameter_varied": param_name,
        "parameter_values": thresholds,
        "n_nodes": n_nodes,
        "total_possible_edges": total_possible_edges,
        "summary": {
            "stable_edge_count": n_stable_edges,
            "variable_edge_count": n_variable_edges,
            "union_edge_count": n_union_edges,
            "stability_ratio": round(stability_ratio, 4),
            "mean_pairwise_jaccard": round(float(np.mean(jaccard_matrix[np.triu_indices(n_runs, k=1)])), 4)
        },
        "pairwise_jaccard_matrix": jaccard_matrix.tolist(),
        "runs": run_details,
        "methodology": (
            "Stability ratio calculated as the number of edges consistently present across all parameter runs "
            "divided by the total union of observed edges. Pairwise similarity computed via Jaccard index: "
            "|E_i \cap E_j| / |E_i \cup E_j|."
        )
    }


def evaluate_connectome_threshold_sensitivity(
    base_matrix: np.ndarray,
    threshold_values: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Evaluate structural connectome sensitivity to edge weight thresholding / pruning.
    Standard thresholds: [0, 1, 2, 5, 10] streamlines.
    """
    if threshold_values is None:
        threshold_values = [0, 1, 2, 5, 10]

    matrices = []
    for th in threshold_values:
        m_thresh = np.copy(base_matrix)
        m_thresh[m_thresh < th] = 0
        matrices.append(m_thresh)

    return compute_edge_stability(matrices, threshold_values, "streamline_count_threshold")
