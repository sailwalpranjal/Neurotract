"""
Statistical Analysis Module
===========================

Classical statistical methods for group comparisons and effect size estimation.

Features:
- Permutation testing for group differences
- FDR correction (Benjamini-Hochberg)
- Effect sizes (Cohen's d, Glass's delta, Hedge's g)
- Bootstrap confidence intervals
- Network-based statistics (NBS)
- Mass univariate testing

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from pathlib import Path
from dataclasses import dataclass
import warnings

from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import networkx as nx
from sklearn.utils import resample


@dataclass
class PermutationTestResult:
    """Results from permutation test."""
    statistic: float
    pvalue: float
    null_distribution: np.ndarray
    n_permutations: int
    alternative: str


@dataclass
class EffectSizeResult:
    """Results from effect size calculation."""
    cohens_d: float
    hedges_g: float
    glass_delta: float
    ci_lower: float
    ci_upper: float
    ci_level: float


@dataclass
class NBSResult:
    """Results from Network-Based Statistic."""
    pvalue: float
    significant_components: List[np.ndarray]
    component_sizes: List[int]
    test_statistics: np.ndarray
    threshold: float
    n_permutations: int


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic_func: Callable = None,
    n_permutations: int = 10000,
    alternative: str = 'two-sided',
    random_state: Optional[int] = None
) -> PermutationTestResult:
    """
    Perform permutation test for group differences.

    Parameters
    ----------
    group1 : ndarray of shape (n_subjects1,) or (n_subjects1, n_features)
        First group data
    group2 : ndarray of shape (n_subjects2,) or (n_subjects2, n_features)
        Second group data
    statistic_func : callable, optional
        Function to compute test statistic. Default is mean difference.
    n_permutations : int, default=10000
        Number of permutations
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'greater', or 'less'
    random_state : int, optional
        Random seed

    Returns
    -------
    result : PermutationTestResult
        Permutation test results
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Default statistic: mean difference
    if statistic_func is None:
        def statistic_func(g1, g2):
            return np.mean(g1, axis=0) - np.mean(g2, axis=0)

    # Compute observed statistic
    observed_stat = statistic_func(group1, group2)

    # Combine groups
    combined = np.vstack([group1, group2])
    n1 = len(group1)
    n_total = len(combined)

    # Permutation test
    null_distribution = []
    for _ in range(n_permutations):
        # Shuffle labels
        permuted_indices = np.random.permutation(n_total)
        perm_group1 = combined[permuted_indices[:n1]]
        perm_group2 = combined[permuted_indices[n1:]]

        # Compute permuted statistic
        perm_stat = statistic_func(perm_group1, perm_group2)
        null_distribution.append(perm_stat)

    null_distribution = np.array(null_distribution)

    # Compute p-value
    if alternative == 'two-sided':
        pvalue = np.mean(np.abs(null_distribution) >= np.abs(observed_stat))
    elif alternative == 'greater':
        pvalue = np.mean(null_distribution >= observed_stat)
    elif alternative == 'less':
        pvalue = np.mean(null_distribution <= observed_stat)
    else:
        raise ValueError(f"Invalid alternative: {alternative}")

    # Handle scalar vs. array statistics
    if isinstance(pvalue, np.ndarray):
        pvalue = pvalue.item() if pvalue.size == 1 else pvalue

    return PermutationTestResult(
        statistic=observed_stat,
        pvalue=pvalue,
        null_distribution=null_distribution,
        n_permutations=n_permutations,
        alternative=alternative
    )


def fdr_correction(
    pvalues: np.ndarray,
    alpha: float = 0.05,
    method: str = 'bh'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform False Discovery Rate (FDR) correction.

    Parameters
    ----------
    pvalues : ndarray
        Array of p-values
    alpha : float, default=0.05
        FDR level
    method : str, default='bh'
        Method: 'bh' (Benjamini-Hochberg) or 'by' (Benjamini-Yekutieli)

    Returns
    -------
    reject : ndarray of bool
        Boolean array indicating which hypotheses are rejected
    pvalues_corrected : ndarray
        Adjusted p-values
    """
    pvalues = np.asarray(pvalues)
    shape = pvalues.shape
    pvalues_flat = pvalues.ravel()

    # Sort p-values
    sort_indices = np.argsort(pvalues_flat)
    pvalues_sorted = pvalues_flat[sort_indices]

    n = len(pvalues_flat)

    if method == 'bh':
        # Benjamini-Hochberg procedure
        critical_values = (np.arange(1, n + 1) / n) * alpha
        reject_sorted = pvalues_sorted <= critical_values

        # Find largest k where p[k] <= (k/n)*alpha
        if np.any(reject_sorted):
            max_k = np.where(reject_sorted)[0].max()
            reject_sorted[:(max_k + 1)] = True

        # Compute adjusted p-values
        pvalues_corrected_sorted = pvalues_sorted * n / np.arange(1, n + 1)

    elif method == 'by':
        # Benjamini-Yekutieli procedure (more conservative)
        c_n = np.sum(1.0 / np.arange(1, n + 1))
        critical_values = (np.arange(1, n + 1) / (n * c_n)) * alpha
        reject_sorted = pvalues_sorted <= critical_values

        if np.any(reject_sorted):
            max_k = np.where(reject_sorted)[0].max()
            reject_sorted[:(max_k + 1)] = True

        pvalues_corrected_sorted = pvalues_sorted * n * c_n / np.arange(1, n + 1)

    else:
        raise ValueError(f"Invalid method: {method}")

    # Ensure monotonicity (cumulative minimum from right to left)
    pvalues_corrected_sorted = np.minimum.accumulate(pvalues_corrected_sorted[::-1])[::-1]
    pvalues_corrected_sorted = np.minimum(pvalues_corrected_sorted, 1.0)

    # Unsort
    reject = np.empty(n, dtype=bool)
    pvalues_corrected = np.empty(n)
    reject[sort_indices] = reject_sorted
    pvalues_corrected[sort_indices] = pvalues_corrected_sorted

    return reject.reshape(shape), pvalues_corrected.reshape(shape)


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    pooled: bool = True
) -> float:
    """
    Compute Cohen's d effect size.

    Parameters
    ----------
    group1 : ndarray
        First group
    group2 : ndarray
        Second group
    pooled : bool, default=True
        Use pooled standard deviation

    Returns
    -------
    d : float
        Cohen's d
    """
    mean1 = np.mean(group1, axis=0)
    mean2 = np.mean(group2, axis=0)

    if pooled:
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        var1 = np.var(group1, axis=0, ddof=1)
        var2 = np.var(group2, axis=0, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    else:
        # Control group standard deviation
        pooled_std = np.std(group2, axis=0, ddof=1)

    return (mean1 - mean2) / pooled_std


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Hedge's g effect size (bias-corrected Cohen's d).

    Parameters
    ----------
    group1 : ndarray
        First group
    group2 : ndarray
        Second group

    Returns
    -------
    g : float
        Hedge's g
    """
    n1, n2 = len(group1), len(group2)
    d = cohens_d(group1, group2, pooled=True)

    # Correction factor
    df = n1 + n2 - 2
    correction = 1 - (3 / (4 * df - 1))

    return d * correction


def glass_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Glass's delta effect size.

    Uses control group standard deviation only.

    Parameters
    ----------
    group1 : ndarray
        Treatment group
    group2 : ndarray
        Control group

    Returns
    -------
    delta : float
        Glass's delta
    """
    mean1 = np.mean(group1, axis=0)
    mean2 = np.mean(group2, axis=0)
    std2 = np.std(group2, axis=0, ddof=1)

    return (mean1 - mean2) / std2


def compute_effect_size(
    group1: np.ndarray,
    group2: np.ndarray,
    ci_level: float = 0.95,
    n_bootstrap: int = 10000,
    random_state: Optional[int] = None
) -> EffectSizeResult:
    """
    Compute multiple effect size measures with confidence intervals.

    Parameters
    ----------
    group1 : ndarray
        First group
    group2 : ndarray
        Second group
    ci_level : float, default=0.95
        Confidence interval level
    n_bootstrap : int, default=10000
        Number of bootstrap samples for CI
    random_state : int, optional
        Random seed

    Returns
    -------
    result : EffectSizeResult
        Effect size results with confidence intervals
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Compute effect sizes
    d = cohens_d(group1, group2, pooled=True)
    g = hedges_g(group1, group2)
    delta = glass_delta(group1, group2)

    # Bootstrap confidence intervals for Cohen's d
    bootstrap_ds = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample1 = resample(group1, random_state=None)
        sample2 = resample(group2, random_state=None)

        # Compute Cohen's d
        bootstrap_d = cohens_d(sample1, sample2, pooled=True)
        bootstrap_ds.append(bootstrap_d)

    bootstrap_ds = np.array(bootstrap_ds)

    # Compute confidence intervals
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_ds, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_ds, 100 * (1 - alpha / 2))

    return EffectSizeResult(
        cohens_d=float(d) if np.isscalar(d) else d,
        hedges_g=float(g) if np.isscalar(g) else g,
        glass_delta=float(delta) if np.isscalar(delta) else delta,
        ci_lower=float(ci_lower) if np.isscalar(ci_lower) else ci_lower,
        ci_upper=float(ci_upper) if np.isscalar(ci_upper) else ci_upper,
        ci_level=ci_level
    )


def bootstrap_ci(
    data: np.ndarray,
    statistic_func: Callable,
    ci_level: float = 0.95,
    n_bootstrap: int = 10000,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals.

    Parameters
    ----------
    data : ndarray
        Data array
    statistic_func : callable
        Function to compute statistic
    ci_level : float, default=0.95
        Confidence level
    n_bootstrap : int, default=10000
        Number of bootstrap samples
    random_state : int, optional
        Random seed

    Returns
    -------
    estimate : float
        Point estimate
    ci_lower : float
        Lower confidence bound
    ci_upper : float
        Upper confidence bound
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Compute point estimate
    estimate = statistic_func(data)

    # Bootstrap
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = resample(data, random_state=None)
        bootstrap_stat = statistic_func(sample)
        bootstrap_stats.append(bootstrap_stat)

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute confidence intervals
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return estimate, ci_lower, ci_upper


def network_based_statistic(
    group1_matrices: np.ndarray,
    group2_matrices: np.ndarray,
    threshold: float = 3.0,
    n_permutations: int = 5000,
    test: str = 'ttest',
    random_state: Optional[int] = None
) -> NBSResult:
    """
    Perform Network-Based Statistic (NBS) for connectome comparison.

    Identifies connected components of edges showing significant group differences.

    Parameters
    ----------
    group1_matrices : ndarray of shape (n_subjects1, n_nodes, n_nodes)
        Connectivity matrices for group 1
    group2_matrices : ndarray of shape (n_subjects2, n_nodes, n_nodes)
        Connectivity matrices for group 2
    threshold : float, default=3.0
        Initial test statistic threshold (e.g., t-value)
    n_permutations : int, default=5000
        Number of permutations
    test : str, default='ttest'
        Statistical test: 'ttest' or 'mean_diff'
    random_state : int, optional
        Random seed

    Returns
    -------
    result : NBSResult
        NBS results
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_nodes = group1_matrices.shape[1]

    # Get upper triangle indices (exclude diagonal)
    triu_indices = np.triu_indices(n_nodes, k=1)

    # Extract edge values
    group1_edges = group1_matrices[:, triu_indices[0], triu_indices[1]]
    group2_edges = group2_matrices[:, triu_indices[0], triu_indices[1]]

    # Compute test statistics for each edge
    if test == 'ttest':
        test_stats, _ = stats.ttest_ind(group1_edges, group2_edges, axis=0)
    elif test == 'mean_diff':
        test_stats = np.mean(group1_edges, axis=0) - np.mean(group2_edges, axis=0)
    else:
        raise ValueError(f"Invalid test: {test}")

    # Apply threshold to create binary adjacency matrix
    supra_threshold = np.abs(test_stats) > threshold
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)
    adj_matrix[triu_indices[0], triu_indices[1]] = supra_threshold
    adj_matrix[triu_indices[1], triu_indices[0]] = supra_threshold

    # Find connected components
    n_components, labels = connected_components(
        csr_matrix(adj_matrix.astype(int)),
        directed=False
    )

    # Get component sizes
    component_sizes = []
    significant_components = []

    for comp_id in range(n_components):
        component_mask = (labels == comp_id)
        component_size = np.sum(component_mask)

        if component_size > 0:
            component_sizes.append(component_size)
            significant_components.append(np.where(component_mask)[0])

    # If no components found, return null result
    if not component_sizes:
        return NBSResult(
            pvalue=1.0,
            significant_components=[],
            component_sizes=[],
            test_statistics=test_stats,
            threshold=threshold,
            n_permutations=n_permutations
        )

    # Get maximum component size
    max_component_size = max(component_sizes)

    # Permutation testing
    combined_matrices = np.vstack([group1_matrices, group2_matrices])
    n1 = len(group1_matrices)
    n_total = len(combined_matrices)

    null_max_sizes = []

    for _ in range(n_permutations):
        # Shuffle group labels
        perm_indices = np.random.permutation(n_total)
        perm_group1 = combined_matrices[perm_indices[:n1]]
        perm_group2 = combined_matrices[perm_indices[n1:]]

        # Extract edges
        perm_edges1 = perm_group1[:, triu_indices[0], triu_indices[1]]
        perm_edges2 = perm_group2[:, triu_indices[0], triu_indices[1]]

        # Compute test statistics
        if test == 'ttest':
            perm_stats, _ = stats.ttest_ind(perm_edges1, perm_edges2, axis=0)
        else:
            perm_stats = np.mean(perm_edges1, axis=0) - np.mean(perm_edges2, axis=0)

        # Apply threshold
        perm_supra = np.abs(perm_stats) > threshold
        perm_adj = np.zeros((n_nodes, n_nodes), dtype=bool)
        perm_adj[triu_indices[0], triu_indices[1]] = perm_supra
        perm_adj[triu_indices[1], triu_indices[0]] = perm_supra

        # Find largest component
        n_comp, comp_labels = connected_components(
            csr_matrix(perm_adj.astype(int)),
            directed=False
        )

        if n_comp > 0:
            perm_max_size = max([np.sum(comp_labels == i) for i in range(n_comp)])
        else:
            perm_max_size = 0

        null_max_sizes.append(perm_max_size)

    null_max_sizes = np.array(null_max_sizes)

    # Compute p-value
    pvalue = np.mean(null_max_sizes >= max_component_size)

    return NBSResult(
        pvalue=pvalue,
        significant_components=significant_components,
        component_sizes=component_sizes,
        test_statistics=test_stats,
        threshold=threshold,
        n_permutations=n_permutations
    )


def mass_univariate_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    correction: str = 'fdr',
    n_permutations: int = 10000,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform mass univariate testing across features.

    Parameters
    ----------
    group1 : ndarray of shape (n_subjects1, n_features)
        First group features
    group2 : ndarray of shape (n_subjects2, n_features)
        Second group features
    alpha : float, default=0.05
        Significance level
    correction : str, default='fdr'
        Multiple comparison correction: 'fdr', 'bonferroni', or 'none'
    n_permutations : int, default=10000
        Number of permutations
    random_state : int, optional
        Random seed

    Returns
    -------
    results : dict
        Test results for each feature
    """
    n_features = group1.shape[1]

    # Compute statistics for each feature
    t_stats = np.zeros(n_features)
    p_values = np.zeros(n_features)
    effect_sizes = np.zeros(n_features)

    for i in range(n_features):
        # T-test
        t_stat, p_val = stats.ttest_ind(group1[:, i], group2[:, i])
        t_stats[i] = t_stat
        p_values[i] = p_val

        # Effect size
        effect_sizes[i] = cohens_d(group1[:, i], group2[:, i])

    # Multiple comparison correction
    if correction == 'fdr':
        reject, p_corrected = fdr_correction(p_values, alpha=alpha, method='bh')
    elif correction == 'bonferroni':
        p_corrected = np.minimum(p_values * n_features, 1.0)
        reject = p_corrected < alpha
    elif correction == 'none':
        p_corrected = p_values
        reject = p_values < alpha
    else:
        raise ValueError(f"Invalid correction method: {correction}")

    results = {
        't_statistics': t_stats,
        'p_values': p_values,
        'p_values_corrected': p_corrected,
        'reject': reject,
        'effect_sizes': effect_sizes,
        'n_significant': np.sum(reject),
        'correction_method': correction,
        'alpha': alpha
    }

    return results


def correlation_with_permutation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'pearson',
    n_permutations: int = 10000,
    random_state: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute correlation with permutation-based p-value.

    Parameters
    ----------
    x : ndarray
        First variable
    y : ndarray
        Second variable
    method : str, default='pearson'
        Correlation method: 'pearson' or 'spearman'
    n_permutations : int, default=10000
        Number of permutations
    random_state : int, optional
        Random seed

    Returns
    -------
    correlation : float
        Correlation coefficient
    pvalue : float
        Permutation-based p-value
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Compute observed correlation
    if method == 'pearson':
        obs_corr, _ = stats.pearsonr(x, y)
    elif method == 'spearman':
        obs_corr, _ = stats.spearmanr(x, y)
    else:
        raise ValueError(f"Invalid method: {method}")

    # Permutation test
    null_corrs = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)

        if method == 'pearson':
            perm_corr, _ = stats.pearsonr(x, y_perm)
        else:
            perm_corr, _ = stats.spearmanr(x, y_perm)

        null_corrs.append(perm_corr)

    null_corrs = np.array(null_corrs)

    # Compute two-tailed p-value
    pvalue = np.mean(np.abs(null_corrs) >= np.abs(obs_corr))

    return obs_corr, pvalue
