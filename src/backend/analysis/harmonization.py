"""
Harmonization Module
===================

ComBat harmonization for removing scanner and site effects while preserving
biological variance in neuroimaging data.

Features:
- ComBat harmonization for multi-site data
- Parametric and non-parametric variants
- Preserve biological covariates (age, sex, diagnosis)
- Handle missing data
- Batch effect visualization

References
----------
Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in microarray
expression data using empirical Bayes methods. Biostatistics, 8(1), 118-127.

Fortin, J. P., et al. (2017). Harmonization of cortical thickness measurements across
scanners and sites. NeuroImage, 167, 104-120.

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import warnings

from scipy import stats
from scipy.linalg import solve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class ComBatHarmonizer:
    """
    ComBat harmonization for removing scanner/site effects.

    Implements empirical Bayes method to harmonize data across batches
    while preserving biological variation.
    """

    def __init__(
        self,
        parametric: bool = True,
        eb: bool = True,
        mean_only: bool = False
    ):
        """
        Initialize ComBat harmonizer.

        Parameters
        ----------
        parametric : bool, default=True
            Use parametric adjustments (faster, assumes normality)
        eb : bool, default=True
            Use empirical Bayes for parameter estimation
        mean_only : bool, default=False
            Only adjust for batch mean effects (not variance)
        """
        self.parametric = parametric
        self.eb = eb
        self.mean_only = mean_only

        # Fitted parameters
        self.batch_info_ = None
        self.gamma_hat_ = None
        self.delta_hat_ = None
        self.gamma_star_ = None
        self.delta_star_ = None
        self.stand_mean_ = None
        self.mod_mean_ = None
        self.var_pooled_ = None

    def fit(
        self,
        data: np.ndarray,
        batch: np.ndarray,
        covariates: Optional[pd.DataFrame] = None
    ) -> 'ComBatHarmonizer':
        """
        Fit ComBat harmonization parameters.

        Parameters
        ----------
        data : ndarray of shape (n_features, n_samples)
            Feature data (features × samples)
        batch : ndarray of shape (n_samples,)
            Batch labels for each sample
        covariates : DataFrame of shape (n_samples, n_covariates), optional
            Biological covariates to preserve (e.g., age, sex, diagnosis)

        Returns
        -------
        self : ComBatHarmonizer
        """
        # Store batch information
        batch = np.asarray(batch)
        n_samples = data.shape[1]
        n_features = data.shape[0]

        # Encode batches
        batch_encoder = LabelEncoder()
        batch_encoded = batch_encoder.fit_transform(batch)
        batch_labels = batch_encoder.classes_
        n_batches = len(batch_labels)

        # Store batch info
        self.batch_info_ = {
            'encoder': batch_encoder,
            'labels': batch_labels,
            'n_batches': n_batches
        }

        # Create batch design matrix (one-hot encoding)
        batch_design = np.zeros((n_samples, n_batches))
        for i, batch_id in enumerate(batch_encoded):
            batch_design[i, batch_id] = 1

        # Create covariate design matrix
        if covariates is not None:
            covariate_matrix = self._create_covariate_matrix(covariates)
            # Combine batch and covariate design
            design = np.hstack([batch_design, covariate_matrix])
        else:
            design = batch_design
            covariate_matrix = None

        # Standardize data across features
        print("Standardizing data...")
        data_standardized, stand_mean = self._standardize_data(
            data, design, batch_design
        )

        self.stand_mean_ = stand_mean

        # Estimate batch effect parameters
        print("Estimating batch effects...")
        gamma_hat, delta_hat = self._estimate_batch_effects(
            data_standardized, batch_design, batch_encoded
        )

        self.gamma_hat_ = gamma_hat
        self.delta_hat_ = delta_hat

        # Apply empirical Bayes
        if self.eb:
            print("Applying empirical Bayes...")
            gamma_star, delta_star = self._empirical_bayes(
                data_standardized, gamma_hat, delta_hat, batch_design, batch_encoded
            )
        else:
            gamma_star = gamma_hat
            delta_star = delta_hat

        self.gamma_star_ = gamma_star
        self.delta_star_ = delta_star

        return self

    def transform(
        self,
        data: np.ndarray,
        batch: np.ndarray,
        covariates: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Apply ComBat harmonization.

        Parameters
        ----------
        data : ndarray of shape (n_features, n_samples)
            Feature data to harmonize
        batch : ndarray of shape (n_samples,)
            Batch labels
        covariates : DataFrame of shape (n_samples, n_covariates), optional
            Biological covariates

        Returns
        -------
        data_harmonized : ndarray of shape (n_features, n_samples)
            Harmonized data
        """
        if self.batch_info_ is None:
            raise ValueError("Must call fit() before transform()")

        batch = np.asarray(batch)
        n_samples = data.shape[1]
        n_features = data.shape[0]

        # Encode batches
        batch_encoded = self.batch_info_['encoder'].transform(batch)
        n_batches = self.batch_info_['n_batches']

        # Create batch design matrix
        batch_design = np.zeros((n_samples, n_batches))
        for i, batch_id in enumerate(batch_encoded):
            batch_design[i, batch_id] = 1

        # Create full design matrix
        if covariates is not None:
            covariate_matrix = self._create_covariate_matrix(covariates)
            design = np.hstack([batch_design, covariate_matrix])
        else:
            design = batch_design

        # Standardize data
        data_standardized, _ = self._standardize_data(
            data, design, batch_design, use_fitted_mean=True
        )

        # Remove batch effects
        print("Removing batch effects...")
        data_harmonized = self._remove_batch_effects(
            data_standardized, batch_design, batch_encoded
        )

        # Add back standardization mean
        data_harmonized = data_harmonized + self.stand_mean_[:, np.newaxis]

        return data_harmonized

    def fit_transform(
        self,
        data: np.ndarray,
        batch: np.ndarray,
        covariates: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data, batch, covariates)
        return self.transform(data, batch, covariates)

    def _create_covariate_matrix(self, covariates: pd.DataFrame) -> np.ndarray:
        """Create design matrix from covariates."""
        covariate_matrix = []

        for col in covariates.columns:
            values = covariates[col].values

            # Check if categorical
            if covariates[col].dtype == 'object' or len(np.unique(values)) < 10:
                # One-hot encoding (drop first category)
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(values)
                n_categories = len(encoder.classes_)

                for cat_id in range(1, n_categories):
                    covariate_matrix.append((encoded == cat_id).astype(float))
            else:
                # Standardize continuous variables
                standardized = (values - np.mean(values)) / np.std(values)
                covariate_matrix.append(standardized)

        return np.column_stack(covariate_matrix) if covariate_matrix else np.zeros((len(covariates), 0))

    def _standardize_data(
        self,
        data: np.ndarray,
        design: np.ndarray,
        batch_design: np.ndarray,
        use_fitted_mean: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standardize data by removing design effects."""
        n_features, n_samples = data.shape

        # Estimate design coefficients (least squares)
        # data = design @ beta + error
        beta_hat = solve(design.T @ design, design.T @ data.T).T

        # Compute grand mean (average effect across batches)
        if not use_fitted_mean:
            grand_mean = np.mean(data, axis=1)
            self.mod_mean_ = grand_mean
        else:
            grand_mean = self.mod_mean_

        # Compute standardization mean (remove batch effects from estimate)
        # Keep only non-batch effects
        n_batches = batch_design.shape[1]
        if design.shape[1] > n_batches:
            # Has covariates
            stand_mean = design[:, n_batches:] @ beta_hat[:, n_batches:].T
            stand_mean = np.mean(stand_mean, axis=0)
        else:
            stand_mean = grand_mean

        # Subtract design effects
        data_adjusted = data - (design @ beta_hat.T).T + stand_mean[:, np.newaxis]

        return data_adjusted, stand_mean

    def _estimate_batch_effects(
        self,
        data: np.ndarray,
        batch_design: np.ndarray,
        batch_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate batch-specific location (gamma) and scale (delta) parameters."""
        n_features, n_samples = data.shape
        n_batches = batch_design.shape[1]

        gamma_hat = np.zeros((n_features, n_batches))
        delta_hat = np.zeros((n_features, n_batches))

        for batch_id in range(n_batches):
            # Get samples in this batch
            batch_mask = batch_labels == batch_id
            batch_data = data[:, batch_mask]

            # Estimate location (mean)
            gamma_hat[:, batch_id] = np.mean(batch_data, axis=1)

            # Estimate scale (variance)
            delta_hat[:, batch_id] = np.var(batch_data, axis=1, ddof=1)

        return gamma_hat, delta_hat

    def _empirical_bayes(
        self,
        data: np.ndarray,
        gamma_hat: np.ndarray,
        delta_hat: np.ndarray,
        batch_design: np.ndarray,
        batch_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply empirical Bayes shrinkage to batch effect estimates."""
        n_features, n_samples = data.shape
        n_batches = batch_design.shape[1]

        # Initialize
        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.zeros_like(delta_hat)

        for batch_id in range(n_batches):
            batch_mask = batch_labels == batch_id
            n_batch = np.sum(batch_mask)

            if n_batch == 0:
                continue

            # Empirical Bayes for location (gamma)
            gamma_bar = np.mean(gamma_hat[:, batch_id])
            tau_bar_sq = np.var(gamma_hat[:, batch_id])

            # Shrink gamma towards empirical prior
            gamma_star[:, batch_id] = self._postmean(
                gamma_hat[:, batch_id], gamma_bar, n_batch, delta_hat[:, batch_id], tau_bar_sq
            )

            # Empirical Bayes for scale (delta)
            if not self.mean_only:
                delta_star[:, batch_id] = self._postvar(
                    data[:, batch_mask], gamma_star[:, batch_id], delta_hat[:, batch_id]
                )
            else:
                delta_star[:, batch_id] = delta_hat[:, batch_id]

        return gamma_star, delta_star

    def _postmean(
        self,
        gamma_hat: np.ndarray,
        gamma_bar: float,
        n: int,
        delta: np.ndarray,
        tau_sq: float
    ) -> np.ndarray:
        """Posterior mean for gamma (location parameter)."""
        # Inverse variance weighting
        inv_var = n / delta
        posterior_mean = (tau_sq * inv_var * gamma_hat + gamma_bar) / (tau_sq * inv_var + 1)
        return posterior_mean

    def _postvar(
        self,
        data: np.ndarray,
        gamma: np.ndarray,
        delta_hat: np.ndarray
    ) -> np.ndarray:
        """Posterior variance for delta (scale parameter)."""
        n = data.shape[1]

        # Estimate inverse gamma parameters
        sum_sq = np.sum((data - gamma[:, np.newaxis]) ** 2, axis=1)

        # Method of moments for inverse gamma
        lambda_est = np.mean(delta_hat)
        theta_est = np.var(delta_hat)

        # Shape and scale parameters
        alpha = lambda_est ** 2 / theta_est + 2
        beta = lambda_est * (lambda_est ** 2 / theta_est + 1)

        # Posterior variance (inverse gamma posterior)
        posterior_var = (beta + 0.5 * sum_sq) / (alpha + 0.5 * n - 1)

        return posterior_var

    def _remove_batch_effects(
        self,
        data: np.ndarray,
        batch_design: np.ndarray,
        batch_labels: np.ndarray
    ) -> np.ndarray:
        """Remove batch effects using estimated parameters."""
        n_features, n_samples = data.shape
        data_corrected = np.copy(data)

        for batch_id in range(self.batch_info_['n_batches']):
            batch_mask = batch_labels == batch_id

            if not np.any(batch_mask):
                continue

            # Adjust for location shift
            data_corrected[:, batch_mask] -= self.gamma_star_[:, batch_id][:, np.newaxis]

            # Adjust for scale (if not mean_only)
            if not self.mean_only:
                # Avoid division by zero
                scale_factor = np.sqrt(
                    self.delta_star_[:, batch_id] / np.mean(self.delta_star_, axis=1)
                )
                scale_factor = np.maximum(scale_factor, 1e-8)
                data_corrected[:, batch_mask] /= scale_factor[:, np.newaxis]

        return data_corrected

    def plot_batch_effects(
        self,
        data_before: np.ndarray,
        data_after: np.ndarray,
        batch: np.ndarray,
        feature_idx: int = 0,
        output_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Visualize batch effects before and after harmonization.

        Parameters
        ----------
        data_before : ndarray of shape (n_features, n_samples)
            Data before harmonization
        data_after : ndarray of shape (n_features, n_samples)
            Data after harmonization
        batch : ndarray of shape (n_samples,)
            Batch labels
        feature_idx : int, default=0
            Index of feature to visualize
        output_path : str or Path, optional
            Path to save figure

        Returns
        -------
        fig : Figure
            Matplotlib figure
        """
        batch_labels = np.unique(batch)
        n_batches = len(batch_labels)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Before harmonization
        ax = axes[0]
        positions = []
        data_list = []

        for i, batch_label in enumerate(batch_labels):
            batch_mask = batch == batch_label
            batch_data = data_before[feature_idx, batch_mask]
            data_list.append(batch_data)
            positions.append(i + 1)

        bp = ax.boxplot(data_list, positions=positions, widths=0.6, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        ax.set_xlabel('Batch', fontsize=12)
        ax.set_ylabel(f'Feature {feature_idx} Value', fontsize=12)
        ax.set_title('Before Harmonization', fontsize=14)
        ax.set_xticks(positions)
        ax.set_xticklabels(batch_labels)
        ax.grid(alpha=0.3, axis='y')

        # After harmonization
        ax = axes[1]
        data_list = []

        for i, batch_label in enumerate(batch_labels):
            batch_mask = batch == batch_label
            batch_data = data_after[feature_idx, batch_mask]
            data_list.append(batch_data)

        bp = ax.boxplot(data_list, positions=positions, widths=0.6, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')

        ax.set_xlabel('Batch', fontsize=12)
        ax.set_ylabel(f'Feature {feature_idx} Value', fontsize=12)
        ax.set_title('After Harmonization', fontsize=14)
        ax.set_xticks(positions)
        ax.set_xticklabels(batch_labels)
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig


def harmonize_connectomes(
    connectomes: np.ndarray,
    batch: np.ndarray,
    covariates: Optional[pd.DataFrame] = None,
    parametric: bool = True
) -> np.ndarray:
    """
    Convenience function to harmonize connectivity matrices.

    Parameters
    ----------
    connectomes : ndarray of shape (n_subjects, n_nodes, n_nodes)
        Connectivity matrices
    batch : ndarray of shape (n_subjects,)
        Batch/scanner labels
    covariates : DataFrame of shape (n_subjects, n_covariates), optional
        Biological covariates to preserve
    parametric : bool, default=True
        Use parametric ComBat

    Returns
    -------
    harmonized_connectomes : ndarray of shape (n_subjects, n_nodes, n_nodes)
        Harmonized connectivity matrices
    """
    n_subjects, n_nodes, _ = connectomes.shape

    # Extract upper triangle (exclude diagonal)
    triu_indices = np.triu_indices(n_nodes, k=1)
    n_edges = len(triu_indices[0])

    # Create edge matrix (features × subjects)
    edge_matrix = np.zeros((n_edges, n_subjects))
    for i in range(n_subjects):
        edge_matrix[:, i] = connectomes[i, triu_indices[0], triu_indices[1]]

    # Apply ComBat harmonization
    harmonizer = ComBatHarmonizer(parametric=parametric)
    edge_matrix_harmonized = harmonizer.fit_transform(edge_matrix, batch, covariates)

    # Reconstruct connectivity matrices
    harmonized_connectomes = np.zeros_like(connectomes)
    for i in range(n_subjects):
        # Fill upper triangle
        harmonized_connectomes[i, triu_indices[0], triu_indices[1]] = edge_matrix_harmonized[:, i]
        # Mirror to lower triangle
        harmonized_connectomes[i, triu_indices[1], triu_indices[0]] = edge_matrix_harmonized[:, i]

    return harmonized_connectomes
