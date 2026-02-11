"""
Pathology Detection Module
==========================

Machine learning models for disease classification from neuroimaging connectome data.

Features:
- Logistic Regression with L1/L2 regularization for feature selection
- Random Forest Classifier with SHAP interpretability
- Cross-validation with stratified splits
- Permutation-based feature importance
- ROC-AUC and precision-recall curves
- Calibration plots
- Age/sex/scanner confound regression

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import pickle
import json
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    learning_curve,
    permutation_test_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    calibration_curve,
    brier_score_loss
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")


class ConfoundRegressor:
    """Remove confounding effects from features using linear regression."""

    def __init__(self, confound_names: Optional[List[str]] = None):
        """
        Initialize confound regressor.

        Parameters
        ----------
        confound_names : list of str, optional
            Names of confound variables (e.g., ['age', 'sex', 'scanner'])
        """
        self.confound_names = confound_names or ['age', 'sex', 'scanner']
        self.coefficients_ = None
        self.feature_means_ = None

    def fit(self, X: np.ndarray, confounds: pd.DataFrame) -> 'ConfoundRegressor':
        """
        Fit confound regression model.

        Parameters
        ----------
        X : ndarray of shape (n_subjects, n_features)
            Feature matrix
        confounds : DataFrame of shape (n_subjects, n_confounds)
            Confound variables

        Returns
        -------
        self : ConfoundRegressor
        """
        # Encode categorical variables
        confounds_encoded = self._encode_confounds(confounds)

        # Add intercept
        confounds_with_intercept = np.column_stack([
            np.ones(len(confounds_encoded)),
            confounds_encoded
        ])

        # Fit linear regression for each feature
        # X = beta * confounds + residuals
        self.coefficients_ = np.linalg.lstsq(
            confounds_with_intercept, X, rcond=None
        )[0]

        self.feature_means_ = np.mean(X, axis=0)

        return self

    def transform(self, X: np.ndarray, confounds: pd.DataFrame) -> np.ndarray:
        """
        Remove confound effects from features.

        Parameters
        ----------
        X : ndarray of shape (n_subjects, n_features)
            Feature matrix
        confounds : DataFrame of shape (n_subjects, n_confounds)
            Confound variables

        Returns
        -------
        X_deconfounded : ndarray of shape (n_subjects, n_features)
            Feature matrix with confound effects removed
        """
        if self.coefficients_ is None:
            raise ValueError("Must call fit() before transform()")

        confounds_encoded = self._encode_confounds(confounds)
        confounds_with_intercept = np.column_stack([
            np.ones(len(confounds_encoded)),
            confounds_encoded
        ])

        # Predict confound effects
        confound_effects = confounds_with_intercept @ self.coefficients_

        # Remove confound effects and add back global mean
        X_deconfounded = X - confound_effects + self.feature_means_

        return X_deconfounded

    def fit_transform(self, X: np.ndarray, confounds: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, confounds).transform(X, confounds)

    def _encode_confounds(self, confounds: pd.DataFrame) -> np.ndarray:
        """Encode confounds (one-hot for categorical, standardize for continuous)."""
        encoded = []

        for col in self.confound_names:
            if col not in confounds.columns:
                continue

            values = confounds[col].values

            # Check if categorical
            if confounds[col].dtype == 'object' or len(np.unique(values)) < 10:
                # One-hot encoding (drop first to avoid collinearity)
                unique_vals = np.unique(values)
                for val in unique_vals[1:]:
                    encoded.append((values == val).astype(float))
            else:
                # Standardize continuous variables
                encoded.append((values - np.mean(values)) / np.std(values))

        return np.column_stack(encoded) if encoded else np.zeros((len(confounds), 1))


class PathologyClassifier:
    """
    ML classifier for pathology detection from connectome features.

    Supports multiple models with cross-validation and interpretability.
    """

    def __init__(
        self,
        model_type: str = 'logistic',
        penalty: str = 'l2',
        C: float = 1.0,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize pathology classifier.

        Parameters
        ----------
        model_type : str, default='logistic'
            Type of model: 'logistic' or 'random_forest'
        penalty : str, default='l2'
            Regularization for logistic regression: 'l1', 'l2', or 'elasticnet'
        C : float, default=1.0
            Inverse regularization strength
        n_estimators : int, default=100
            Number of trees for random forest
        max_depth : int, optional
            Maximum depth of trees
        random_state : int, default=42
            Random seed for reproducibility
        n_jobs : int, default=-1
            Number of parallel jobs
        """
        self.model_type = model_type
        self.penalty = penalty
        self.C = C
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Initialize model
        if model_type == 'logistic':
            solver = 'liblinear' if penalty in ['l1', 'l2'] else 'saga'
            self.model = LogisticRegression(
                penalty=penalty,
                C=C,
                solver=solver,
                max_iter=1000,
                random_state=random_state,
                n_jobs=n_jobs
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=n_jobs,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.classes_ = None
        self.is_fitted_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'PathologyClassifier':
        """
        Fit the classifier.

        Parameters
        ----------
        X : ndarray of shape (n_subjects, n_features)
            Training features
        y : ndarray of shape (n_subjects,)
            Training labels
        feature_names : list of str, optional
            Names of features

        Returns
        -------
        self : PathologyClassifier
        """
        # Store feature names
        self.feature_names_ = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self.classes_ = np.unique(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted_ = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self._check_fitted()
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self._check_fitted()
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform stratified cross-validation.

        Parameters
        ----------
        X : ndarray of shape (n_subjects, n_features)
            Features
        y : ndarray of shape (n_subjects,)
            Labels
        cv : int, default=5
            Number of cross-validation folds
        feature_names : list of str, optional
            Names of features

        Returns
        -------
        results : dict
            Cross-validation results with metrics
        """
        self.feature_names_ = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Define cross-validation strategy
        cv_splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )

        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'roc_auc': 'roc_auc',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        }

        # Perform cross-validation
        cv_results = cross_validate(
            self.model,
            X_scaled,
            y,
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,
            n_jobs=self.n_jobs
        )

        # Compile results
        results = {
            'test_accuracy': cv_results['test_accuracy'],
            'test_roc_auc': cv_results['test_roc_auc'],
            'test_precision': cv_results['test_precision'],
            'test_recall': cv_results['test_recall'],
            'test_f1': cv_results['test_f1'],
            'train_accuracy': cv_results['train_accuracy'],
            'train_roc_auc': cv_results['train_roc_auc'],
            'estimators': cv_results['estimator'],
            'mean_test_accuracy': np.mean(cv_results['test_accuracy']),
            'std_test_accuracy': np.std(cv_results['test_accuracy']),
            'mean_test_roc_auc': np.mean(cv_results['test_roc_auc']),
            'std_test_roc_auc': np.std(cv_results['test_roc_auc']),
        }

        return results

    def permutation_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_permutations: int = 1000,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Perform permutation test to assess statistical significance.

        Parameters
        ----------
        X : ndarray of shape (n_subjects, n_features)
            Features
        y : ndarray of shape (n_subjects,)
            Labels
        n_permutations : int, default=1000
            Number of permutations
        cv : int, default=5
            Number of cross-validation folds

        Returns
        -------
        results : dict
            Permutation test results
        """
        X_scaled = self.scaler.fit_transform(X)

        cv_splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )

        score, perm_scores, pvalue = permutation_test_score(
            self.model,
            X_scaled,
            y,
            scoring='roc_auc',
            cv=cv_splitter,
            n_permutations=n_permutations,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

        results = {
            'score': score,
            'permutation_scores': perm_scores,
            'pvalue': pvalue,
            'n_permutations': n_permutations
        }

        return results

    def get_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'permutation',
        n_repeats: int = 10
    ) -> pd.DataFrame:
        """
        Get feature importance scores.

        Parameters
        ----------
        X : ndarray of shape (n_subjects, n_features)
            Features
        y : ndarray of shape (n_subjects,)
            Labels
        method : str, default='permutation'
            Method: 'permutation', 'coefficients', or 'tree'
        n_repeats : int, default=10
            Number of repeats for permutation importance

        Returns
        -------
        importance_df : DataFrame
            Feature importance scores
        """
        self._check_fitted()
        X_scaled = self.scaler.transform(X)

        if method == 'permutation':
            # Permutation importance (model-agnostic)
            result = permutation_importance(
                self.model,
                X_scaled,
                y,
                n_repeats=n_repeats,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                scoring='roc_auc'
            )

            importance_df = pd.DataFrame({
                'feature': self.feature_names_,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            })

        elif method == 'coefficients' and self.model_type == 'logistic':
            # Logistic regression coefficients
            coefs = np.abs(self.model.coef_[0])

            importance_df = pd.DataFrame({
                'feature': self.feature_names_,
                'importance_mean': coefs,
                'importance_std': np.zeros_like(coefs)
            })

        elif method == 'tree' and self.model_type == 'random_forest':
            # Random forest feature importances
            importances = self.model.feature_importances_

            # Get standard deviation from trees
            tree_importances = np.array([
                tree.feature_importances_ for tree in self.model.estimators_
            ])
            importance_std = np.std(tree_importances, axis=0)

            importance_df = pd.DataFrame({
                'feature': self.feature_names_,
                'importance_mean': importances,
                'importance_std': importance_std
            })

        else:
            raise ValueError(f"Invalid method '{method}' for model type '{self.model_type}'")

        # Sort by importance
        importance_df = importance_df.sort_values('importance_mean', ascending=False)

        return importance_df

    def plot_roc_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[float, plt.Figure]:
        """
        Plot ROC curve.

        Parameters
        ----------
        X : ndarray of shape (n_subjects, n_features)
            Features
        y : ndarray of shape (n_subjects,)
            True labels
        output_path : str or Path, optional
            Path to save figure

        Returns
        -------
        auc : float
            Area under ROC curve
        fig : Figure
            Matplotlib figure
        """
        self._check_fitted()

        # Get predictions
        y_proba = self.predict_proba(X)[:, 1]

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

        return auc, fig

    def plot_precision_recall_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[float, plt.Figure]:
        """
        Plot precision-recall curve.

        Parameters
        ----------
        X : ndarray of shape (n_subjects, n_features)
            Features
        y : ndarray of shape (n_subjects,)
            True labels
        output_path : str or Path, optional
            Path to save figure

        Returns
        -------
        ap : float
            Average precision score
        fig : Figure
            Matplotlib figure
        """
        self._check_fitted()

        # Get predictions
        y_proba = self.predict_proba(X)[:, 1]

        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y, y_proba)
        ap = average_precision_score(y, y_proba)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2, label=f'PR curve (AP = {ap:.3f})')
        ax.axhline(np.mean(y), color='k', linestyle='--', linewidth=1, label='Baseline')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14)
        ax.legend(loc='best', fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

        return ap, fig

    def plot_calibration_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[float, plt.Figure]:
        """
        Plot calibration curve to assess probability calibration.

        Parameters
        ----------
        X : ndarray of shape (n_subjects, n_features)
            Features
        y : ndarray of shape (n_subjects,)
            True labels
        n_bins : int, default=10
            Number of bins for calibration curve
        output_path : str or Path, optional
            Path to save figure

        Returns
        -------
        brier : float
            Brier score (lower is better)
        fig : Figure
            Matplotlib figure
        """
        self._check_fitted()

        # Get predictions
        y_proba = self.predict_proba(X)[:, 1]

        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=n_bins, strategy='uniform')
        brier = brier_score_loss(y, y_proba)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8,
                label=f'Model (Brier = {brier:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('True Probability', fontsize=12)
        ax.set_title('Calibration Curve', fontsize=14)
        ax.legend(loc='best', fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

        return brier, fig

    def get_shap_values(
        self,
        X: np.ndarray,
        n_background: int = 100
    ) -> Optional[np.ndarray]:
        """
        Compute SHAP values for model interpretability.

        Parameters
        ----------
        X : ndarray of shape (n_subjects, n_features)
            Features to explain
        n_background : int, default=100
            Number of background samples for SHAP

        Returns
        -------
        shap_values : ndarray or None
            SHAP values if SHAP is available, else None
        """
        if not SHAP_AVAILABLE:
            warnings.warn("SHAP not available. Install with: pip install shap")
            return None

        self._check_fitted()
        X_scaled = self.scaler.transform(X)

        # Create background dataset
        if len(X_scaled) > n_background:
            background_indices = np.random.choice(
                len(X_scaled), n_background, replace=False
            )
            background = X_scaled[background_indices]
        else:
            background = X_scaled

        # Create SHAP explainer
        if self.model_type == 'logistic':
            explainer = shap.LinearExplainer(self.model, background)
        elif self.model_type == 'random_forest':
            explainer = shap.TreeExplainer(self.model)
        else:
            explainer = shap.KernelExplainer(self.model.predict_proba, background)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_scaled)

        # For binary classification, extract positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return shap_values

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names_,
            'classes': self.classes_,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted_
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, filepath: Union[str, Path]) -> 'PathologyClassifier':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names_ = model_data['feature_names']
        self.classes_ = model_data['classes']
        self.model_type = model_data['model_type']
        self.is_fitted_ = model_data['is_fitted']

        return self

    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")


def create_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Create comprehensive classification report.

    Parameters
    ----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    y_proba : ndarray
        Predicted probabilities
    class_names : list of str, optional
        Class names
    output_path : str or Path, optional
        Path to save report

    Returns
    -------
    report : dict
        Classification metrics
    """
    # Compute metrics
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    # Classification report
    clf_report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )

    report = {
        'confusion_matrix': cm.tolist(),
        'roc_auc': auc,
        'average_precision': ap,
        'classification_report': clf_report
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

    return report
