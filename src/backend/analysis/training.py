"""
Training Pipeline Module
========================

Complete ML training pipeline for pathology detection from connectome features.

Features:
- Load connectome features from HDF5/CSV
- Subject-level train/validation/test splits
- Hyperparameter tuning with GridSearchCV
- Model training and evaluation
- Save trained models
- Generate comprehensive performance reports

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import pickle
import warnings
from datetime import datetime

import h5py
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from .pathology_detection import PathologyClassifier, create_classification_report
from .harmonization import harmonize_connectomes
from .statistical_analysis import compute_effect_size


class ConnectomeDataLoader:
    """Load and prepare connectome data for ML."""

    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize data loader.

        Parameters
        ----------
        data_path : str or Path
            Path to data file (HDF5 or CSV)
        """
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    def load_connectomes(
        self,
        feature_type: str = 'edges'
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load connectome features.

        Parameters
        ----------
        feature_type : str, default='edges'
            Type of features: 'edges', 'graph_metrics', or 'both'

        Returns
        -------
        features : ndarray of shape (n_subjects, n_features)
            Feature matrix
        metadata : DataFrame
            Subject metadata (IDs, labels, covariates)
        """
        if self.data_path.suffix == '.h5' or self.data_path.suffix == '.hdf5':
            return self._load_from_hdf5(feature_type)
        elif self.data_path.suffix == '.csv':
            return self._load_from_csv(feature_type)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

    def _load_from_hdf5(self, feature_type: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load from HDF5 file."""
        with h5py.File(self.data_path, 'r') as f:
            # Load metadata
            if 'metadata' in f:
                metadata = pd.DataFrame({
                    key: f['metadata'][key][:] for key in f['metadata'].keys()
                })
            else:
                raise ValueError("HDF5 file must contain 'metadata' group")

            # Load features
            if feature_type == 'edges':
                if 'connectomes' in f:
                    # Extract upper triangle from matrices
                    connectomes = f['connectomes'][:]
                    n_subjects, n_nodes, _ = connectomes.shape
                    triu_indices = np.triu_indices(n_nodes, k=1)
                    features = connectomes[:, triu_indices[0], triu_indices[1]]
                elif 'edge_features' in f:
                    features = f['edge_features'][:]
                else:
                    raise ValueError("HDF5 must contain 'connectomes' or 'edge_features'")

            elif feature_type == 'graph_metrics':
                if 'graph_metrics' in f:
                    features = f['graph_metrics'][:]
                else:
                    raise ValueError("HDF5 must contain 'graph_metrics'")

            elif feature_type == 'both':
                # Concatenate edges and graph metrics
                if 'connectomes' in f:
                    connectomes = f['connectomes'][:]
                    n_subjects, n_nodes, _ = connectomes.shape
                    triu_indices = np.triu_indices(n_nodes, k=1)
                    edge_features = connectomes[:, triu_indices[0], triu_indices[1]]
                elif 'edge_features' in f:
                    edge_features = f['edge_features'][:]
                else:
                    raise ValueError("Missing edge features")

                if 'graph_metrics' in f:
                    graph_features = f['graph_metrics'][:]
                    features = np.hstack([edge_features, graph_features])
                else:
                    features = edge_features
            else:
                raise ValueError(f"Invalid feature_type: {feature_type}")

        return features, metadata

    def _load_from_csv(self, feature_type: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load from CSV file."""
        df = pd.read_csv(self.data_path)

        # Separate features and metadata
        metadata_cols = ['subject_id', 'label', 'age', 'sex', 'scanner', 'site', 'diagnosis']
        metadata_cols = [col for col in metadata_cols if col in df.columns]

        metadata = df[metadata_cols].copy()
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        features = df[feature_cols].values

        return features, metadata


class ModelTrainer:
    """Train and evaluate pathology detection models."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        random_state: int = 42
    ):
        """
        Initialize model trainer.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save models and results
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        self.best_model_ = None
        self.best_params_ = None
        self.feature_names_ = None
        self.label_encoder_ = None

    def prepare_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.2,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/validation/test sets.

        Parameters
        ----------
        features : ndarray of shape (n_subjects, n_features)
            Feature matrix
        labels : ndarray of shape (n_subjects,)
            Labels
        test_size : float, default=0.2
            Proportion of test set
        val_size : float, default=0.2
            Proportion of validation set (from remaining data)
        stratify : bool, default=True
            Use stratified splitting

        Returns
        -------
        X_train : ndarray
            Training features
        X_val : ndarray
            Validation features
        X_test : ndarray
            Test features
        y_train : ndarray
            Training labels
        y_val : ndarray
            Validation labels
        y_test : ndarray
            Test labels
        """
        # First split: separate test set
        stratify_test = labels if stratify else None

        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels,
            test_size=test_size,
            stratify=stratify_test,
            random_state=self.random_state
        )

        # Second split: separate train and validation
        stratify_val = y_temp if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            stratify=stratify_val,
            random_state=self.random_state
        )

        print(f"Data split:")
        print(f"  Training:   {len(X_train)} subjects")
        print(f"  Validation: {len(X_val)} subjects")
        print(f"  Test:       {len(X_test)} subjects")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = 'logistic',
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning with grid search.

        Parameters
        ----------
        X_train : ndarray
            Training features
        y_train : ndarray
            Training labels
        model_type : str, default='logistic'
            Model type: 'logistic' or 'random_forest'
        param_grid : dict, optional
            Parameter grid for search
        cv : int, default=5
            Number of cross-validation folds
        n_jobs : int, default=-1
            Number of parallel jobs

        Returns
        -------
        results : dict
            Grid search results
        """
        print(f"\nTuning {model_type} hyperparameters...")

        # Default parameter grids
        if param_grid is None:
            if model_type == 'logistic':
                param_grid = {
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                }
            elif model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

        # Create base classifier
        base_classifier = PathologyClassifier(
            model_type=model_type,
            random_state=self.random_state,
            n_jobs=n_jobs
        )

        # Grid search with cross-validation
        cv_splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )

        grid_search = GridSearchCV(
            base_classifier.model,
            param_grid,
            cv=cv_splitter,
            scoring='roc_auc',
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Fit grid search
        grid_search.fit(X_train_scaled, y_train)

        # Store best parameters
        self.best_params_ = grid_search.best_params_

        print(f"Best parameters: {self.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.3f}")

        # Create results dictionary
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': pd.DataFrame(grid_search.cv_results_),
            'grid_search': grid_search
        }

        return results

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = 'logistic',
        params: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None
    ) -> PathologyClassifier:
        """
        Train final model with best parameters.

        Parameters
        ----------
        X_train : ndarray
            Training features
        y_train : ndarray
            Training labels
        model_type : str, default='logistic'
            Model type
        params : dict, optional
            Model parameters (use best_params_ if None)
        feature_names : list of str, optional
            Feature names

        Returns
        -------
        model : PathologyClassifier
            Trained model
        """
        print(f"\nTraining {model_type} model...")

        # Use best parameters if available
        if params is None and self.best_params_ is not None:
            params = self.best_params_

        # Create and train classifier
        if params:
            classifier = PathologyClassifier(
                model_type=model_type,
                random_state=self.random_state,
                **params
            )
        else:
            classifier = PathologyClassifier(
                model_type=model_type,
                random_state=self.random_state
            )

        classifier.fit(X_train, y_train, feature_names=feature_names)

        self.best_model_ = classifier
        self.feature_names_ = feature_names

        print("Model trained successfully!")

        return classifier

    def evaluate_model(
        self,
        model: PathologyClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset_name: str = 'test'
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Parameters
        ----------
        model : PathologyClassifier
            Trained model
        X_test : ndarray
            Test features
        y_test : ndarray
            Test labels
        dataset_name : str, default='test'
            Name of dataset being evaluated

        Returns
        -------
        metrics : dict
            Evaluation metrics
        """
        print(f"\nEvaluating model on {dataset_name} set...")

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }

        print(f"Performance on {dataset_name} set:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-score:  {metrics['f1']:.3f}")

        return metrics

    def generate_report(
        self,
        model: PathologyClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Parameters
        ----------
        model : PathologyClassifier
            Trained model
        X_train, y_train : ndarray
            Training data
        X_val, y_val : ndarray
            Validation data
        X_test, y_test : ndarray
            Test data
        class_names : list of str, optional
            Class names

        Returns
        -------
        report : dict
            Comprehensive performance report
        """
        print("\nGenerating performance report...")

        # Evaluate on all sets
        train_metrics = self.evaluate_model(model, X_train, y_train, 'train')
        val_metrics = self.evaluate_model(model, X_val, y_val, 'validation')
        test_metrics = self.evaluate_model(model, X_test, y_test, 'test')

        # Get predictions
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Generate plots
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        # ROC curve
        roc_auc, roc_fig = model.plot_roc_curve(
            X_test, y_test,
            output_path=plots_dir / 'roc_curve.png'
        )
        plt.close(roc_fig)

        # Precision-recall curve
        pr_auc, pr_fig = model.plot_precision_recall_curve(
            X_test, y_test,
            output_path=plots_dir / 'precision_recall_curve.png'
        )
        plt.close(pr_fig)

        # Calibration curve
        brier, cal_fig = model.plot_calibration_curve(
            X_test, y_test,
            output_path=plots_dir / 'calibration_curve.png'
        )
        plt.close(cal_fig)

        # Feature importance
        importance_df = model.get_feature_importance(
            X_test, y_test, method='permutation', n_repeats=10
        )

        # Save feature importance
        importance_df.to_csv(self.output_dir / 'feature_importance.csv', index=False)

        # Compile report
        report = {
            'model_type': model.model_type,
            'timestamp': datetime.now().isoformat(),
            'data_splits': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            },
            'metrics': {
                'train': train_metrics,
                'validation': val_metrics,
                'test': test_metrics
            },
            'feature_importance': {
                'top_10_features': importance_df.head(10).to_dict('records')
            },
            'classification_report': classification_report(
                y_test, y_test_pred,
                target_names=class_names,
                output_dict=True
            )
        }

        # Save report
        report_path = self.output_dir / 'performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved to: {report_path}")

        return report

    def save_model(
        self,
        model: PathologyClassifier,
        filename: str = 'model.pkl'
    ) -> Path:
        """
        Save trained model to disk.

        Parameters
        ----------
        model : PathologyClassifier
            Trained model to save
        filename : str, default='model.pkl'
            Output filename

        Returns
        -------
        model_path : Path
            Path to saved model
        """
        model_path = self.output_dir / filename
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        return model_path


def train_pathology_classifier(
    data_path: Union[str, Path],
    output_dir: Union[str, Path],
    feature_type: str = 'edges',
    label_column: str = 'label',
    model_type: str = 'logistic',
    harmonize: bool = False,
    batch_column: Optional[str] = None,
    covariate_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    tune_hyperparams: bool = True,
    param_grid: Optional[Dict] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Complete training pipeline for pathology classification.

    Parameters
    ----------
    data_path : str or Path
        Path to data file
    output_dir : str or Path
        Output directory for models and reports
    feature_type : str, default='edges'
        Feature type to use
    label_column : str, default='label'
        Name of label column in metadata
    model_type : str, default='logistic'
        Model type: 'logistic' or 'random_forest'
    harmonize : bool, default=False
        Apply ComBat harmonization
    batch_column : str, optional
        Batch/scanner column for harmonization
    covariate_columns : list of str, optional
        Covariate columns to preserve during harmonization
    test_size : float, default=0.2
        Test set proportion
    val_size : float, default=0.2
        Validation set proportion
    tune_hyperparams : bool, default=True
        Perform hyperparameter tuning
    param_grid : dict, optional
        Custom parameter grid
    random_state : int, default=42
        Random seed

    Returns
    -------
    results : dict
        Training results
    """
    print("=" * 60)
    print("NeuroTract Pathology Classification Training Pipeline")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    loader = ConnectomeDataLoader(data_path)
    features, metadata = loader.load_connectomes(feature_type=feature_type)

    print(f"Loaded {len(features)} subjects with {features.shape[1]} features")

    # Extract labels
    if label_column not in metadata.columns:
        raise ValueError(f"Label column '{label_column}' not found in metadata")

    labels = metadata[label_column].values

    # Harmonization
    if harmonize:
        if batch_column is None:
            raise ValueError("Must specify batch_column for harmonization")

        print("\nApplying ComBat harmonization...")
        batch = metadata[batch_column].values

        covariates = None
        if covariate_columns:
            covariates = metadata[covariate_columns]

        # Harmonize (assumes features are edges)
        features = features.T  # Transpose to (features, subjects)
        features = harmonize_connectomes(
            features[np.newaxis, :, :], batch, covariates
        )[0]
        features = features.T  # Back to (subjects, features)

    # Initialize trainer
    trainer = ModelTrainer(output_dir, random_state=random_state)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        features, labels, test_size=test_size, val_size=val_size
    )

    # Hyperparameter tuning
    if tune_hyperparams:
        tuning_results = trainer.tune_hyperparameters(
            X_train, y_train,
            model_type=model_type,
            param_grid=param_grid
        )
    else:
        tuning_results = None

    # Train model
    model = trainer.train_model(X_train, y_train, model_type=model_type)

    # Generate report
    report = trainer.generate_report(
        model, X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Save model
    model_path = trainer.save_model(model)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

    return {
        'model': model,
        'model_path': model_path,
        'report': report,
        'tuning_results': tuning_results
    }
