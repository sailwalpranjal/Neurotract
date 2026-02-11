"""
Prediction Pipeline Module
==========================

Inference pipeline for pathology detection on new subjects.

Features:
- Load pre-trained models
- Predict on new subjects
- Generate confidence scores
- Create explanations (SHAP values)
- Batch prediction support
- Uncertainty quantification

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import pickle
import warnings
from dataclasses import dataclass, asdict

from .pathology_detection import PathologyClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


@dataclass
class PredictionResult:
    """Container for prediction results."""
    subject_id: str
    predicted_label: int
    predicted_class: str
    confidence: float
    probability_scores: Dict[str, float]
    feature_contributions: Optional[Dict[str, float]] = None
    uncertainty: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class PathologyPredictor:
    """
    Inference pipeline for pathology detection.

    Loads trained models and makes predictions on new subjects.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize predictor.

        Parameters
        ----------
        model_path : str or Path
            Path to trained model file
        class_names : list of str, optional
            Names of classes (e.g., ['Control', 'Patient'])
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load model
        print(f"Loading model from {self.model_path}...")
        self.model = PathologyClassifier()
        self.model.load(self.model_path)

        # Set class names
        if class_names is None:
            # Default names
            self.class_names = [f"Class_{i}" for i in range(len(self.model.classes_))]
        else:
            self.class_names = class_names

        print(f"Model loaded successfully!")
        print(f"  Model type: {self.model.model_type}")
        print(f"  Classes: {self.class_names}")
        print(f"  Number of features: {len(self.model.feature_names_)}")

    def predict_single(
        self,
        features: np.ndarray,
        subject_id: str = "unknown",
        compute_shap: bool = False,
        compute_uncertainty: bool = False
    ) -> PredictionResult:
        """
        Predict pathology for a single subject.

        Parameters
        ----------
        features : ndarray of shape (n_features,)
            Feature vector for subject
        subject_id : str, default="unknown"
            Subject identifier
        compute_shap : bool, default=False
            Compute SHAP values for explanation
        compute_uncertainty : bool, default=False
            Compute prediction uncertainty

        Returns
        -------
        result : PredictionResult
            Prediction result with confidence and explanations
        """
        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Predict
        y_pred = self.model.predict(features)[0]
        y_proba = self.model.predict_proba(features)[0]

        # Confidence (max probability)
        confidence = float(np.max(y_proba))

        # Probability scores for each class
        probability_scores = {
            self.class_names[i]: float(y_proba[i])
            for i in range(len(self.class_names))
        }

        # Predicted class name
        predicted_class = self.class_names[y_pred]

        # Feature contributions (SHAP values)
        feature_contributions = None
        if compute_shap:
            shap_values = self.model.get_shap_values(features)
            if shap_values is not None:
                # Get top contributing features
                shap_abs = np.abs(shap_values[0])
                top_indices = np.argsort(shap_abs)[-10:][::-1]

                feature_contributions = {
                    self.model.feature_names_[idx]: float(shap_values[0, idx])
                    for idx in top_indices
                }

        # Uncertainty estimation
        uncertainty = None
        if compute_uncertainty:
            # Use entropy of probability distribution
            epsilon = 1e-10
            uncertainty = -np.sum(y_proba * np.log(y_proba + epsilon))

        result = PredictionResult(
            subject_id=subject_id,
            predicted_label=int(y_pred),
            predicted_class=predicted_class,
            confidence=confidence,
            probability_scores=probability_scores,
            feature_contributions=feature_contributions,
            uncertainty=float(uncertainty) if uncertainty is not None else None
        )

        return result

    def predict_batch(
        self,
        features: np.ndarray,
        subject_ids: Optional[List[str]] = None,
        compute_shap: bool = False,
        compute_uncertainty: bool = False
    ) -> List[PredictionResult]:
        """
        Predict pathology for multiple subjects.

        Parameters
        ----------
        features : ndarray of shape (n_subjects, n_features)
            Feature matrix
        subject_ids : list of str, optional
            Subject identifiers
        compute_shap : bool, default=False
            Compute SHAP values
        compute_uncertainty : bool, default=False
            Compute prediction uncertainty

        Returns
        -------
        results : list of PredictionResult
            Prediction results for all subjects
        """
        n_subjects = len(features)

        if subject_ids is None:
            subject_ids = [f"subject_{i}" for i in range(n_subjects)]

        results = []
        for i in range(n_subjects):
            result = self.predict_single(
                features[i],
                subject_id=subject_ids[i],
                compute_shap=compute_shap,
                compute_uncertainty=compute_uncertainty
            )
            results.append(result)

        return results

    def predict_from_connectome(
        self,
        connectome: np.ndarray,
        subject_id: str = "unknown",
        compute_shap: bool = False
    ) -> PredictionResult:
        """
        Predict from connectivity matrix.

        Parameters
        ----------
        connectome : ndarray of shape (n_nodes, n_nodes)
            Connectivity matrix
        subject_id : str, default="unknown"
            Subject identifier
        compute_shap : bool, default=False
            Compute SHAP values

        Returns
        -------
        result : PredictionResult
            Prediction result
        """
        # Extract upper triangle (excluding diagonal)
        n_nodes = connectome.shape[0]
        triu_indices = np.triu_indices(n_nodes, k=1)
        features = connectome[triu_indices[0], triu_indices[1]]

        return self.predict_single(features, subject_id, compute_shap)

    def explain_prediction(
        self,
        features: np.ndarray,
        subject_id: str = "unknown",
        n_features: int = 20,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[Dict[str, float], plt.Figure]:
        """
        Generate detailed explanation for prediction.

        Parameters
        ----------
        features : ndarray of shape (n_features,)
            Feature vector
        subject_id : str, default="unknown"
            Subject identifier
        n_features : int, default=20
            Number of top features to show
        output_path : str or Path, optional
            Path to save explanation plot

        Returns
        -------
        explanation : dict
            Feature importance scores
        fig : Figure
            Explanation figure
        """
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get SHAP values
        shap_values = self.model.get_shap_values(features)

        if shap_values is None:
            warnings.warn("SHAP not available, using permutation importance")
            # Fallback to feature importance
            importance_df = self.model.get_feature_importance(
                features, np.array([0]), method='permutation'
            )
            top_features = importance_df.head(n_features)

            explanation = dict(zip(
                top_features['feature'],
                top_features['importance_mean']
            ))

            # Create simple bar plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(top_features)), top_features['importance_mean'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Importance')
            ax.set_title(f'Feature Importance for {subject_id}')
            ax.invert_yaxis()
            plt.tight_layout()

        else:
            # Use SHAP values
            shap_values_single = shap_values[0]

            # Get top features by absolute SHAP value
            shap_abs = np.abs(shap_values_single)
            top_indices = np.argsort(shap_abs)[-n_features:][::-1]

            top_features_names = [self.model.feature_names_[i] for i in top_indices]
            top_shap_values = [shap_values_single[i] for i in top_indices]

            explanation = dict(zip(top_features_names, top_shap_values))

            # Create SHAP-style waterfall plot
            fig, ax = plt.subplots(figsize=(10, 8))

            colors = ['red' if v > 0 else 'blue' for v in top_shap_values]
            ax.barh(range(len(top_shap_values)), top_shap_values, color=colors, alpha=0.7)
            ax.set_yticks(range(len(top_features_names)))
            ax.set_yticklabels(top_features_names)
            ax.set_xlabel('SHAP Value (impact on prediction)', fontsize=11)
            ax.set_title(f'Feature Contributions for {subject_id}', fontsize=13)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.invert_yaxis()

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.7, label='Positive contribution'),
                Patch(facecolor='blue', alpha=0.7, label='Negative contribution')
            ]
            ax.legend(handles=legend_elements, loc='lower right')

            plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

        return explanation, fig

    def save_predictions(
        self,
        results: List[PredictionResult],
        output_path: Union[str, Path],
        format: str = 'csv'
    ) -> None:
        """
        Save predictions to file.

        Parameters
        ----------
        results : list of PredictionResult
            Prediction results
        output_path : str or Path
            Output file path
        format : str, default='csv'
            Output format: 'csv' or 'json'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'csv':
            # Convert to DataFrame
            data = []
            for result in results:
                row = {
                    'subject_id': result.subject_id,
                    'predicted_label': result.predicted_label,
                    'predicted_class': result.predicted_class,
                    'confidence': result.confidence,
                }

                # Add probability scores
                for class_name, prob in result.probability_scores.items():
                    row[f'prob_{class_name}'] = prob

                # Add uncertainty if available
                if result.uncertainty is not None:
                    row['uncertainty'] = result.uncertainty

                data.append(row)

            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)

        elif format == 'json':
            # Save as JSON
            data = [result.to_dict() for result in results]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Predictions saved to: {output_path}")

    def generate_prediction_report(
        self,
        results: List[PredictionResult],
        output_dir: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive prediction report.

        Parameters
        ----------
        results : list of PredictionResult
            Prediction results
        output_dir : str or Path
            Output directory

        Returns
        -------
        report : dict
            Prediction report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute statistics
        n_subjects = len(results)
        predicted_classes = [r.predicted_class for r in results]
        confidences = [r.confidence for r in results]

        # Class distribution
        class_counts = pd.Series(predicted_classes).value_counts().to_dict()

        # Confidence statistics
        confidence_stats = {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'median': float(np.median(confidences))
        }

        # Create report
        report = {
            'n_subjects': n_subjects,
            'model_type': self.model.model_type,
            'class_names': self.class_names,
            'class_distribution': class_counts,
            'confidence_statistics': confidence_stats
        }

        # Save predictions
        self.save_predictions(results, output_dir / 'predictions.csv', format='csv')
        self.save_predictions(results, output_dir / 'predictions.json', format='json')

        # Plot class distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Class distribution
        ax = axes[0]
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        ax.bar(classes, counts, color='steelblue', alpha=0.7)
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Predicted Class Distribution', fontsize=13)
        ax.grid(alpha=0.3, axis='y')

        # Confidence distribution
        ax = axes[1]
        ax.hist(confidences, bins=20, color='coral', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(confidences), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Prediction Confidence Distribution', fontsize=13)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        fig.savefig(output_dir / 'prediction_summary.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Save report
        report_path = output_dir / 'prediction_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nPrediction report saved to: {output_dir}")

        return report


def predict_from_model(
    model_path: Union[str, Path],
    features: np.ndarray,
    subject_ids: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    compute_shap: bool = False,
    generate_report: bool = True
) -> List[PredictionResult]:
    """
    Convenience function for batch prediction.

    Parameters
    ----------
    model_path : str or Path
        Path to trained model
    features : ndarray of shape (n_subjects, n_features)
        Feature matrix
    subject_ids : list of str, optional
        Subject identifiers
    class_names : list of str, optional
        Class names
    output_dir : str or Path, optional
        Output directory for results
    compute_shap : bool, default=False
        Compute SHAP values
    generate_report : bool, default=True
        Generate prediction report

    Returns
    -------
    results : list of PredictionResult
        Prediction results
    """
    # Initialize predictor
    predictor = PathologyPredictor(model_path, class_names)

    # Make predictions
    print(f"\nPredicting for {len(features)} subjects...")
    results = predictor.predict_batch(
        features, subject_ids, compute_shap=compute_shap
    )

    # Generate report
    if generate_report and output_dir is not None:
        predictor.generate_prediction_report(results, output_dir)

    # Print summary
    print("\nPrediction Summary:")
    print(f"  Total subjects: {len(results)}")
    for class_name in predictor.class_names:
        count = sum(1 for r in results if r.predicted_class == class_name)
        print(f"  {class_name}: {count} ({100*count/len(results):.1f}%)")

    avg_confidence = np.mean([r.confidence for r in results])
    print(f"  Average confidence: {avg_confidence:.3f}")

    return results


def explain_prediction_interactive(
    model_path: Union[str, Path],
    features: np.ndarray,
    subject_id: str = "subject",
    class_names: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> PredictionResult:
    """
    Make prediction with detailed explanation.

    Parameters
    ----------
    model_path : str or Path
        Path to trained model
    features : ndarray of shape (n_features,)
        Feature vector
    subject_id : str, default="subject"
        Subject identifier
    class_names : list of str, optional
        Class names
    output_dir : str or Path, optional
        Output directory

    Returns
    -------
    result : PredictionResult
        Prediction result with explanations
    """
    # Initialize predictor
    predictor = PathologyPredictor(model_path, class_names)

    # Make prediction
    result = predictor.predict_single(features, subject_id, compute_shap=True)

    # Print result
    print("\n" + "=" * 60)
    print(f"Prediction for {subject_id}")
    print("=" * 60)
    print(f"Predicted class: {result.predicted_class}")
    print(f"Confidence: {result.confidence:.3f}")
    print("\nProbability scores:")
    for class_name, prob in result.probability_scores.items():
        print(f"  {class_name}: {prob:.3f}")

    # Generate explanation
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        explanation, fig = predictor.explain_prediction(
            features, subject_id,
            output_path=output_dir / f'{subject_id}_explanation.png'
        )

        print("\nTop contributing features:")
        for i, (feature, value) in enumerate(list(explanation.items())[:10], 1):
            print(f"  {i}. {feature}: {value:.4f}")

        plt.close(fig)

    return result
