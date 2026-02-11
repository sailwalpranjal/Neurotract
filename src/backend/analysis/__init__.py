"""
Analysis Module
===============

Machine learning models and statistical analysis for pathology detection
from neuroimaging connectome data.

Modules
-------
pathology_detection : ML classifiers with interpretability
statistical_analysis : Classical statistics and hypothesis testing
harmonization : ComBat harmonization for multi-site data
training : Model training pipeline
prediction : Inference pipeline for new subjects

"""

__version__ = "0.1.0"

# Import main classes and functions
from .pathology_detection import (
    PathologyClassifier,
    ConfoundRegressor,
    create_classification_report
)

from .statistical_analysis import (
    permutation_test,
    fdr_correction,
    cohens_d,
    hedges_g,
    glass_delta,
    compute_effect_size,
    bootstrap_ci,
    network_based_statistic,
    mass_univariate_test,
    correlation_with_permutation,
    PermutationTestResult,
    EffectSizeResult,
    NBSResult
)

from .harmonization import (
    ComBatHarmonizer,
    harmonize_connectomes
)

from .training import (
    ConnectomeDataLoader,
    ModelTrainer,
    train_pathology_classifier
)

from .prediction import (
    PathologyPredictor,
    PredictionResult,
    predict_from_model,
    explain_prediction_interactive
)

__all__ = [
    # Pathology detection
    'PathologyClassifier',
    'ConfoundRegressor',
    'create_classification_report',

    # Statistical analysis
    'permutation_test',
    'fdr_correction',
    'cohens_d',
    'hedges_g',
    'glass_delta',
    'compute_effect_size',
    'bootstrap_ci',
    'network_based_statistic',
    'mass_univariate_test',
    'correlation_with_permutation',
    'PermutationTestResult',
    'EffectSizeResult',
    'NBSResult',

    # Harmonization
    'ComBatHarmonizer',
    'harmonize_connectomes',

    # Training
    'ConnectomeDataLoader',
    'ModelTrainer',
    'train_pathology_classifier',

    # Prediction
    'PathologyPredictor',
    'PredictionResult',
    'predict_from_model',
    'explain_prediction_interactive',
]
