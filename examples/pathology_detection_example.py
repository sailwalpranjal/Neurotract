"""
Pathology Detection Example
===========================

Comprehensive example demonstrating the complete ML pipeline for pathology
detection from connectome data.

This example shows:
1. Loading connectome data
2. Data harmonization (ComBat)
3. Statistical analysis
4. Model training with hyperparameter tuning
5. Model evaluation
6. Making predictions on new subjects
"""

import numpy as np
import pandas as pd
from pathlib import Path

# NeuroTract analysis modules
from src.backend.analysis import (
    # Training pipeline
    train_pathology_classifier,
    ConnectomeDataLoader,
    ModelTrainer,

    # Prediction
    PathologyPredictor,
    predict_from_model,
    explain_prediction_interactive,

    # Statistical analysis
    permutation_test,
    fdr_correction,
    compute_effect_size,
    network_based_statistic,
    mass_univariate_test,

    # Harmonization
    harmonize_connectomes,
    ComBatHarmonizer,

    # Pathology detection
    PathologyClassifier,
    ConfoundRegressor
)


def example_1_training_pipeline():
    """Example 1: Complete training pipeline from data to trained model."""

    print("\n" + "=" * 70)
    print("EXAMPLE 1: Complete Training Pipeline")
    print("=" * 70)

    # Paths
    data_path = Path("data/processed/connectomes.h5")
    output_dir = Path("models/pathology_detection")

    # Check if data exists
    if not data_path.exists():
        print(f"\nData file not found: {data_path}")
        print("This is a demonstration - generate synthetic data instead...")
        generate_synthetic_data(data_path)

    # Train model using the complete pipeline
    results = train_pathology_classifier(
        data_path=data_path,
        output_dir=output_dir,
        feature_type='edges',
        label_column='diagnosis',
        model_type='logistic',
        harmonize=True,
        batch_column='scanner',
        covariate_columns=['age', 'sex'],
        test_size=0.2,
        val_size=0.2,
        tune_hyperparams=True,
        param_grid={
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1.0, 10.0]
        },
        random_state=42
    )

    print("\nTraining completed!")
    print(f"Model saved to: {results['model_path']}")
    print(f"\nTest set performance:")
    print(f"  ROC-AUC: {results['report']['metrics']['test']['roc_auc']:.3f}")
    print(f"  Accuracy: {results['report']['metrics']['test']['accuracy']:.3f}")

    return results


def example_2_manual_training():
    """Example 2: Manual training with full control over each step."""

    print("\n" + "=" * 70)
    print("EXAMPLE 2: Manual Training with Full Control")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, y, metadata = generate_synthetic_connectome_data(n_subjects=200, n_nodes=84)

    # Split into control and patient groups
    controls = X[y == 0]
    patients = X[y == 1]

    # 2. Statistical Analysis
    print("\n2. Performing statistical analysis...")

    # Permutation test
    perm_result = permutation_test(
        controls, patients,
        n_permutations=5000,
        random_state=42
    )
    print(f"   Permutation test p-value: {perm_result.pvalue:.4f}")

    # Effect size
    effect_size = compute_effect_size(controls, patients, n_bootstrap=5000)
    print(f"   Cohen's d: {effect_size.cohens_d:.3f}")
    print(f"   95% CI: [{effect_size.ci_lower:.3f}, {effect_size.ci_upper:.3f}]")

    # Mass univariate testing
    mut_results = mass_univariate_test(
        controls, patients,
        alpha=0.05,
        correction='fdr'
    )
    print(f"   Significant features (FDR corrected): {mut_results['n_significant']}")

    # 3. Harmonization
    print("\n3. Applying ComBat harmonization...")

    # Create batch labels (simulate scanner effects)
    batches = metadata['scanner'].values

    # Preserve biological covariates
    covariates = metadata[['age', 'sex']]

    # Harmonize
    harmonizer = ComBatHarmonizer(parametric=True)
    X_harmonized = harmonizer.fit_transform(X.T, batches, covariates).T

    print("   Harmonization complete!")

    # 4. Confound Regression
    print("\n4. Removing confound effects...")

    confound_regressor = ConfoundRegressor(confound_names=['age', 'sex', 'scanner'])
    X_deconfounded = confound_regressor.fit_transform(X_harmonized, metadata)

    print("   Confounds removed!")

    # 5. Train Classifier
    print("\n5. Training classifier...")

    # Initialize trainer
    trainer = ModelTrainer(output_dir="models/manual_training", random_state=42)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        X_deconfounded, y, test_size=0.2, val_size=0.2
    )

    # Hyperparameter tuning
    tuning_results = trainer.tune_hyperparameters(
        X_train, y_train,
        model_type='logistic',
        param_grid={'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10.0]}
    )

    # Train final model
    model = trainer.train_model(X_train, y_train, model_type='logistic')

    # Evaluate
    test_metrics = trainer.evaluate_model(model, X_test, y_test)

    # Generate report
    report = trainer.generate_report(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        class_names=['Control', 'Patient']
    )

    # Save model
    model_path = trainer.save_model(model)

    print("\n6. Model training complete!")
    print(f"   Test ROC-AUC: {test_metrics['roc_auc']:.3f}")

    return model, model_path


def example_3_prediction():
    """Example 3: Making predictions on new subjects."""

    print("\n" + "=" * 70)
    print("EXAMPLE 3: Making Predictions on New Subjects")
    print("=" * 70)

    # First, train a model (or load existing)
    print("\n1. Training model...")
    model, model_path = example_2_manual_training()

    # Generate new test subjects
    print("\n2. Generating new test subjects...")
    X_new, y_new, metadata_new = generate_synthetic_connectome_data(
        n_subjects=20, n_nodes=84
    )
    subject_ids = [f"NEW_{i:03d}" for i in range(len(X_new))]

    # 3. Make predictions
    print("\n3. Making predictions...")

    predictor = PathologyPredictor(
        model_path=model_path,
        class_names=['Control', 'Patient']
    )

    # Batch prediction
    results = predictor.predict_batch(
        X_new, subject_ids=subject_ids, compute_shap=True
    )

    # Print results
    print("\nPrediction results:")
    for result in results[:5]:  # Show first 5
        print(f"\n  {result.subject_id}:")
        print(f"    Predicted: {result.predicted_class}")
        print(f"    Confidence: {result.confidence:.3f}")
        print(f"    Probabilities: {result.probability_scores}")

    # 4. Generate prediction report
    print("\n4. Generating prediction report...")
    report = predictor.generate_prediction_report(
        results, output_dir="predictions/new_subjects"
    )

    # 5. Explain individual prediction
    print("\n5. Explaining individual prediction...")
    explanation_result = explain_prediction_interactive(
        model_path=model_path,
        features=X_new[0],
        subject_id=subject_ids[0],
        class_names=['Control', 'Patient'],
        output_dir="predictions/explanations"
    )

    return results


def example_4_network_based_statistic():
    """Example 4: Network-Based Statistic for group comparison."""

    print("\n" + "=" * 70)
    print("EXAMPLE 4: Network-Based Statistic (NBS)")
    print("=" * 70)

    # Generate connectivity matrices for two groups
    print("\n1. Generating connectivity matrices...")

    n_subjects1 = 30
    n_subjects2 = 30
    n_nodes = 84

    # Group 1 (controls)
    group1_matrices = np.random.randn(n_subjects1, n_nodes, n_nodes) * 0.3
    group1_matrices = (group1_matrices + group1_matrices.transpose(0, 2, 1)) / 2

    # Group 2 (patients) - add effect to some edges
    group2_matrices = np.random.randn(n_subjects2, n_nodes, n_nodes) * 0.3
    # Add group difference to specific region
    group2_matrices[:, 10:20, 10:20] += 0.5
    group2_matrices = (group2_matrices + group2_matrices.transpose(0, 2, 1)) / 2

    # 2. Perform NBS
    print("\n2. Performing Network-Based Statistic...")

    nbs_result = network_based_statistic(
        group1_matrices,
        group2_matrices,
        threshold=2.5,
        n_permutations=1000,
        test='ttest',
        random_state=42
    )

    print(f"\n3. NBS Results:")
    print(f"   P-value: {nbs_result.pvalue:.4f}")
    print(f"   Number of significant components: {len(nbs_result.significant_components)}")

    if nbs_result.significant_components:
        print(f"   Largest component size: {max(nbs_result.component_sizes)} nodes")

    return nbs_result


def example_5_cross_validation():
    """Example 5: Cross-validation with multiple models."""

    print("\n" + "=" * 70)
    print("EXAMPLE 5: Model Comparison with Cross-Validation")
    print("=" * 70)

    # Generate data
    X, y, metadata = generate_synthetic_connectome_data(n_subjects=150, n_nodes=84)

    # Models to compare
    models = {
        'Logistic (L1)': PathologyClassifier(model_type='logistic', penalty='l1'),
        'Logistic (L2)': PathologyClassifier(model_type='logistic', penalty='l2'),
        'Random Forest': PathologyClassifier(model_type='random_forest', n_estimators=100)
    }

    print("\nComparing models with 5-fold cross-validation...\n")

    results_summary = []

    for name, model in models.items():
        print(f"Evaluating {name}...")

        # Cross-validation
        cv_results = model.cross_validate(X, y, cv=5)

        results_summary.append({
            'Model': name,
            'Mean ROC-AUC': cv_results['mean_test_roc_auc'],
            'Std ROC-AUC': cv_results['std_test_roc_auc'],
            'Mean Accuracy': cv_results['mean_test_accuracy'],
            'Std Accuracy': cv_results['std_test_accuracy']
        })

    # Display results
    results_df = pd.DataFrame(results_summary)
    print("\n" + "=" * 70)
    print("Cross-Validation Results:")
    print("=" * 70)
    print(results_df.to_string(index=False))

    return results_df


# ============================================================================
# Utility functions for generating synthetic data
# ============================================================================

def generate_synthetic_connectome_data(
    n_subjects: int = 100,
    n_nodes: int = 84,
    effect_size: float = 0.5
) -> tuple:
    """Generate synthetic connectome data for demonstration."""

    # Generate edge features (upper triangle of connectivity matrix)
    n_edges = n_nodes * (n_nodes - 1) // 2

    # Generate two groups
    n_controls = n_subjects // 2
    n_patients = n_subjects - n_controls

    # Control group
    X_controls = np.random.randn(n_controls, n_edges) * 0.3

    # Patient group (with added effect)
    X_patients = np.random.randn(n_patients, n_edges) * 0.3
    # Add effect to subset of edges
    X_patients[:, :n_edges//4] += effect_size

    # Combine
    X = np.vstack([X_controls, X_patients])
    y = np.array([0] * n_controls + [1] * n_patients)

    # Generate metadata
    metadata = pd.DataFrame({
        'subject_id': [f'SUB_{i:03d}' for i in range(n_subjects)],
        'diagnosis': y,
        'age': np.random.normal(45, 15, n_subjects),
        'sex': np.random.choice(['M', 'F'], n_subjects),
        'scanner': np.random.choice(['Scanner_A', 'Scanner_B', 'Scanner_C'], n_subjects)
    })

    return X, y, metadata


def generate_synthetic_data(output_path: Path):
    """Generate and save synthetic HDF5 data."""

    import h5py

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate data
    X, y, metadata = generate_synthetic_connectome_data(n_subjects=200, n_nodes=84)

    # Save to HDF5
    with h5py.File(output_path, 'w') as f:
        # Save edge features
        f.create_dataset('edge_features', data=X)

        # Save metadata
        metadata_group = f.create_group('metadata')
        for col in metadata.columns:
            if metadata[col].dtype == 'object':
                # String data
                metadata_group.create_dataset(
                    col, data=metadata[col].astype('S')
                )
            else:
                metadata_group.create_dataset(col, data=metadata[col].values)

    print(f"Synthetic data saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples."""

    print("\n" + "=" * 70)
    print("NeuroTract Pathology Detection Examples")
    print("=" * 70)

    # Run examples
    try:
        # Example 1: Complete training pipeline
        # Uncomment if you have real data
        # example_1_training_pipeline()

        # Example 2: Manual training
        example_2_manual_training()

        # Example 3: Predictions
        example_3_prediction()

        # Example 4: Network-based statistic
        example_4_network_based_statistic()

        # Example 5: Model comparison
        example_5_cross_validation()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
