#!/usr/bin/env python
"""
Example Usage of NeuroTract Preprocessing Pipeline

Demonstrates how to use the preprocessing modules for diffusion MRI data.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.preprocessing import (
    PreprocessingPipeline,
    MotionCorrector,
    BrainExtractor,
    BiasFieldCorrector,
    GradientCorrector,
)
from backend.data.data_loader import DataLoader
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def example_full_pipeline():
    """
    Example: Run complete preprocessing pipeline

    This is the recommended way to preprocess data - uses the orchestrator
    to coordinate all steps automatically.
    """
    print("=" * 70)
    print("EXAMPLE 1: Full Preprocessing Pipeline")
    print("=" * 70)

    # Initialize pipeline
    pipeline = PreprocessingPipeline(
        output_dir="output/preprocessed",
        skip_motion_correction=False,
        skip_brain_extraction=False,
        skip_bias_correction=False,
        motion_registration="affine",
        brain_method="median_otsu",
        bias_method="auto",
        save_intermediate=True,
        save_qc_reports=True,
    )

    # Run pipeline
    # Replace these paths with your actual data paths
    outputs = pipeline.run(
        dwi_path="data/sub-01/dwi/sub-01_dwi.nii.gz",
        bval_path="data/sub-01/dwi/sub-01_dwi.bval",
        bvec_path="data/sub-01/dwi/sub-01_dwi.bvec",
        output_prefix="sub-01_preprocessed",
    )

    print("\nOutput files:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")


def example_individual_steps():
    """
    Example: Run preprocessing steps individually

    Useful when you need fine-grained control over each step or want to
    customize the workflow.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Individual Preprocessing Steps")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    loader = DataLoader(use_mmap=True, validate=True)
    dwi_data = loader.load_diffusion(
        nifti_path="data/sub-01/dwi/sub-01_dwi.nii.gz",
        auto_detect=True,
    )
    print(f"   Loaded: {dwi_data.shape}, {dwi_data.num_volumes} volumes")

    # Step 1: Gradient correction
    print("\n2. Correcting gradient table...")
    gradient_corrector = GradientCorrector(normalize_bvecs=True)
    corrected_bvals, corrected_bvecs, grad_metrics = gradient_corrector.correct(
        dwi_data.bvals,
        dwi_data.bvecs,
        output_dir=Path("output/qc/gradients"),
    )
    print(f"   {grad_metrics.metrics['num_shells']} shells: {grad_metrics.metrics['shells']}")

    # Step 2: Motion correction
    print("\n3. Correcting motion...")
    motion_corrector = MotionCorrector(
        registration_type="affine",
        metric="MI",
        level_iters=[10000, 1000, 100],
    )
    corrected_data, corrected_bvecs, motion_metrics = motion_corrector.correct(
        dwi_data.data,
        corrected_bvals,
        corrected_bvecs,
        dwi_data.affine,
        mask=None,
        output_dir=Path("output/qc/motion"),
    )
    summary = motion_metrics.compute_summary_stats()
    print(f"   Mean displacement: {summary['mean_translation_mm']:.2f} mm")
    print(f"   Max displacement: {summary['max_translation_mm']:.2f} mm")

    # Step 3: Brain extraction
    print("\n4. Extracting brain...")
    brain_extractor = BrainExtractor(
        method="median_otsu",
        median_radius=4,
        num_pass=4,
        dilate=1,
    )
    masked_data, mask, mask_metrics = brain_extractor.extract(
        corrected_data,
        dwi_data.affine,
        output_dir=Path("output/qc/brain_mask"),
    )
    print(f"   Brain fraction: {mask_metrics.metrics['brain_fraction']:.3f}")
    print(f"   CNR: {mask_metrics.metrics['cnr']:.2f}")

    # Step 4: Bias correction
    print("\n5. Correcting bias field...")
    bias_corrector = BiasFieldCorrector(
        method="auto",
        convergence_threshold=0.001,
        max_iterations=50,
    )
    final_data, bias_field, bias_metrics = bias_corrector.correct(
        masked_data,
        dwi_data.affine,
        mask=mask,
        output_dir=Path("output/qc/bias"),
    )
    print(f"   CV improvement: {bias_metrics.metrics['cv_improvement_percent']:.2f}%")

    print("\n✓ All steps completed successfully!")


def example_gradient_validation_only():
    """
    Example: Validate gradient table without full preprocessing

    Useful for quick QC of acquisition before running full pipeline.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Gradient Table Validation Only")
    print("=" * 70)

    import numpy as np

    # Load just the gradient table
    bvals = np.loadtxt("data/sub-01/dwi/sub-01_dwi.bval")
    bvecs = np.loadtxt("data/sub-01/dwi/sub-01_dwi.bvec")

    # Validate and correct
    gradient_corrector = GradientCorrector(
        normalize_bvecs=True,
        flip_x=False,
        flip_y=False,
        flip_z=False,
    )

    corrected_bvals, corrected_bvecs, metrics = gradient_corrector.correct(
        bvals, bvecs, output_dir=Path("output/gradient_qc")
    )

    # Print detailed metrics
    print("\nGradient Table Metrics:")
    print(f"  Total volumes: {metrics.metrics['num_volumes']}")
    print(f"  b0 volumes: {metrics.metrics['num_b0']}")
    print(f"  DWI volumes: {metrics.metrics['num_dwi']}")
    print(f"  Number of shells: {metrics.metrics['num_shells']}")
    print(f"  Shell b-values: {metrics.metrics['shells']}")
    print(f"  Gradient norm (mean ± std): {metrics.metrics['norm_mean']:.4f} ± {metrics.metrics['norm_std']:.4f}")
    print(f"  Min nearest-neighbor angle: {metrics.metrics['min_angle_deg']:.2f}°")
    print(f"  Mean nearest-neighbor angle: {metrics.metrics['mean_angle_deg']:.2f}°")
    print(f"  Duplicate directions: {metrics.metrics['num_duplicates']}")

    # Validation
    is_valid, message = metrics.is_valid()
    print(f"\nValidation: {'✓ PASS' if is_valid else '✗ FAIL'}")
    print(f"Message: {message}")


def example_brain_extraction_only():
    """
    Example: Brain extraction on pre-motion-corrected data

    Useful when you have already motion-corrected data and just need masking.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Brain Extraction Only")
    print("=" * 70)

    # Load motion-corrected data
    loader = DataLoader(use_mmap=True, validate=True)
    dwi_data = loader.load_diffusion(
        nifti_path="output/preprocessed/intermediate/motion_corrected.nii.gz",
        auto_detect=True,
    )

    # Extract brain with different methods for comparison
    methods = ["median_otsu", "threshold"]

    for method in methods:
        print(f"\nTrying method: {method}")

        extractor = BrainExtractor(
            method=method,
            median_radius=4 if method == "median_otsu" else None,
            num_pass=4 if method == "median_otsu" else None,
            dilate=1,
        )

        masked_data, mask, metrics = extractor.extract(
            dwi_data.data,
            dwi_data.affine,
            output_dir=Path(f"output/brain_mask_{method}"),
        )

        print(f"  Brain fraction: {metrics.metrics['brain_fraction']:.3f}")
        print(f"  CNR: {metrics.metrics['cnr']:.2f}")
        print(f"  Components: {metrics.metrics['num_components']}")

        is_valid, message = metrics.is_valid()
        print(f"  Validation: {'✓ PASS' if is_valid else '✗ FAIL'} - {message}")


def example_custom_parameters():
    """
    Example: Custom pipeline with specific parameters

    Shows how to customize each component's parameters.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Custom Parameters")
    print("=" * 70)

    # Create custom pipeline with specific settings
    pipeline = PreprocessingPipeline(
        output_dir="output/custom_preprocessing",
        skip_motion_correction=False,
        skip_brain_extraction=False,
        skip_bias_correction=False,
        motion_registration="rigid",  # Faster, less flexible
        brain_method="median_otsu",
        bias_method="polynomial",  # Fast fallback
        save_intermediate=True,
        save_qc_reports=True,
    )

    # Customize individual components before running
    # Motion correction: faster settings
    pipeline.motion_corrector.level_iters = [5000, 500, 50]
    pipeline.motion_corrector.sampling_proportion = 0.1  # Use 10% of voxels

    # Brain extraction: more conservative
    pipeline.brain_extractor.median_radius = 3
    pipeline.brain_extractor.num_pass = 3
    pipeline.brain_extractor.dilate = 2  # More generous mask

    # Bias correction: polynomial only (fast)
    pipeline.bias_corrector.method = "polynomial"

    print("Custom settings applied:")
    print(f"  Motion: rigid registration, faster iterations")
    print(f"  Brain: conservative extraction with dilation=2")
    print(f"  Bias: polynomial method only")

    # Run with custom settings
    # outputs = pipeline.run(
    #     dwi_path="data/sub-01/dwi/sub-01_dwi.nii.gz",
    #     output_prefix="sub-01_custom",
    # )


def main():
    """Run examples"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "NeuroTract Preprocessing Examples" + " " * 24 + "║")
    print("╚" + "═" * 68 + "╝")

    # Note: These examples use placeholder paths
    # Replace with actual data paths to run

    print("\nNOTE: These examples use placeholder data paths.")
    print("Replace paths with your actual data to run the examples.\n")

    try:
        # Uncomment the examples you want to run:

        # Full automated pipeline (recommended)
        # example_full_pipeline()

        # Individual steps for fine control
        # example_individual_steps()

        # Quick gradient table validation
        # example_gradient_validation_only()

        # Just brain extraction
        # example_brain_extraction_only()

        # Custom parameters
        example_custom_parameters()

        print("\n" + "=" * 70)
        print("Examples completed!")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("(This is expected if data paths don't exist)")


if __name__ == "__main__":
    main()
