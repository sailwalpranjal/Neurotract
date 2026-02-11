"""
Example Usage of Probabilistic Tractography Pipeline

Demonstrates complete workflow from seed generation to QC.
"""

import numpy as np
import logging
from pathlib import Path

from .rk4_integrator import RK4Integrator
from .seeding import SeedGenerator
from .probabilistic_tracker import ProbabilisticTracker
from .streamline_utils import StreamlineUtils
from .quality_control import TractographyQC

logger = logging.getLogger(__name__)


def example_whole_brain_tractography(
    fa_volume: np.ndarray,
    mask_volume: np.ndarray,
    affine: np.ndarray,
    voxel_size: np.ndarray,
    output_dir: str = "tractography_output"
):
    """
    Complete example: whole-brain probabilistic tractography

    Args:
        fa_volume: Fractional anisotropy map (x, y, z)
        mask_volume: Binary brain mask (x, y, z)
        affine: Affine transformation matrix (4, 4)
        voxel_size: Voxel dimensions in mm (3,)
        output_dir: Output directory for results
    """
    logger.info("Starting whole-brain tractography example")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate seeds
    logger.info("Step 1: Generating seeds")
    seed_generator = SeedGenerator(rng_seed=42)

    seeds, seed_metadata = seed_generator.whole_brain_seeds(
        mask=mask_volume,
        seeds_per_voxel=2,
        voxel_size=voxel_size,
        jitter=True,
        jitter_scale=0.5
    )

    logger.info(f"Generated {len(seeds)} seeds")

    # Save seeds for reproducibility
    seed_generator.save_seeds_to_file(
        seeds,
        str(output_path / "seeds.npy")
    )
    seed_generator.save_seed_log("seed_log.json")

    # Step 2: Define direction getter (example - should be replaced with actual FOD sampling)
    def mock_direction_getter(position):
        """Mock direction getter - replace with actual FOD peak sampling"""
        # In real use, this would sample from FOD peaks
        # For now, return principal eigenvector from DTI
        x, y, z = position.astype(int)

        # Bounds check
        if (x < 0 or y < 0 or z < 0 or
            x >= fa_volume.shape[0] or
            y >= fa_volume.shape[1] or
            z >= fa_volume.shape[2]):
            return np.array([0.0, 0.0, 0.0])

        # Return a direction based on FA (mock - use actual eigenvector in practice)
        fa_val = fa_volume[x, y, z]
        if fa_val > 0.1:
            # Mock direction - in reality, sample from FOD
            return np.array([1.0, 0.0, 0.0])
        else:
            return np.array([0.0, 0.0, 0.0])

    # Step 3: Initialize tracker
    logger.info("Step 2: Initializing probabilistic tracker")
    tracker = ProbabilisticTracker(
        voxel_size=voxel_size,
        step_size=0.5,
        max_angle=60.0,
        fa_threshold=0.1,
        fod_threshold=0.1,
        max_length=200.0,
        min_length=10.0,
        n_samples_per_seed=1,  # Monte Carlo samples per seed
        rng_seed=42,
        memory_limit_gb=10.0
    )

    # Step 4: Perform tracking
    logger.info("Step 3: Performing tractography")

    # For large datasets, stream to HDF5
    streamlines, tracking_metadata = tracker.track(
        seeds=seeds,
        direction_getter=mock_direction_getter,
        fa_volume=fa_volume,
        mask_volume=mask_volume,
        affine=affine,
        output_file=str(output_path / "streamlines.h5"),
        save_rejected=True,
        batch_size=1000
    )

    # Print statistics
    print(tracker.get_statistics_summary())

    # Step 5: Load streamlines (if streamed to disk)
    if isinstance(streamlines, str):
        logger.info("Step 4: Loading streamlines from disk")
        streamlines = ProbabilisticTracker.load_streamlines_from_hdf5(streamlines)

    # Step 6: Filter streamlines
    logger.info("Step 5: Filtering streamlines")

    # Filter by length
    streamlines, _ = StreamlineUtils.filter_by_length(
        streamlines,
        min_length=15.0,
        max_length=150.0
    )

    # Filter by curvature
    streamlines, _ = StreamlineUtils.filter_by_curvature(
        streamlines,
        max_curvature=90.0
    )

    logger.info(f"After filtering: {len(streamlines)} streamlines")

    # Step 7: Smooth streamlines
    logger.info("Step 6: Smoothing streamlines")
    smoothed_streamlines = [
        StreamlineUtils.smooth_streamline(s, sigma=1.0)
        for s in streamlines
    ]

    # Step 8: Cluster streamlines
    logger.info("Step 7: Clustering streamlines")
    labels, centroids = StreamlineUtils.cluster_streamlines(
        streamlines,
        threshold=10.0,
        method='quickbundles'
    )

    logger.info(f"Found {len(centroids)} clusters")

    # Step 9: Quality control
    logger.info("Step 8: Running quality control")
    qc = TractographyQC(output_dir=str(output_path / "qc"))

    # Check length distribution
    length_metrics = qc.check_length_distribution(
        streamlines,
        expected_mean=50.0,
        expected_range=(10.0, 150.0)
    )

    # Check anatomical plausibility
    plausibility_metrics = qc.check_anatomical_plausibility(
        streamlines,
        mask=mask_volume,
        affine=affine,
        fa_volume=fa_volume
    )

    # Check spatial coverage
    coverage_metrics = qc.check_spatial_coverage(
        streamlines,
        mask=mask_volume,
        affine=affine,
        min_coverage=0.3
    )

    # Generate comprehensive report
    qc_report = qc.generate_qc_report(streamlines)

    # Visualize
    qc.visualize_streamlines(
        streamlines,
        n_streamlines=100,
        color_by='length'
    )

    # Step 10: Save outputs
    logger.info("Step 9: Saving outputs")

    # Save in multiple formats
    StreamlineUtils.save_trk(
        streamlines,
        str(output_path / "tractogram.trk"),
        affine=affine,
        voxel_size=voxel_size,
        dimensions=np.array(mask_volume.shape)
    )

    StreamlineUtils.save_tck(
        streamlines,
        str(output_path / "tractogram.tck")
    )

    StreamlineUtils.save_vtk(
        streamlines,
        str(output_path / "tractogram.vtk"),
        scalars={'length': StreamlineUtils.compute_lengths(streamlines)}
    )

    # Compute and save bundle statistics
    stats = StreamlineUtils.compute_bundle_statistics(streamlines)
    logger.info(f"Bundle statistics: {stats}")

    logger.info("Tractography pipeline complete!")

    return {
        'streamlines': streamlines,
        'tracking_metadata': tracking_metadata,
        'qc_report': qc_report,
        'bundle_stats': stats
    }


def example_roi_to_roi_tractography(
    fa_volume: np.ndarray,
    mask_volume: np.ndarray,
    roi1_mask: np.ndarray,
    roi2_mask: np.ndarray,
    affine: np.ndarray,
    voxel_size: np.ndarray,
    output_dir: str = "roi_tractography"
):
    """
    Example: Region-of-interest to ROI tractography

    Args:
        fa_volume: FA map
        mask_volume: Brain mask
        roi1_mask: First ROI mask
        roi2_mask: Second ROI mask
        affine: Affine matrix
        voxel_size: Voxel dimensions
        output_dir: Output directory
    """
    logger.info("Starting ROI-to-ROI tractography")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate seeds in first ROI
    seed_generator = SeedGenerator(rng_seed=42)
    seeds, _ = seed_generator.roi_seeds(
        roi_mask=roi1_mask,
        seeds_per_voxel=5,
        jitter=True,
        boundary_only=False
    )

    # Mock direction getter
    def direction_getter(position):
        x, y, z = position.astype(int)
        if (0 <= x < fa_volume.shape[0] and
            0 <= y < fa_volume.shape[1] and
            0 <= z < fa_volume.shape[2]):
            if fa_volume[x, y, z] > 0.1:
                return np.array([1.0, 0.0, 0.0])
        return np.array([0.0, 0.0, 0.0])

    # Track
    tracker = ProbabilisticTracker(
        voxel_size=voxel_size,
        step_size=0.5,
        max_angle=60.0,
        fa_threshold=0.1,
        n_samples_per_seed=1,
        rng_seed=42
    )

    streamlines, _ = tracker.track(
        seeds=seeds,
        direction_getter=direction_getter,
        fa_volume=fa_volume,
        mask_volume=mask_volume,
        affine=affine
    )

    # Filter streamlines that reach second ROI
    logger.info("Filtering for ROI-to-ROI connections")
    affine_inv = np.linalg.inv(affine)

    roi_connecting_streamlines = []
    for streamline in streamlines:
        # Check if streamline enters ROI2
        n_points = len(streamline)
        points_hom = np.hstack([streamline, np.ones((n_points, 1))])
        points_voxel = (affine_inv @ points_hom.T).T[:, :3]

        reaches_roi2 = False
        for point in points_voxel:
            x, y, z = point.astype(int)
            if (0 <= x < roi2_mask.shape[0] and
                0 <= y < roi2_mask.shape[1] and
                0 <= z < roi2_mask.shape[2]):
                if roi2_mask[x, y, z] > 0:
                    reaches_roi2 = True
                    break

        if reaches_roi2:
            roi_connecting_streamlines.append(streamline)

    logger.info(
        f"Found {len(roi_connecting_streamlines)} streamlines "
        f"connecting ROIs ({len(roi_connecting_streamlines)/len(streamlines)*100:.1f}%)"
    )

    # Save results
    StreamlineUtils.save_trk(
        roi_connecting_streamlines,
        str(output_path / "roi_to_roi.trk"),
        affine=affine,
        voxel_size=voxel_size,
        dimensions=np.array(mask_volume.shape)
    )

    return roi_connecting_streamlines


def example_reproducibility_test(
    fa_volume: np.ndarray,
    mask_volume: np.ndarray,
    affine: np.ndarray,
    voxel_size: np.ndarray,
    output_dir: str = "reproducibility_test"
):
    """
    Example: Test reproducibility with multiple runs

    Args:
        fa_volume: FA map
        mask_volume: Brain mask
        affine: Affine matrix
        voxel_size: Voxel dimensions
        output_dir: Output directory
    """
    logger.info("Testing reproducibility with multiple runs")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Same seeds for all runs
    seed_generator = SeedGenerator(rng_seed=42)
    seeds, _ = seed_generator.whole_brain_seeds(
        mask=mask_volume,
        seeds_per_voxel=1,
        jitter=False  # No jitter for reproducibility test
    )

    # Subsample for testing
    seeds = seeds[:1000]

    def direction_getter(position):
        x, y, z = position.astype(int)
        if (0 <= x < fa_volume.shape[0] and
            0 <= y < fa_volume.shape[1] and
            0 <= z < fa_volume.shape[2]):
            if fa_volume[x, y, z] > 0.1:
                return np.array([1.0, 0.0, 0.0])
        return np.array([0.0, 0.0, 0.0])

    # Run tractography multiple times with same RNG seed
    bundles = []
    for run_idx in range(3):
        logger.info(f"Run {run_idx + 1}/3")

        tracker = ProbabilisticTracker(
            voxel_size=voxel_size,
            step_size=0.5,
            max_angle=60.0,
            fa_threshold=0.1,
            n_samples_per_seed=1,
            rng_seed=42  # Same seed = same results
        )

        streamlines, _ = tracker.track(
            seeds=seeds,
            direction_getter=direction_getter,
            fa_volume=fa_volume,
            mask_volume=mask_volume,
            affine=affine
        )

        bundles.append(streamlines)

    # Quality control - check consistency
    qc = TractographyQC(output_dir=str(output_path / "qc"))

    consistency_metrics = qc.check_bundle_consistency(
        bundles,
        bundle_names=[f"Run_{i+1}" for i in range(3)]
    )

    reproducibility_metrics = qc.check_reproducibility(
        bundles[0],
        bundles[1],
        distance_threshold=5.0
    )

    logger.info("Reproducibility test complete")
    logger.info(f"Consistency: {consistency_metrics}")
    logger.info(f"Reproducibility: {reproducibility_metrics}")

    return {
        'bundles': bundles,
        'consistency': consistency_metrics,
        'reproducibility': reproducibility_metrics
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create mock data for testing
    shape = (50, 50, 50)
    fa_volume = np.random.rand(*shape) * 0.8
    mask_volume = np.ones(shape, dtype=np.uint8)
    affine = np.eye(4)
    affine[:3, :3] = np.diag([2.0, 2.0, 2.0])
    voxel_size = np.array([2.0, 2.0, 2.0])

    # Run example
    results = example_whole_brain_tractography(
        fa_volume,
        mask_volume,
        affine,
        voxel_size,
        output_dir="example_output"
    )

    print("\nTractography complete!")
    print(f"Generated {len(results['streamlines'])} streamlines")
