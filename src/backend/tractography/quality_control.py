"""
Quality Control for Tractography

Comprehensive QC tools for tractography results including:
- Bundle consistency checks across multiple runs
- Reproducibility metrics
- Anatomical plausibility validation
- Visualization and reporting
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .streamline_utils import StreamlineUtils

logger = logging.getLogger(__name__)


class TractographyQC:
    """
    Quality control for tractography results
    """

    def __init__(self, output_dir: str = "analysis_and_decisions/tractography_qc"):
        """
        Initialize QC module

        Args:
            output_dir: Directory for QC outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.qc_results = {
            'timestamp': datetime.now().isoformat(),
            'checks': []
        }

        logger.info(f"TractographyQC initialized: output_dir={output_dir}")

    def check_bundle_consistency(
        self,
        bundles: List[List[np.ndarray]],
        bundle_names: Optional[List[str]] = None,
        overlap_threshold: float = 0.5
    ) -> Dict:
        """
        Check consistency across multiple tractography runs

        Args:
            bundles: List of bundles (each bundle is list of streamlines)
            bundle_names: Optional names for each bundle
            overlap_threshold: Minimum overlap for consistency

        Returns:
            Consistency metrics
        """
        logger.info(f"Checking consistency across {len(bundles)} runs")

        if bundle_names is None:
            bundle_names = [f"Run_{i+1}" for i in range(len(bundles))]

        # Compute statistics for each bundle
        bundle_stats = []
        for bundle, name in zip(bundles, bundle_names):
            stats = StreamlineUtils.compute_bundle_statistics(bundle)
            stats['name'] = name
            bundle_stats.append(stats)

        # Check length consistency
        lengths = [s['mean_length'] for s in bundle_stats]
        length_cv = np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0

        # Check count consistency
        counts = [s['n_streamlines'] for s in bundle_stats]
        count_cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0

        consistency_metrics = {
            'n_runs': len(bundles),
            'bundle_statistics': bundle_stats,
            'length_mean': np.mean(lengths),
            'length_std': np.std(lengths),
            'length_cv': length_cv,
            'count_mean': np.mean(counts),
            'count_std': np.std(counts),
            'count_cv': count_cv,
            'consistent': length_cv < 0.2 and count_cv < 0.3  # Heuristic thresholds
        }

        # Generate report
        report_path = self.output_dir / "bundle_consistency_report.txt"
        self._write_consistency_report(consistency_metrics, report_path)

        self.qc_results['checks'].append({
            'check': 'bundle_consistency',
            'passed': consistency_metrics['consistent'],
            'metrics': consistency_metrics
        })

        logger.info(
            f"Bundle consistency: length_CV={length_cv:.3f}, "
            f"count_CV={count_cv:.3f}, "
            f"consistent={'PASS' if consistency_metrics['consistent'] else 'FAIL'}"
        )

        return consistency_metrics

    def check_reproducibility(
        self,
        streamlines_1: List[np.ndarray],
        streamlines_2: List[np.ndarray],
        distance_threshold: float = 5.0
    ) -> Dict:
        """
        Check reproducibility between two tractography runs

        Computes overlap and spatial agreement metrics

        Args:
            streamlines_1: First set of streamlines
            streamlines_2: Second set of streamlines
            distance_threshold: Distance threshold for overlap (mm)

        Returns:
            Reproducibility metrics
        """
        logger.info(
            f"Checking reproducibility: {len(streamlines_1)} vs "
            f"{len(streamlines_2)} streamlines"
        )

        # Compute bundle statistics
        stats_1 = StreamlineUtils.compute_bundle_statistics(streamlines_1)
        stats_2 = StreamlineUtils.compute_bundle_statistics(streamlines_2)

        # Count overlap (streamlines within distance threshold)
        overlap_count = 0
        sample_size = min(len(streamlines_1), 1000)  # Subsample for efficiency

        # Resample streamlines for distance computation
        resampled_1 = [
            StreamlineUtils.resample_streamline(s, n_points=20)
            for s in np.random.choice(streamlines_1, sample_size, replace=False)
        ]
        resampled_2 = [
            StreamlineUtils.resample_streamline(s, n_points=20)
            for s in streamlines_2
        ]

        for s1 in resampled_1:
            min_dist = float('inf')
            for s2 in resampled_2:
                dist = StreamlineUtils._mdf_distance(s1, s2)
                if dist < min_dist:
                    min_dist = dist

            if min_dist < distance_threshold:
                overlap_count += 1

        overlap_fraction = overlap_count / sample_size if sample_size > 0 else 0

        # Dice coefficient for streamline counts
        dice = 2 * min(stats_1['n_streamlines'], stats_2['n_streamlines']) / \
               (stats_1['n_streamlines'] + stats_2['n_streamlines'])

        # Length agreement
        length_diff = abs(stats_1['mean_length'] - stats_2['mean_length'])
        length_agreement = length_diff / max(stats_1['mean_length'], stats_2['mean_length'])

        reproducibility_metrics = {
            'overlap_fraction': overlap_fraction,
            'dice_coefficient': dice,
            'length_difference': length_diff,
            'length_agreement': length_agreement,
            'count_1': stats_1['n_streamlines'],
            'count_2': stats_2['n_streamlines'],
            'mean_length_1': stats_1['mean_length'],
            'mean_length_2': stats_2['mean_length'],
            'reproducible': overlap_fraction > 0.5 and dice > 0.7
        }

        self.qc_results['checks'].append({
            'check': 'reproducibility',
            'passed': reproducibility_metrics['reproducible'],
            'metrics': reproducibility_metrics
        })

        logger.info(
            f"Reproducibility: overlap={overlap_fraction:.3f}, "
            f"dice={dice:.3f}, "
            f"reproducible={'PASS' if reproducibility_metrics['reproducible'] else 'FAIL'}"
        )

        return reproducibility_metrics

    def check_anatomical_plausibility(
        self,
        streamlines: List[np.ndarray],
        mask: np.ndarray,
        affine: np.ndarray,
        fa_volume: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Check anatomical plausibility of streamlines

        Verifies streamlines stay within brain mask and white matter regions

        Args:
            streamlines: List of streamlines in world coordinates
            mask: Brain mask in voxel space
            affine: World to voxel transformation
            fa_volume: Optional FA volume for WM checking

        Returns:
            Plausibility metrics
        """
        logger.info(f"Checking anatomical plausibility for {len(streamlines)} streamlines")

        # Transform to voxel coordinates
        affine_inv = np.linalg.inv(affine)

        violations = {
            'outside_mask': 0,
            'low_fa': 0,
            'total_streamlines': len(streamlines)
        }

        for streamline in streamlines:
            # Transform to voxel coordinates
            n_points = len(streamline)
            points_hom = np.hstack([streamline, np.ones((n_points, 1))])
            points_voxel = (affine_inv @ points_hom.T).T[:, :3]

            # Check mask violations
            for point in points_voxel:
                x, y, z = point.astype(int)

                # Bounds check
                if (x < 0 or y < 0 or z < 0 or
                    x >= mask.shape[0] or y >= mask.shape[1] or z >= mask.shape[2]):
                    violations['outside_mask'] += 1
                    break

                # Mask check
                if mask[x, y, z] == 0:
                    violations['outside_mask'] += 1
                    break

                # FA check (white matter)
                if fa_volume is not None:
                    if fa_volume[x, y, z] < 0.2:  # Typical WM threshold
                        violations['low_fa'] += 1
                        break

        # Calculate metrics
        mask_violation_rate = violations['outside_mask'] / len(streamlines)
        fa_violation_rate = violations['low_fa'] / len(streamlines) if fa_volume is not None else 0

        plausibility_metrics = {
            'n_streamlines': len(streamlines),
            'mask_violations': violations['outside_mask'],
            'mask_violation_rate': mask_violation_rate,
            'fa_violations': violations['low_fa'],
            'fa_violation_rate': fa_violation_rate,
            'plausible': mask_violation_rate < 0.1
        }

        self.qc_results['checks'].append({
            'check': 'anatomical_plausibility',
            'passed': plausibility_metrics['plausible'],
            'metrics': plausibility_metrics
        })

        logger.info(
            f"Anatomical plausibility: mask_violations={mask_violation_rate:.3f}, "
            f"plausible={'PASS' if plausibility_metrics['plausible'] else 'FAIL'}"
        )

        return plausibility_metrics

    def check_length_distribution(
        self,
        streamlines: List[np.ndarray],
        expected_mean: float = 50.0,
        expected_range: Tuple[float, float] = (10.0, 150.0)
    ) -> Dict:
        """
        Check if streamline length distribution is reasonable

        Args:
            streamlines: List of streamlines
            expected_mean: Expected mean length (mm)
            expected_range: Expected range (min, max) in mm

        Returns:
            Length distribution metrics
        """
        logger.info(f"Checking length distribution for {len(streamlines)} streamlines")

        lengths = StreamlineUtils.compute_lengths(streamlines)

        metrics = {
            'n_streamlines': len(streamlines),
            'mean_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'expected_mean': expected_mean,
            'expected_range': expected_range
        }

        # Check if distribution is reasonable
        mean_in_range = expected_range[0] < metrics['mean_length'] < expected_range[1]
        spread_reasonable = metrics['std_length'] < metrics['mean_length']  # CV < 1

        metrics['distribution_ok'] = mean_in_range and spread_reasonable

        # Create histogram
        self._plot_length_distribution(lengths, expected_mean, expected_range)

        self.qc_results['checks'].append({
            'check': 'length_distribution',
            'passed': metrics['distribution_ok'],
            'metrics': metrics
        })

        logger.info(
            f"Length distribution: mean={metrics['mean_length']:.1f}mm, "
            f"range=[{metrics['min_length']:.1f}, {metrics['max_length']:.1f}], "
            f"ok={'PASS' if metrics['distribution_ok'] else 'FAIL'}"
        )

        return metrics

    def check_spatial_coverage(
        self,
        streamlines: List[np.ndarray],
        mask: np.ndarray,
        affine: np.ndarray,
        min_coverage: float = 0.3
    ) -> Dict:
        """
        Check spatial coverage of tractography

        Args:
            streamlines: List of streamlines
            mask: Brain mask
            affine: Affine transformation
            min_coverage: Minimum fraction of mask to cover

        Returns:
            Coverage metrics
        """
        logger.info(f"Checking spatial coverage for {len(streamlines)} streamlines")

        # Create coverage map
        coverage = np.zeros_like(mask, dtype=np.uint32)
        affine_inv = np.linalg.inv(affine)

        for streamline in streamlines:
            # Transform to voxel coordinates
            n_points = len(streamline)
            points_hom = np.hstack([streamline, np.ones((n_points, 1))])
            points_voxel = (affine_inv @ points_hom.T).T[:, :3]

            # Mark visited voxels
            for point in points_voxel:
                x, y, z = point.astype(int)

                if (0 <= x < mask.shape[0] and
                    0 <= y < mask.shape[1] and
                    0 <= z < mask.shape[2]):
                    coverage[x, y, z] += 1

        # Calculate coverage metrics
        mask_voxels = np.sum(mask > 0)
        covered_voxels = np.sum((coverage > 0) & (mask > 0))
        coverage_fraction = covered_voxels / mask_voxels if mask_voxels > 0 else 0

        # Visit density
        mean_visits = np.mean(coverage[mask > 0])
        max_visits = np.max(coverage[mask > 0])

        metrics = {
            'n_streamlines': len(streamlines),
            'mask_voxels': int(mask_voxels),
            'covered_voxels': int(covered_voxels),
            'coverage_fraction': coverage_fraction,
            'mean_visits_per_voxel': float(mean_visits),
            'max_visits_per_voxel': int(max_visits),
            'adequate_coverage': coverage_fraction >= min_coverage
        }

        self.qc_results['checks'].append({
            'check': 'spatial_coverage',
            'passed': metrics['adequate_coverage'],
            'metrics': metrics
        })

        logger.info(
            f"Spatial coverage: {coverage_fraction:.3f} of mask covered, "
            f"adequate={'PASS' if metrics['adequate_coverage'] else 'FAIL'}"
        )

        return metrics

    def generate_qc_report(
        self,
        streamlines: List[np.ndarray],
        output_filename: str = "tractography_qc_report.json"
    ) -> Dict:
        """
        Generate comprehensive QC report

        Args:
            streamlines: Streamlines to assess
            output_filename: Output file name

        Returns:
            Complete QC report
        """
        logger.info("Generating comprehensive QC report")

        # Compute overall statistics
        stats = StreamlineUtils.compute_bundle_statistics(streamlines)

        # Compile report
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_streamlines': len(streamlines),
            'statistics': stats,
            'qc_checks': self.qc_results['checks'],
            'overall_pass': all(
                check['passed'] for check in self.qc_results['checks']
            )
        }

        # Save report
        report_path = self.output_dir / output_filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"QC report saved to: {report_path}")

        # Generate summary
        self._generate_summary_report(report)

        return report

    def _write_consistency_report(self, metrics: Dict, filepath: Path):
        """Write consistency report to text file"""
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("BUNDLE CONSISTENCY REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Number of runs: {metrics['n_runs']}\n")
            f.write(f"Overall consistency: {'PASS' if metrics['consistent'] else 'FAIL'}\n\n")

            f.write("Length Statistics:\n")
            f.write(f"  Mean: {metrics['length_mean']:.2f} mm\n")
            f.write(f"  Std:  {metrics['length_std']:.2f} mm\n")
            f.write(f"  CV:   {metrics['length_cv']:.4f}\n\n")

            f.write("Count Statistics:\n")
            f.write(f"  Mean: {metrics['count_mean']:.0f} streamlines\n")
            f.write(f"  Std:  {metrics['count_std']:.0f} streamlines\n")
            f.write(f"  CV:   {metrics['count_cv']:.4f}\n\n")

            f.write("Per-Bundle Statistics:\n")
            for stats in metrics['bundle_statistics']:
                f.write(f"\n  {stats['name']}:\n")
                f.write(f"    Count: {stats['n_streamlines']}\n")
                f.write(f"    Mean length: {stats['mean_length']:.2f} mm\n")
                f.write(f"    Std length:  {stats['std_length']:.2f} mm\n")

            f.write("\n" + "=" * 60 + "\n")

        logger.info(f"Consistency report written to: {filepath}")

    def _plot_length_distribution(
        self,
        lengths: np.ndarray,
        expected_mean: float,
        expected_range: Tuple[float, float]
    ):
        """Plot streamline length distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        ax.hist(lengths, bins=50, alpha=0.7, edgecolor='black')

        # Expected values
        ax.axvline(expected_mean, color='r', linestyle='--', label=f'Expected mean: {expected_mean}mm')
        ax.axvline(expected_range[0], color='g', linestyle='--', label=f'Min: {expected_range[0]}mm')
        ax.axvline(expected_range[1], color='g', linestyle='--', label=f'Max: {expected_range[1]}mm')

        # Actual statistics
        ax.axvline(np.mean(lengths), color='b', linestyle='-', linewidth=2, label=f'Actual mean: {np.mean(lengths):.1f}mm')

        ax.set_xlabel('Streamline Length (mm)')
        ax.set_ylabel('Count')
        ax.set_title('Streamline Length Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save figure
        output_path = self.output_dir / "length_distribution.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Length distribution plot saved to: {output_path}")

    def _generate_summary_report(self, report: Dict):
        """Generate human-readable summary report"""
        summary_path = self.output_dir / "qc_summary.txt"

        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TRACTOGRAPHY QUALITY CONTROL SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"Number of streamlines: {report['n_streamlines']}\n")
            f.write(f"Overall QC: {'PASS' if report['overall_pass'] else 'FAIL'}\n\n")

            f.write("Bundle Statistics:\n")
            f.write(f"  Mean length:   {report['statistics']['mean_length']:.2f} mm\n")
            f.write(f"  Median length: {report['statistics']['median_length']:.2f} mm\n")
            f.write(f"  Std length:    {report['statistics']['std_length']:.2f} mm\n")
            f.write(f"  Min length:    {report['statistics']['min_length']:.2f} mm\n")
            f.write(f"  Max length:    {report['statistics']['max_length']:.2f} mm\n")
            f.write(f"  Total length:  {report['statistics']['total_length']:.2f} mm\n\n")

            f.write("QC Checks:\n")
            for check in report['qc_checks']:
                status = "PASS" if check['passed'] else "FAIL"
                f.write(f"  [{status}] {check['check']}\n")

            f.write("\n" + "=" * 70 + "\n")

        logger.info(f"Summary report saved to: {summary_path}")

    def visualize_streamlines(
        self,
        streamlines: List[np.ndarray],
        output_filename: str = "streamlines_3d.png",
        n_streamlines: int = 100,
        color_by: str = 'length'
    ):
        """
        Create 3D visualization of streamlines

        Args:
            streamlines: List of streamlines
            output_filename: Output file name
            n_streamlines: Number of streamlines to visualize
            color_by: Color scheme ('length', 'orientation', 'uniform')
        """
        from mpl_toolkits.mplot3d import Axes3D

        logger.info(f"Visualizing {min(n_streamlines, len(streamlines))} streamlines")

        # Subsample streamlines
        if len(streamlines) > n_streamlines:
            indices = np.random.choice(len(streamlines), n_streamlines, replace=False)
            vis_streamlines = [streamlines[i] for i in indices]
        else:
            vis_streamlines = streamlines

        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Color mapping
        if color_by == 'length':
            lengths = StreamlineUtils.compute_lengths(vis_streamlines)
            colors = plt.cm.viridis((lengths - lengths.min()) / (lengths.max() - lengths.min()))
        else:
            colors = ['blue'] * len(vis_streamlines)

        # Plot streamlines
        for streamline, color in zip(vis_streamlines, colors):
            ax.plot(
                streamline[:, 0],
                streamline[:, 1],
                streamline[:, 2],
                color=color,
                alpha=0.5,
                linewidth=0.5
            )

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Tractography Visualization ({len(vis_streamlines)} streamlines)')

        # Save
        output_path = self.output_dir / output_filename
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Visualization saved to: {output_path}")
