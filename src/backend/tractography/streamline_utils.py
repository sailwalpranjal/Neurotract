"""
Streamline Manipulation and Analysis Utilities

Provides comprehensive tools for:
- Filtering streamlines by length, curvature, anatomical criteria
- Smoothing and resampling
- Format conversion (TRK, TCK, VTK)
- Bundle statistics and analysis
- Clustering and bundle extraction
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


class StreamlineUtils:
    """Utilities for streamline manipulation and analysis"""

    @staticmethod
    def compute_length(streamline: np.ndarray) -> float:
        """
        Compute streamline length in mm

        Args:
            streamline: Array of points (N, 3)

        Returns:
            Total length in mm
        """
        if len(streamline) < 2:
            return 0.0

        segments = np.diff(streamline, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        return np.sum(lengths)

    @staticmethod
    def compute_lengths(streamlines: List[np.ndarray]) -> np.ndarray:
        """
        Compute lengths for multiple streamlines

        Args:
            streamlines: List of streamlines

        Returns:
            Array of lengths (N,)
        """
        return np.array([StreamlineUtils.compute_length(s) for s in streamlines])

    @staticmethod
    def filter_by_length(
        streamlines: List[np.ndarray],
        min_length: float = 10.0,
        max_length: float = 200.0
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Filter streamlines by length

        Args:
            streamlines: Input streamlines
            min_length: Minimum length in mm
            max_length: Maximum length in mm

        Returns:
            filtered_streamlines: Filtered streamlines
            kept_indices: Indices of kept streamlines
        """
        lengths = StreamlineUtils.compute_lengths(streamlines)
        keep_mask = (lengths >= min_length) & (lengths <= max_length)
        kept_indices = np.where(keep_mask)[0]

        filtered = [streamlines[i] for i in kept_indices]

        logger.info(
            f"Length filter: kept {len(filtered)}/{len(streamlines)} streamlines "
            f"({min_length:.1f}-{max_length:.1f}mm)"
        )

        return filtered, kept_indices

    @staticmethod
    def compute_curvature(streamline: np.ndarray) -> np.ndarray:
        """
        Compute local curvature along streamline

        Curvature is computed as the angle between consecutive segments

        Args:
            streamline: Array of points (N, 3)

        Returns:
            Curvature angles in degrees (N-2,)
        """
        if len(streamline) < 3:
            return np.array([])

        # Compute tangent vectors
        tangents = np.diff(streamline, axis=0)
        tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / (tangent_norms + 1e-10)

        # Compute angles between consecutive tangents
        dot_products = np.sum(tangents[:-1] * tangents[1:], axis=1)
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angles = np.arccos(dot_products)

        # Convert to degrees
        curvatures = np.degrees(angles)

        return curvatures

    @staticmethod
    def filter_by_curvature(
        streamlines: List[np.ndarray],
        max_curvature: float = 90.0
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Filter streamlines by maximum curvature

        Args:
            streamlines: Input streamlines
            max_curvature: Maximum allowed curvature in degrees

        Returns:
            filtered_streamlines: Filtered streamlines
            kept_indices: Indices of kept streamlines
        """
        keep_indices = []

        for i, streamline in enumerate(streamlines):
            curvatures = StreamlineUtils.compute_curvature(streamline)

            if len(curvatures) == 0 or np.max(curvatures) <= max_curvature:
                keep_indices.append(i)

        kept_indices = np.array(keep_indices)
        filtered = [streamlines[i] for i in kept_indices]

        logger.info(
            f"Curvature filter: kept {len(filtered)}/{len(streamlines)} streamlines "
            f"(max_curvature={max_curvature}°)"
        )

        return filtered, kept_indices

    @staticmethod
    def smooth_streamline(
        streamline: np.ndarray,
        sigma: float = 1.0
    ) -> np.ndarray:
        """
        Smooth streamline using Gaussian filter

        Args:
            streamline: Input streamline (N, 3)
            sigma: Gaussian kernel standard deviation (points)

        Returns:
            Smoothed streamline (N, 3)
        """
        if len(streamline) < 3:
            return streamline

        # Smooth each coordinate independently
        smoothed = np.zeros_like(streamline)
        for i in range(3):
            smoothed[:, i] = gaussian_filter1d(
                streamline[:, i],
                sigma=sigma,
                mode='nearest'
            )

        return smoothed

    @staticmethod
    def resample_streamline(
        streamline: np.ndarray,
        n_points: Optional[int] = None,
        step_size: Optional[float] = None
    ) -> np.ndarray:
        """
        Resample streamline to uniform spacing

        Args:
            streamline: Input streamline (N, 3)
            n_points: Target number of points (exclusive with step_size)
            step_size: Target spacing in mm (exclusive with n_points)

        Returns:
            Resampled streamline (M, 3)
        """
        if len(streamline) < 2:
            return streamline

        if n_points is None and step_size is None:
            raise ValueError("Either n_points or step_size must be specified")

        if n_points is not None and step_size is not None:
            raise ValueError("Only one of n_points or step_size can be specified")

        # Compute cumulative arc length
        segments = np.diff(streamline, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        arc_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = arc_length[-1]

        # Determine target arc length samples
        if n_points is not None:
            target_arc = np.linspace(0, total_length, n_points)
        else:
            n_points = max(2, int(np.ceil(total_length / step_size)) + 1)
            target_arc = np.linspace(0, total_length, n_points)

        # Interpolate coordinates
        resampled = np.zeros((len(target_arc), 3))
        for i in range(3):
            interp_func = interp1d(
                arc_length,
                streamline[:, i],
                kind='linear',
                assume_sorted=True
            )
            resampled[:, i] = interp_func(target_arc)

        return resampled

    @staticmethod
    def compute_bundle_statistics(
        streamlines: List[np.ndarray]
    ) -> Dict:
        """
        Compute statistics for a bundle of streamlines

        Args:
            streamlines: List of streamlines

        Returns:
            Dictionary of statistics
        """
        if not streamlines:
            return {
                'n_streamlines': 0,
                'mean_length': 0.0,
                'std_length': 0.0,
                'min_length': 0.0,
                'max_length': 0.0
            }

        lengths = StreamlineUtils.compute_lengths(streamlines)
        n_points = np.array([len(s) for s in streamlines])

        stats = {
            'n_streamlines': len(streamlines),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'median_length': np.median(lengths),
            'mean_n_points': np.mean(n_points),
            'total_length': np.sum(lengths)
        }

        return stats

    @staticmethod
    def cluster_streamlines(
        streamlines: List[np.ndarray],
        threshold: float = 10.0,
        method: str = 'quickbundles'
    ) -> Tuple[List[int], List[np.ndarray]]:
        """
        Cluster streamlines into bundles

        Args:
            streamlines: Input streamlines
            threshold: Distance threshold in mm for clustering
            method: Clustering method ('quickbundles' or 'hierarchical')

        Returns:
            labels: Cluster labels for each streamline
            centroids: Representative streamline for each cluster
        """
        logger.info(
            f"Clustering {len(streamlines)} streamlines using {method} "
            f"(threshold={threshold}mm)"
        )

        if method == 'quickbundles':
            labels, centroids = StreamlineUtils._quickbundles_clustering(
                streamlines, threshold
            )
        elif method == 'hierarchical':
            labels, centroids = StreamlineUtils._hierarchical_clustering(
                streamlines, threshold
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        n_clusters = len(np.unique(labels))
        logger.info(f"Found {n_clusters} clusters")

        return labels, centroids

    @staticmethod
    def _quickbundles_clustering(
        streamlines: List[np.ndarray],
        threshold: float
    ) -> Tuple[List[int], List[np.ndarray]]:
        """
        QuickBundles clustering algorithm

        Simple greedy clustering based on MDF (Minimum Direct-Flip) distance
        """
        if not streamlines:
            return [], []

        # Resample all streamlines to same number of points for distance computation
        n_points = 20
        resampled = [
            StreamlineUtils.resample_streamline(s, n_points=n_points)
            for s in streamlines
        ]

        clusters = []
        centroids = []
        labels = [-1] * len(streamlines)

        for i, streamline in enumerate(resampled):
            # Find closest cluster
            min_distance = float('inf')
            min_cluster = -1

            for cluster_idx, centroid in enumerate(centroids):
                distance = StreamlineUtils._mdf_distance(streamline, centroid)

                if distance < min_distance:
                    min_distance = distance
                    min_cluster = cluster_idx

            # Assign to cluster or create new
            if min_distance < threshold:
                clusters[min_cluster].append(i)
                labels[i] = min_cluster

                # Update centroid (mean of cluster)
                cluster_streamlines = [resampled[j] for j in clusters[min_cluster]]
                centroids[min_cluster] = np.mean(cluster_streamlines, axis=0)
            else:
                # Create new cluster
                clusters.append([i])
                centroids.append(streamline)
                labels[i] = len(clusters) - 1

        # Convert centroids to original streamlines
        centroid_streamlines = []
        for cluster in clusters:
            # Use medoid (most representative streamline)
            cluster_resampled = [resampled[i] for i in cluster]
            centroid = centroids[len(centroid_streamlines)]

            distances = [
                StreamlineUtils._mdf_distance(s, centroid)
                for s in cluster_resampled
            ]
            medoid_idx = cluster[np.argmin(distances)]
            centroid_streamlines.append(streamlines[medoid_idx])

        return labels, centroid_streamlines

    @staticmethod
    def _hierarchical_clustering(
        streamlines: List[np.ndarray],
        threshold: float
    ) -> Tuple[List[int], List[np.ndarray]]:
        """
        Hierarchical clustering of streamlines

        Uses average linkage with MDF distance
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        # Resample streamlines
        n_points = 20
        resampled = [
            StreamlineUtils.resample_streamline(s, n_points=n_points)
            for s in streamlines
        ]

        # Compute pairwise distance matrix
        n_streamlines = len(resampled)
        distances = np.zeros((n_streamlines, n_streamlines))

        for i in range(n_streamlines):
            for j in range(i + 1, n_streamlines):
                dist = StreamlineUtils._mdf_distance(resampled[i], resampled[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Convert to condensed distance matrix
        condensed = squareform(distances)

        # Hierarchical clustering
        linkage_matrix = linkage(condensed, method='average')
        labels = fcluster(linkage_matrix, threshold, criterion='distance')

        # Convert to 0-based indexing
        labels = labels - 1

        # Compute centroids
        unique_labels = np.unique(labels)
        centroids = []

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_streamlines = [resampled[i] for i in cluster_indices]

            # Compute mean centroid
            centroid = np.mean(cluster_streamlines, axis=0)

            # Find medoid
            distances_to_centroid = [
                StreamlineUtils._mdf_distance(s, centroid)
                for s in cluster_streamlines
            ]
            medoid_idx = cluster_indices[np.argmin(distances_to_centroid)]
            centroids.append(streamlines[medoid_idx])

        return labels.tolist(), centroids

    @staticmethod
    def _mdf_distance(
        streamline1: np.ndarray,
        streamline2: np.ndarray
    ) -> float:
        """
        Minimum Direct-Flip (MDF) distance between streamlines

        Considers both forward and flipped orientations

        Args:
            streamline1: First streamline (N, 3)
            streamline2: Second streamline (N, 3)

        Returns:
            MDF distance in mm
        """
        # Direct distance
        direct = np.mean(np.linalg.norm(streamline1 - streamline2, axis=1))

        # Flipped distance
        flipped = np.mean(np.linalg.norm(streamline1 - streamline2[::-1], axis=1))

        return min(direct, flipped)

    @staticmethod
    def save_trk(
        streamlines: List[np.ndarray],
        filepath: str,
        affine: np.ndarray,
        voxel_size: np.ndarray,
        dimensions: np.ndarray
    ):
        """
        Save streamlines in TrackVis TRK format

        Args:
            streamlines: List of streamlines in world coordinates (RASMM)
            filepath: Output file path
            affine: Voxel-to-world affine transformation matrix (4x4)
            voxel_size: Voxel dimensions (3,)
            dimensions: Volume dimensions (3,)
        """
        import nibabel as nib
        from nibabel.streamlines import Tractogram
        from nibabel.streamlines.trk import TrkFile

        logger.info(f"Saving {len(streamlines)} streamlines to TRK: {filepath}")

        # Convert streamlines from RASMM (world) to voxel space
        # so that affine_to_rasmm correctly maps them back when loading
        inv_affine = np.linalg.inv(affine)
        streamlines_voxel = []
        for sl in streamlines:
            sl_hom = np.hstack([sl, np.ones((len(sl), 1))])
            sl_vox = (inv_affine @ sl_hom.T).T[:, :3]
            streamlines_voxel.append(sl_vox.astype(np.float32))

        # Create tractogram with voxel-space streamlines and the correct affine
        tractogram = Tractogram(
            streamlines=streamlines_voxel,
            affine_to_rasmm=affine
        )

        # Create and save TRK file
        trk = TrkFile(tractogram, header=None)
        TrkFile.save(trk, filepath)

        logger.info(f"Saved to: {filepath}")

    @staticmethod
    def save_tck(
        streamlines: List[np.ndarray],
        filepath: str
    ):
        """
        Save streamlines in MRtrix TCK format

        Args:
            streamlines: List of streamlines in world coordinates
            filepath: Output file path
        """
        from nibabel.streamlines import Tractogram, TckFile

        logger.info(f"Saving {len(streamlines)} streamlines to TCK: {filepath}")

        tractogram = Tractogram(streamlines=streamlines)
        TckFile(tractogram).save(filepath)

        logger.info(f"Saved to: {filepath}")

    @staticmethod
    def save_vtk(
        streamlines: List[np.ndarray],
        filepath: str,
        scalars: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Save streamlines in VTK format for visualization

        Args:
            streamlines: List of streamlines
            filepath: Output file path
            scalars: Optional scalar data per streamline
        """
        logger.info(f"Saving {len(streamlines)} streamlines to VTK: {filepath}")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            # Header
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Streamlines\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")

            # Count total points
            total_points = sum(len(s) for s in streamlines)
            f.write(f"POINTS {total_points} float\n")

            # Write points
            for streamline in streamlines:
                for point in streamline:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")

            # Write lines
            total_connectivity = sum(len(s) + 1 for s in streamlines)
            f.write(f"\nLINES {len(streamlines)} {total_connectivity}\n")

            point_offset = 0
            for streamline in streamlines:
                n_points = len(streamline)
                indices = " ".join(str(point_offset + i) for i in range(n_points))
                f.write(f"{n_points} {indices}\n")
                point_offset += n_points

            # Write scalar data if provided
            if scalars:
                f.write(f"\nCELL_DATA {len(streamlines)}\n")
                for name, data in scalars.items():
                    f.write(f"SCALARS {name} float 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for value in data:
                        f.write(f"{value}\n")

        logger.info(f"Saved to: {filepath}")

    @staticmethod
    def load_trk(filepath: str) -> Tuple[List[np.ndarray], Dict]:
        """
        Load streamlines from TRK file

        Args:
            filepath: Path to TRK file

        Returns:
            streamlines: List of streamlines (in RASMM/world coordinates)
            header: TRK header information
        """
        from nibabel.streamlines import load

        logger.info(f"Loading TRK file: {filepath}")

        trk = load(filepath)
        streamlines = list(trk.streamlines)

        # Extract header info safely
        header_info = {}
        try:
            header_info['voxel_to_rasmm'] = trk.header.get('voxel_to_rasmm', None)
            header_info['dimensions'] = trk.header.get('dimensions', None)
            header_info['voxel_sizes'] = trk.header.get('voxel_sizes', None)
            header_info['nb_streamlines'] = len(streamlines)
        except Exception:
            header_info['nb_streamlines'] = len(streamlines)

        logger.info(f"Loaded {len(streamlines)} streamlines")

        return streamlines, header_info

    @staticmethod
    def load_tck(filepath: str) -> List[np.ndarray]:
        """
        Load streamlines from TCK file

        Args:
            filepath: Path to TCK file

        Returns:
            streamlines: List of streamlines
        """
        from nibabel.streamlines import load

        logger.info(f"Loading TCK file: {filepath}")

        tck = load(filepath)
        streamlines = tck.streamlines

        logger.info(f"Loaded {len(streamlines)} streamlines")

        return list(streamlines)
