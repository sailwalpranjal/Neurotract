"""
Connectome Construction Module

Builds structural connectomes from tractography streamlines and parcellations.
Implements multiple edge weighting strategies.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ConnectomeBuilder:
    """
    Construct structural connectome from streamlines and parcellation
    """

    def __init__(
        self,
        parcellation: np.ndarray,
        n_parcels: Optional[int] = None,
        parcel_labels: Optional[List[str]] = None,
        affine: Optional[np.ndarray] = None
    ):
        """
        Initialize connectome builder

        Args:
            parcellation: 3D parcellation map (x, y, z)
            n_parcels: Number of parcels (auto-detected if None)
            parcel_labels: Names of parcels
            affine: Parcellation voxel-to-world affine (4x4). If provided,
                    streamline points are converted from world to voxel space.
        """
        self.parcellation = parcellation
        self.shape = parcellation.shape

        # Store affine for world-to-voxel conversion
        self.affine = affine
        if affine is not None:
            self.inv_affine = np.linalg.inv(affine)
        else:
            self.inv_affine = None

        if n_parcels is None:
            self.n_parcels = int(np.max(parcellation)) + 1
        else:
            self.n_parcels = n_parcels

        if parcel_labels is None:
            self.parcel_labels = [f"parcel_{i}" for i in range(self.n_parcels)]
        else:
            self.parcel_labels = parcel_labels

        logger.info(f"Connectome builder initialized: {self.n_parcels} parcels"
                   f"{', with affine transform' if affine is not None else ''}")

    def build_connectome(
        self,
        streamlines: List[np.ndarray],
        weighting: str = 'count',
        microstructure_map: Optional[np.ndarray] = None,
        symmetric: bool = True,
        normalize: bool = False
    ) -> np.ndarray:
        """
        Build connectome from streamlines

        Args:
            streamlines: List of streamlines, each shape (n_points, 3)
            weighting: Edge weighting strategy:
                - 'count': Number of streamlines
                - 'length_normalized': Count / mean_length
                - 'mean_fa': Mean FA along streamline
                - 'mean_md': Mean MD along streamline
                - 'hybrid': Combines count and microstructure
            microstructure_map: 3D map for microstructure weighting (e.g., FA, MD)
            symmetric: Make adjacency matrix symmetric
            normalize: Normalize by seeding density

        Returns:
            Adjacency matrix (n_parcels, n_parcels)
        """
        logger.info(f"Building connectome with {len(streamlines)} streamlines...")
        logger.info(f"Weighting strategy: {weighting}")

        # Initialize adjacency matrix
        adjacency = np.zeros((self.n_parcels, self.n_parcels), dtype=np.float64)

        # Count connections and accumulate weights
        streamline_lengths = []
        streamline_weights = []

        for streamline_idx, streamline in enumerate(streamlines):
            if len(streamline) < 2:
                continue

            # Find endpoint parcels
            start_parcel = self._get_parcel_at_point(streamline[0])
            end_parcel = self._get_parcel_at_point(streamline[-1])

            if start_parcel < 0 or end_parcel < 0:
                continue  # Streamline endpoints not in parcellation

            if start_parcel == end_parcel:
                continue  # Skip loops

            # Compute weight based on strategy
            weight = self._compute_streamline_weight(
                streamline, weighting, microstructure_map
            )

            # Add to adjacency matrix
            adjacency[start_parcel, end_parcel] += weight
            if symmetric:
                adjacency[end_parcel, start_parcel] += weight

            # Track for normalization
            streamline_lengths.append(self._compute_streamline_length(streamline))
            streamline_weights.append(weight)

            # Progress logging
            if (streamline_idx + 1) % max(1, len(streamlines) // 10) == 0:
                logger.debug(f"Processed {streamline_idx+1}/{len(streamlines)} streamlines")

        # Normalize if requested
        if normalize:
            logger.info("Normalizing by parcel volumes...")
            adjacency = self._normalize_by_volume(adjacency)

        logger.info(f"Connectome built: {np.count_nonzero(adjacency)} edges, "
                   f"density={np.count_nonzero(adjacency) / (self.n_parcels ** 2):.3f}")

        return adjacency

    def _point_to_voxel(self, point: np.ndarray) -> np.ndarray:
        """Convert point from world to voxel coordinates if affine is set"""
        if self.inv_affine is not None:
            point_hom = np.array([point[0], point[1], point[2], 1.0])
            voxel = self.inv_affine @ point_hom
            return voxel[:3]
        return point

    def _get_parcel_at_point(self, point: np.ndarray) -> int:
        """
        Get parcel label at a point

        Args:
            point: 3D coordinates (world or voxel space)

        Returns:
            Parcel index, or -1 if out of bounds
        """
        # Convert to voxel coordinates if needed
        voxel_point = self._point_to_voxel(point)
        x, y, z = np.round(voxel_point).astype(int)

        # Check bounds
        if (x < 0 or x >= self.shape[0] or
            y < 0 or y >= self.shape[1] or
            z < 0 or z >= self.shape[2]):
            return -1

        parcel = int(self.parcellation[x, y, z])
        return parcel if parcel < self.n_parcels else -1

    def _compute_streamline_weight(
        self,
        streamline: np.ndarray,
        strategy: str,
        microstructure_map: Optional[np.ndarray]
    ) -> float:
        """
        Compute streamline weight based on strategy

        Args:
            streamline: Streamline coordinates (n_points, 3)
            strategy: Weighting strategy
            microstructure_map: Optional microstructure map (FA, MD, etc.)

        Returns:
            Weight value
        """
        if strategy == 'count':
            return 1.0

        elif strategy == 'length_normalized':
            length = self._compute_streamline_length(streamline)
            return 1.0 / (length + 1e-6)

        elif strategy in ['mean_fa', 'mean_md']:
            if microstructure_map is None:
                logger.warning(f"{strategy} requires microstructure_map, using count")
                return 1.0

            # Sample microstructure along streamline
            values = []
            for point in streamline:
                value = self._sample_at_point(point, microstructure_map)
                if value is not None:
                    values.append(value)

            if len(values) == 0:
                return 0.0

            return np.mean(values)

        elif strategy == 'hybrid':
            # Combine count with FA
            if microstructure_map is None:
                return 1.0

            fa_mean = 0.0
            values = []
            for point in streamline:
                value = self._sample_at_point(point, microstructure_map)
                if value is not None:
                    values.append(value)

            if len(values) > 0:
                fa_mean = np.mean(values)

            # Hybrid: count * (1 + FA)
            return 1.0 * (1.0 + fa_mean)

        else:
            logger.warning(f"Unknown weighting strategy: {strategy}, using count")
            return 1.0

    def _sample_at_point(
        self,
        point: np.ndarray,
        volume: np.ndarray
    ) -> Optional[float]:
        """
        Sample volume at point using trilinear interpolation

        Args:
            point: 3D coordinates (world or voxel space)
            volume: 3D volume

        Returns:
            Interpolated value, or None if out of bounds
        """
        voxel_point = self._point_to_voxel(point)
        x, y, z = voxel_point

        # Get integer coordinates
        x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

        # Check bounds
        if (x0 < 0 or x1 >= volume.shape[0] or
            y0 < 0 or y1 >= volume.shape[1] or
            z0 < 0 or z1 >= volume.shape[2]):
            return None

        # Trilinear interpolation weights
        xd = x - x0
        yd = y - y0
        zd = z - z0

        # Interpolate
        c000 = volume[x0, y0, z0]
        c001 = volume[x0, y0, z1]
        c010 = volume[x0, y1, z0]
        c011 = volume[x0, y1, z1]
        c100 = volume[x1, y0, z0]
        c101 = volume[x1, y0, z1]
        c110 = volume[x1, y1, z0]
        c111 = volume[x1, y1, z1]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        value = c0 * (1 - zd) + c1 * zd

        return float(value)

    def _compute_streamline_length(self, streamline: np.ndarray) -> float:
        """Compute total length of streamline"""
        if len(streamline) < 2:
            return 0.0

        diffs = np.diff(streamline, axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
        return np.sum(lengths)

    def _normalize_by_volume(self, adjacency: np.ndarray) -> np.ndarray:
        """Normalize edge weights by parcel volumes"""

        # Compute parcel volumes (number of voxels)
        volumes = np.array([
            np.sum(self.parcellation == i) for i in range(self.n_parcels)
        ])

        # Avoid division by zero
        volumes = np.maximum(volumes, 1)

        # Normalize each edge by geometric mean of endpoint volumes
        normalized = adjacency.copy()
        for i in range(self.n_parcels):
            for j in range(self.n_parcels):
                if adjacency[i, j] > 0:
                    norm_factor = np.sqrt(volumes[i] * volumes[j])
                    normalized[i, j] = adjacency[i, j] / norm_factor

        return normalized

    def compute_edge_statistics(
        self,
        streamlines: List[np.ndarray],
        adjacency: np.ndarray
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Compute per-edge statistics (length, count, variance, etc.)

        Args:
            streamlines: List of streamlines
            adjacency: Pre-computed adjacency matrix

        Returns:
            Dictionary mapping (i, j) to edge statistics
        """
        logger.info("Computing edge statistics...")

        edge_stats = {}

        # Group streamlines by endpoints
        for streamline in streamlines:
            if len(streamline) < 2:
                continue

            start = self._get_parcel_at_point(streamline[0])
            end = self._get_parcel_at_point(streamline[-1])

            if start < 0 or end < 0 or start == end:
                continue

            edge = (start, end)
            if edge not in edge_stats:
                edge_stats[edge] = {
                    'lengths': [],
                    'count': 0
                }

            length = self._compute_streamline_length(streamline)
            edge_stats[edge]['lengths'].append(length)
            edge_stats[edge]['count'] += 1

        # Compute summary statistics
        for edge, stats in edge_stats.items():
            lengths = np.array(stats['lengths'])
            stats['mean_length'] = np.mean(lengths)
            stats['std_length'] = np.std(lengths)
            stats['min_length'] = np.min(lengths)
            stats['max_length'] = np.max(lengths)

        logger.info(f"Computed statistics for {len(edge_stats)} edges")

        return edge_stats


def load_parcellation(
    parcellation_file: str,
    labels_file: Optional[str] = None
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Load parcellation from file

    Args:
        parcellation_file: Path to parcellation NIfTI
        labels_file: Optional path to label names (text file, one per line)

    Returns:
        Tuple of (parcellation array, label names, affine matrix)
    """
    import nibabel as nib

    logger.info(f"Loading parcellation from {parcellation_file}")

    img = nib.load(parcellation_file)
    parcellation = img.get_fdata().astype(int)
    affine = img.affine

    n_parcels = int(np.max(parcellation)) + 1

    if labels_file is not None:
        with open(labels_file, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
    else:
        labels = [f"parcel_{i}" for i in range(n_parcels)]

    logger.info(f"Loaded parcellation: {n_parcels} parcels")

    return parcellation, labels, affine
