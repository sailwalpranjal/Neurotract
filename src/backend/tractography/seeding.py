"""
Seed Generation Strategies for Tractography

Implements various seeding strategies including:
- Whole-brain dense seeding
- FOD amplitude threshold seeding
- Region-of-interest seeding
- Parcel boundary seeding
- Deterministic RNG with seed logging for reproducibility
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class SeedGenerator:
    """
    Generates seed points for tractography with deterministic RNG
    """

    def __init__(
        self,
        rng_seed: Optional[int] = None,
        log_dir: Optional[str] = "analysis_and_decisions"
    ):
        """
        Initialize seed generator

        Args:
            rng_seed: Random number generator seed (None = current time)
            log_dir: Directory to log seed information
        """
        if rng_seed is None:
            rng_seed = int(datetime.now().timestamp() * 1000000) % (2**32)

        self.rng_seed = rng_seed
        self.rng = np.random.RandomState(rng_seed)
        self.log_dir = Path(log_dir) if log_dir else None

        # Store generated seeds for reproducibility
        self.seed_log = {
            'rng_seed': rng_seed,
            'timestamp': datetime.now().isoformat(),
            'seed_strategies': []
        }

        logger.info(f"SeedGenerator initialized with RNG seed: {rng_seed}")

    def whole_brain_seeds(
        self,
        mask: np.ndarray,
        seeds_per_voxel: int = 1,
        voxel_size: Optional[np.ndarray] = None,
        jitter: bool = True,
        jitter_scale: float = 0.5
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate seeds uniformly distributed across all brain voxels

        Args:
            mask: Binary brain mask (x, y, z)
            seeds_per_voxel: Number of seeds per voxel
            voxel_size: Voxel dimensions (3,) for density calculation
            jitter: Whether to add random jitter to seed positions
            jitter_scale: Scale of jitter relative to voxel size (0-1)

        Returns:
            seeds: Array of seed positions in voxel coordinates (N, 3)
            metadata: Dictionary with seeding information
        """
        logger.info(
            f"Generating whole-brain seeds: {seeds_per_voxel} seeds/voxel, "
            f"jitter={jitter}"
        )

        # Find all voxels inside mask
        voxel_indices = np.array(np.where(mask > 0)).T  # (N_voxels, 3)
        n_voxels = voxel_indices.shape[0]

        if n_voxels == 0:
            raise ValueError("Mask is empty - no seeds can be generated")

        # Generate seeds
        if seeds_per_voxel == 1 and not jitter:
            # Simple case: one seed per voxel at center
            seeds = voxel_indices.astype(np.float32) + 0.5
        else:
            # Multiple seeds per voxel or with jitter
            seeds = []
            for voxel in voxel_indices:
                for _ in range(seeds_per_voxel):
                    if jitter:
                        # Random position within voxel
                        offset = self.rng.uniform(0, 1, size=3) * jitter_scale
                        seed = voxel.astype(np.float32) + 0.5 - jitter_scale/2 + offset
                    else:
                        # Center of voxel
                        seed = voxel.astype(np.float32) + 0.5

                    seeds.append(seed)

            seeds = np.array(seeds, dtype=np.float32)

        # Calculate seeding density
        if voxel_size is not None:
            voxel_volume_mm3 = np.prod(voxel_size)
            total_volume_mm3 = n_voxels * voxel_volume_mm3
            density_per_mm3 = len(seeds) / total_volume_mm3
        else:
            density_per_mm3 = None

        # Metadata
        metadata = {
            'strategy': 'whole_brain',
            'seeds_per_voxel': seeds_per_voxel,
            'n_voxels': n_voxels,
            'n_seeds': len(seeds),
            'jitter': jitter,
            'jitter_scale': jitter_scale,
            'density_per_mm3': density_per_mm3,
            'rng_seed': self.rng_seed
        }

        self._log_seed_strategy(metadata)

        logger.info(
            f"Generated {len(seeds)} seeds from {n_voxels} voxels "
            f"(density: {density_per_mm3:.2f} seeds/mm³)" if density_per_mm3
            else f"Generated {len(seeds)} seeds from {n_voxels} voxels"
        )

        return seeds, metadata

    def fod_threshold_seeds(
        self,
        fod_amplitude: np.ndarray,
        mask: np.ndarray,
        threshold: float = 0.1,
        seeds_per_voxel: int = 2,
        adaptive_density: bool = True,
        jitter: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate seeds based on FOD amplitude threshold

        Seeds are placed in voxels with sufficient FOD amplitude.
        Optionally adapts seed density to FOD amplitude.

        Args:
            fod_amplitude: FOD amplitude map (x, y, z)
            mask: Binary brain mask (x, y, z)
            threshold: Minimum FOD amplitude for seeding
            seeds_per_voxel: Base number of seeds per voxel
            adaptive_density: Increase seed density in high FOD regions
            jitter: Add random jitter to seed positions

        Returns:
            seeds: Array of seed positions (N, 3)
            metadata: Seeding metadata
        """
        logger.info(
            f"Generating FOD threshold seeds: threshold={threshold}, "
            f"seeds_per_voxel={seeds_per_voxel}, adaptive={adaptive_density}"
        )

        # Apply threshold to mask
        seeding_mask = (fod_amplitude >= threshold) & (mask > 0)
        voxel_indices = np.array(np.where(seeding_mask)).T

        if len(voxel_indices) == 0:
            raise ValueError(
                f"No voxels exceed FOD threshold {threshold} - "
                "try lowering threshold"
            )

        seeds = []

        for voxel in voxel_indices:
            x, y, z = voxel

            # Determine number of seeds for this voxel
            if adaptive_density:
                # More seeds in high FOD regions
                fod_val = fod_amplitude[x, y, z]
                # Scale seeds: threshold -> 1x, max FOD -> 3x
                max_fod = np.max(fod_amplitude[seeding_mask])
                scale = 1.0 + 2.0 * (fod_val - threshold) / (max_fod - threshold)
                n_seeds_voxel = max(1, int(seeds_per_voxel * scale))
            else:
                n_seeds_voxel = seeds_per_voxel

            # Generate seeds in this voxel
            for _ in range(n_seeds_voxel):
                if jitter:
                    offset = self.rng.uniform(0, 1, size=3)
                    seed = voxel.astype(np.float32) + offset
                else:
                    seed = voxel.astype(np.float32) + 0.5

                seeds.append(seed)

        seeds = np.array(seeds, dtype=np.float32)

        # Metadata
        metadata = {
            'strategy': 'fod_threshold',
            'threshold': threshold,
            'seeds_per_voxel': seeds_per_voxel,
            'adaptive_density': adaptive_density,
            'n_voxels': len(voxel_indices),
            'n_seeds': len(seeds),
            'jitter': jitter,
            'rng_seed': self.rng_seed
        }

        self._log_seed_strategy(metadata)

        logger.info(
            f"Generated {len(seeds)} seeds from {len(voxel_indices)} voxels "
            f"exceeding FOD threshold"
        )

        return seeds, metadata

    def roi_seeds(
        self,
        roi_mask: np.ndarray,
        seeds_per_voxel: int = 5,
        jitter: bool = True,
        boundary_only: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate seeds within region-of-interest

        Args:
            roi_mask: Binary ROI mask (x, y, z)
            seeds_per_voxel: Number of seeds per voxel
            jitter: Add random jitter to positions
            boundary_only: Only seed at ROI boundary voxels

        Returns:
            seeds: Seed positions (N, 3)
            metadata: Seeding metadata
        """
        logger.info(
            f"Generating ROI seeds: seeds_per_voxel={seeds_per_voxel}, "
            f"boundary_only={boundary_only}"
        )

        if boundary_only:
            # Extract boundary voxels using erosion
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(roi_mask)
            boundary_mask = roi_mask & ~eroded
            voxel_indices = np.array(np.where(boundary_mask)).T
        else:
            voxel_indices = np.array(np.where(roi_mask > 0)).T

        if len(voxel_indices) == 0:
            raise ValueError("ROI mask is empty")

        seeds = []
        for voxel in voxel_indices:
            for _ in range(seeds_per_voxel):
                if jitter:
                    offset = self.rng.uniform(0, 1, size=3)
                    seed = voxel.astype(np.float32) + offset
                else:
                    seed = voxel.astype(np.float32) + 0.5

                seeds.append(seed)

        seeds = np.array(seeds, dtype=np.float32)

        metadata = {
            'strategy': 'roi',
            'boundary_only': boundary_only,
            'seeds_per_voxel': seeds_per_voxel,
            'n_voxels': len(voxel_indices),
            'n_seeds': len(seeds),
            'jitter': jitter,
            'rng_seed': self.rng_seed
        }

        self._log_seed_strategy(metadata)

        logger.info(f"Generated {len(seeds)} seeds in ROI")

        return seeds, metadata

    def parcel_boundary_seeds(
        self,
        parcellation: np.ndarray,
        seeds_per_boundary: int = 10,
        min_parcel_size: int = 10
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate seeds at boundaries between parcels

        Useful for connectome construction to ensure inter-parcel connections

        Args:
            parcellation: Parcellation label volume (x, y, z)
            seeds_per_boundary: Seeds per boundary voxel
            min_parcel_size: Minimum parcel size to consider

        Returns:
            seeds: Seed positions (N, 3)
            metadata: Seeding metadata
        """
        logger.info(
            f"Generating parcel boundary seeds: "
            f"seeds_per_boundary={seeds_per_boundary}"
        )

        # Find parcel labels
        unique_labels = np.unique(parcellation)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background

        # Filter small parcels
        valid_labels = []
        for label in unique_labels:
            if np.sum(parcellation == label) >= min_parcel_size:
                valid_labels.append(label)

        logger.info(f"Found {len(valid_labels)} valid parcels")

        # Find boundary voxels (voxels neighboring different labels)
        from scipy.ndimage import generate_binary_structure, generic_filter

        def is_boundary(values):
            """Check if center voxel is at boundary"""
            center = values[len(values)//2]
            if center == 0:
                return 0
            return int(len(np.unique(values[values > 0])) > 1)

        # Create connectivity structure (6-connectivity)
        struct = generate_binary_structure(3, 1)

        # Find boundary voxels
        boundary_mask = generic_filter(
            parcellation.astype(np.int32),
            is_boundary,
            footprint=struct,
            mode='constant',
            cval=0
        ).astype(bool)

        voxel_indices = np.array(np.where(boundary_mask)).T

        if len(voxel_indices) == 0:
            raise ValueError("No parcel boundaries found")

        # Generate seeds
        seeds = []
        for voxel in voxel_indices:
            for _ in range(seeds_per_boundary):
                offset = self.rng.uniform(0, 1, size=3)
                seed = voxel.astype(np.float32) + offset
                seeds.append(seed)

        seeds = np.array(seeds, dtype=np.float32)

        metadata = {
            'strategy': 'parcel_boundary',
            'n_parcels': len(valid_labels),
            'seeds_per_boundary': seeds_per_boundary,
            'n_boundary_voxels': len(voxel_indices),
            'n_seeds': len(seeds),
            'min_parcel_size': min_parcel_size,
            'rng_seed': self.rng_seed
        }

        self._log_seed_strategy(metadata)

        logger.info(
            f"Generated {len(seeds)} seeds at {len(voxel_indices)} "
            f"boundary voxels"
        )

        return seeds, metadata

    def random_subsample(
        self,
        seeds: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Randomly subsample seeds for tractography

        Args:
            seeds: Input seed array (N, 3)
            n_samples: Number of seeds to sample

        Returns:
            Subsampled seeds (n_samples, 3)
        """
        if n_samples >= len(seeds):
            logger.warning(
                f"Requested {n_samples} samples but only {len(seeds)} "
                "seeds available - returning all seeds"
            )
            return seeds

        # Random selection without replacement
        indices = self.rng.choice(len(seeds), size=n_samples, replace=False)
        subsampled = seeds[indices]

        logger.info(f"Subsampled {n_samples} seeds from {len(seeds)} total")

        return subsampled

    def _log_seed_strategy(self, metadata: Dict):
        """Log seed strategy to seed log"""
        self.seed_log['seed_strategies'].append({
            'timestamp': datetime.now().isoformat(),
            **metadata
        })

    def save_seed_log(self, filename: Optional[str] = None):
        """
        Save seed log to file for reproducibility

        Args:
            filename: Output filename (auto-generated if None)
        """
        if self.log_dir is None:
            logger.warning("No log directory specified - skipping seed log save")
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"seed_log_{timestamp}.json"

        filepath = self.log_dir / filename

        with open(filepath, 'w') as f:
            json.dump(self.seed_log, f, indent=2)

        logger.info(f"Seed log saved to: {filepath}")

    @staticmethod
    def load_seeds_from_file(filepath: str) -> np.ndarray:
        """
        Load seeds from saved file

        Args:
            filepath: Path to seed file (.npy or .txt)

        Returns:
            Loaded seeds (N, 3)
        """
        filepath = Path(filepath)

        if filepath.suffix == '.npy':
            seeds = np.load(filepath)
        elif filepath.suffix == '.txt':
            seeds = np.loadtxt(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        logger.info(f"Loaded {len(seeds)} seeds from {filepath}")

        return seeds

    @staticmethod
    def save_seeds_to_file(seeds: np.ndarray, filepath: str):
        """
        Save seeds to file

        Args:
            seeds: Seed array (N, 3)
            filepath: Output path (.npy or .txt)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.suffix == '.npy':
            np.save(filepath, seeds)
        elif filepath.suffix == '.txt':
            np.savetxt(filepath, seeds, fmt='%.6f')
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        logger.info(f"Saved {len(seeds)} seeds to {filepath}")

    def get_reproducibility_info(self) -> Dict:
        """Get all information needed for reproducibility"""
        return {
            'rng_seed': self.rng_seed,
            'seed_log': self.seed_log,
            'numpy_version': np.__version__
        }
