"""
Brain Extraction Module

Implements robust brain extraction/skull-stripping for diffusion MRI data:
1. DIPY median_otsu (primary method) - optimized for DWI
2. Threshold-based extraction (fallback) - simple and fast
3. Morphological operations for mask refinement
4. Quality validation and hole filling

Produces clean brain masks suitable for tractography and analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import nibabel as nib
from scipy import ndimage
from dipy.segment.mask import median_otsu

from ..utils.logger import get_logger, log_decision
from ..utils.memory_manager import get_memory_manager

logger = get_logger(__name__)


class BrainExtractionError(Exception):
    """Exception raised for brain extraction failures"""
    pass


class MaskQualityMetrics:
    """Container for brain mask quality metrics"""

    def __init__(self, mask: np.ndarray, original_data: np.ndarray):
        self.mask = mask
        self.original_data = original_data
        self.metrics = self._compute_metrics()

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute quality metrics for brain mask"""
        total_voxels = np.prod(self.mask.shape)
        brain_voxels = np.sum(self.mask)
        brain_fraction = brain_voxels / total_voxels

        # Compute signal statistics inside and outside mask
        inside_values = self.original_data[self.mask > 0]
        outside_values = self.original_data[self.mask == 0]

        metrics = {
            'total_voxels': int(total_voxels),
            'brain_voxels': int(brain_voxels),
            'brain_fraction': float(brain_fraction),
            'mean_signal_inside': float(np.mean(inside_values)) if len(inside_values) > 0 else 0.0,
            'mean_signal_outside': float(np.mean(outside_values)) if len(outside_values) > 0 else 0.0,
            'std_signal_inside': float(np.std(inside_values)) if len(inside_values) > 0 else 0.0,
            'std_signal_outside': float(np.std(outside_values)) if len(outside_values) > 0 else 0.0,
        }

        # Compute contrast-to-noise ratio
        if metrics['std_signal_outside'] > 0:
            metrics['cnr'] = (
                (metrics['mean_signal_inside'] - metrics['mean_signal_outside']) /
                metrics['std_signal_outside']
            )
        else:
            metrics['cnr'] = 0.0

        # Count connected components
        labeled, num_components = ndimage.label(self.mask)
        metrics['num_components'] = int(num_components)

        # Find largest component size
        if num_components > 0:
            component_sizes = [
                np.sum(labeled == i) for i in range(1, num_components + 1)
            ]
            metrics['largest_component_fraction'] = float(
                max(component_sizes) / brain_voxels if brain_voxels > 0 else 0
            )
        else:
            metrics['largest_component_fraction'] = 0.0

        return metrics

    def is_valid(
        self,
        min_brain_fraction: float = 0.05,
        max_brain_fraction: float = 0.95,
        min_cnr: float = 2.0,
    ) -> Tuple[bool, str]:
        """
        Validate mask quality

        Parameters
        ----------
        min_brain_fraction : float
            Minimum acceptable brain fraction
        max_brain_fraction : float
            Maximum acceptable brain fraction
        min_cnr : float
            Minimum contrast-to-noise ratio

        Returns
        -------
        is_valid : bool
            Whether mask passes quality checks
        message : str
            Validation message
        """
        issues = []

        if self.metrics['brain_fraction'] < min_brain_fraction:
            issues.append(
                f"Brain fraction too low: {self.metrics['brain_fraction']:.3f} "
                f"< {min_brain_fraction}"
            )

        if self.metrics['brain_fraction'] > max_brain_fraction:
            issues.append(
                f"Brain fraction too high: {self.metrics['brain_fraction']:.3f} "
                f"> {max_brain_fraction}"
            )

        if self.metrics['cnr'] < min_cnr:
            issues.append(
                f"Contrast-to-noise ratio too low: {self.metrics['cnr']:.3f} "
                f"< {min_cnr}"
            )

        if self.metrics['num_components'] > 5:
            issues.append(
                f"Too many components: {self.metrics['num_components']} > 5"
            )

        if self.metrics['largest_component_fraction'] < 0.8:
            issues.append(
                f"Largest component too small: "
                f"{self.metrics['largest_component_fraction']:.3f} < 0.8"
            )

        if issues:
            return False, "; ".join(issues)
        else:
            return True, "Mask quality acceptable"

    def save(self, output_path: Path):
        """Save quality metrics to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("Brain Mask Quality Metrics\n")
            f.write("=" * 50 + "\n\n")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value}\n")

            f.write("\n")
            is_valid, message = self.is_valid()
            f.write(f"Validation: {'PASS' if is_valid else 'FAIL'}\n")
            f.write(f"Message: {message}\n")

        logger.info(f"Mask quality metrics saved to {output_path}")


class BrainExtractor:
    """
    Robust brain extraction for diffusion MRI data

    Implements multiple methods with automatic fallback and quality validation
    """

    def __init__(
        self,
        method: str = "median_otsu",
        median_radius: int = 4,
        num_pass: int = 4,
        autocrop: bool = False,
        dilate: Optional[int] = None,
    ):
        """
        Initialize brain extractor

        Parameters
        ----------
        method : str
            Extraction method: 'median_otsu' or 'threshold' (default: 'median_otsu')
        median_radius : int
            Radius for median filter in median_otsu (default: 4)
        num_pass : int
            Number of passes for median_otsu (default: 4)
        autocrop : bool
            Automatically crop to brain bounding box (default: False)
        dilate : int, optional
            Number of dilation iterations to expand mask (None = no dilation)
        """
        self.method = method
        self.median_radius = median_radius
        self.num_pass = num_pass
        self.autocrop = autocrop
        self.dilate = dilate

        self.memory_manager = get_memory_manager()

        logger.info(
            f"BrainExtractor initialized: method={method}, "
            f"median_radius={median_radius}, num_pass={num_pass}"
        )

    def extract(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        vol_idx: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray, MaskQualityMetrics]:
        """
        Extract brain mask from DWI data

        Parameters
        ----------
        data : np.ndarray
            3D or 4D image data
        affine : np.ndarray
            4x4 affine matrix
        vol_idx : int, optional
            Volume index to use for 4D data (None = mean of all volumes)
        output_dir : Path, optional
            Directory to save QC outputs

        Returns
        -------
        masked_data : np.ndarray
            Brain-extracted data (same shape as input)
        mask : np.ndarray
            Binary brain mask (3D)
        metrics : MaskQualityMetrics
            Quality metrics for the mask
        """
        logger.info("Starting brain extraction")

        # Handle 4D data
        if data.ndim == 4:
            if vol_idx is not None:
                reference = data[..., vol_idx]
                logger.info(f"Using volume {vol_idx} for brain extraction")
            else:
                logger.info("Computing mean volume for brain extraction")
                reference = np.mean(data, axis=-1)
        elif data.ndim == 3:
            reference = data
        else:
            raise BrainExtractionError(
                f"Expected 3D or 4D data, got {data.ndim}D: {data.shape}"
            )

        # Log decision about method
        log_decision(
            decision_id="brain_extraction_method",
            component="preprocessing.brain_extraction",
            decision=f"Using {self.method} method for brain extraction",
            rationale=(
                f"Method selected based on initialization. "
                f"median_otsu is optimized for DWI data."
            ),
            parameters={
                'method': self.method,
                'median_radius': self.median_radius,
                'num_pass': self.num_pass,
                'autocrop': self.autocrop,
                'dilate': self.dilate,
            },
        )

        # Extract mask using selected method
        try:
            if self.method == "median_otsu":
                mask = self._extract_median_otsu(reference)
            elif self.method == "threshold":
                mask = self._extract_threshold(reference)
            else:
                raise BrainExtractionError(f"Unknown method: {self.method}")

        except Exception as e:
            logger.warning(f"Primary method {self.method} failed: {e}")
            logger.info("Falling back to threshold method")
            mask = self._extract_threshold(reference)

        # Refine mask
        mask = self._refine_mask(mask)

        # Apply mask to data
        if data.ndim == 4:
            masked_data = data * mask[..., np.newaxis]
        else:
            masked_data = data * mask

        # Compute quality metrics
        metrics = MaskQualityMetrics(mask, reference)

        # Validate mask quality
        is_valid, message = metrics.is_valid()
        if is_valid:
            logger.info(f"Brain mask quality: {message}")
        else:
            logger.warning(f"Brain mask quality issues: {message}")

        # Save outputs if directory provided
        if output_dir is not None:
            self._save_outputs(mask, metrics, output_dir)

        logger.info(
            f"Brain extraction completed: "
            f"brain_fraction={metrics.metrics['brain_fraction']:.3f}, "
            f"cnr={metrics.metrics['cnr']:.2f}"
        )

        return masked_data, mask, metrics

    def _extract_median_otsu(self, data: np.ndarray) -> np.ndarray:
        """
        Extract brain mask using DIPY's median_otsu method

        Optimized for diffusion-weighted images
        """
        logger.info(
            f"Running median_otsu: radius={self.median_radius}, "
            f"num_pass={self.num_pass}"
        )

        # DIPY's median_otsu expects 3D or 4D data
        # For 4D data, we need to specify which volumes to use
        if data.ndim == 3:
            data_for_otsu = data[..., np.newaxis]
            vol_idx = [0]  # Specify the single volume
        else:
            data_for_otsu = data
            # For 4D diffusion data, use first 10 volumes (typically includes b0)
            vol_idx = range(min(10, data.shape[-1]))

        try:
            _, mask = median_otsu(
                data_for_otsu,
                median_radius=self.median_radius,
                numpass=self.num_pass,
                autocrop=self.autocrop,
                vol_idx=vol_idx,
                dilate=self.dilate,
            )

            # Ensure 3D mask
            if mask.ndim == 4:
                mask = mask[..., 0]

        except Exception as e:
            raise BrainExtractionError(f"median_otsu failed: {e}")

        return mask.astype(np.uint8)

    def _extract_threshold(
        self,
        data: np.ndarray,
        threshold_method: str = "otsu"
    ) -> np.ndarray:
        """
        Extract brain mask using simple thresholding

        Fallback method when median_otsu is unavailable
        """
        logger.info(f"Running threshold-based extraction: method={threshold_method}")

        if threshold_method == "otsu":
            threshold = self._compute_otsu_threshold(data)
        else:
            # Use percentile-based threshold
            threshold = np.percentile(data[data > 0], 25)

        logger.info(f"Threshold value: {threshold:.2f}")

        # Create initial mask
        mask = data > threshold

        # Remove small objects
        mask = ndimage.binary_opening(mask, iterations=2)
        mask = ndimage.binary_closing(mask, iterations=2)

        return mask.astype(np.uint8)

    def _compute_otsu_threshold(self, data: np.ndarray) -> float:
        """
        Compute Otsu's threshold for automatic thresholding

        Parameters
        ----------
        data : np.ndarray
            Input image data

        Returns
        -------
        threshold : float
            Optimal threshold value
        """
        # Flatten and remove zeros
        values = data.flatten()
        values = values[values > 0]

        if len(values) == 0:
            return 0.0

        # Compute histogram
        hist, bin_edges = np.histogram(values, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize histogram
        hist = hist.astype(float)
        hist /= hist.sum()

        # Compute cumulative sums
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]

        # Compute cumulative means
        mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-10)
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / (weight2 + 1e-10))[::-1]

        # Compute between-class variance
        variance = weight1 * weight2 * (mean1 - mean2) ** 2

        # Find threshold that maximizes variance
        idx = np.argmax(variance)
        threshold = bin_centers[idx]

        return threshold

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine brain mask using morphological operations

        - Fill holes
        - Remove small components
        - Keep only largest connected component
        """
        logger.info("Refining brain mask")

        # Fill holes
        mask = ndimage.binary_fill_holes(mask)

        # Remove small components and keep largest
        labeled, num_components = ndimage.label(mask)

        if num_components > 1:
            logger.info(f"Found {num_components} components, keeping largest")
            component_sizes = [
                (i, np.sum(labeled == i)) for i in range(1, num_components + 1)
            ]
            largest_component = max(component_sizes, key=lambda x: x[1])[0]
            mask = (labeled == largest_component)

        # Smooth edges
        mask = ndimage.binary_erosion(mask, iterations=1)
        mask = ndimage.binary_dilation(mask, iterations=2)
        mask = ndimage.binary_erosion(mask, iterations=1)

        # Additional dilation if requested
        if self.dilate is not None and self.dilate > 0:
            logger.info(f"Dilating mask by {self.dilate} iterations")
            mask = ndimage.binary_dilation(mask, iterations=self.dilate)

        # Fill any remaining holes
        mask = ndimage.binary_fill_holes(mask)

        return mask.astype(np.uint8)

    def _save_outputs(
        self,
        mask: np.ndarray,
        metrics: MaskQualityMetrics,
        output_dir: Path,
    ):
        """Save mask and quality metrics"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save quality metrics
        metrics.save(output_dir / "brain_mask_qc.txt")

        logger.info(f"Brain extraction outputs saved to {output_dir}")


def save_brain_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    output_path: Path,
):
    """
    Save brain mask to NIfTI file

    Parameters
    ----------
    mask : np.ndarray
        Binary brain mask
    affine : np.ndarray
        4x4 affine matrix
    header : nib.Nifti1Header
        NIfTI header
    output_path : Path
        Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure mask is uint8
    mask = mask.astype(np.uint8)

    # Create NIfTI image
    mask_img = nib.Nifti1Image(mask, affine, header)

    # Update header description
    mask_img.header['descrip'] = b'Brain mask generated by NeuroTract'

    # Save
    nib.save(mask_img, output_path)
    logger.info(f"Saved brain mask to {output_path}")


def apply_mask(
    data: np.ndarray,
    mask: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Apply brain mask to data

    Parameters
    ----------
    data : np.ndarray
        Input data (3D or 4D)
    mask : np.ndarray
        Binary brain mask (3D)
    fill_value : float
        Value to use outside mask (default: 0.0)

    Returns
    -------
    masked_data : np.ndarray
        Masked data (same shape as input)
    """
    if data.ndim == 4:
        mask_4d = mask[..., np.newaxis]
        masked_data = np.where(mask_4d, data, fill_value)
    else:
        masked_data = np.where(mask, data, fill_value)

    return masked_data
