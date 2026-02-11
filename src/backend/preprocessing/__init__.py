"""
Preprocessing Module

Handles motion correction, eddy current correction, brain extraction,
and bias field correction for diffusion MRI data.

Main Components:
- PreprocessingPipeline: Main orchestrator for all preprocessing steps
- MotionCorrector: Motion and eddy current correction
- BrainExtractor: Brain extraction and skull-stripping
- BiasFieldCorrector: Bias field correction
- GradientCorrector: Gradient table validation and correction
"""

__version__ = "0.1.0"

from .pipeline import PreprocessingPipeline, PreprocessingError
from .motion_correction import (
    MotionCorrector,
    MotionMetrics,
    MotionCorrectionError,
    save_motion_corrected_data,
)
from .brain_extraction import (
    BrainExtractor,
    MaskQualityMetrics,
    BrainExtractionError,
    save_brain_mask,
    apply_mask,
)
from .bias_correction import (
    BiasFieldCorrector,
    BiasCorrectionMetrics,
    BiasCorrectionError,
    save_bias_corrected_data,
)
from .gradient_correction import (
    GradientCorrector,
    GradientQualityMetrics,
    GradientCorrectionError,
    reorient_gradients,
)

__all__ = [
    # Pipeline
    "PreprocessingPipeline",
    "PreprocessingError",
    # Motion correction
    "MotionCorrector",
    "MotionMetrics",
    "MotionCorrectionError",
    "save_motion_corrected_data",
    # Brain extraction
    "BrainExtractor",
    "MaskQualityMetrics",
    "BrainExtractionError",
    "save_brain_mask",
    "apply_mask",
    # Bias correction
    "BiasFieldCorrector",
    "BiasCorrectionMetrics",
    "BiasCorrectionError",
    "save_bias_corrected_data",
    # Gradient correction
    "GradientCorrector",
    "GradientQualityMetrics",
    "GradientCorrectionError",
    "reorient_gradients",
]
