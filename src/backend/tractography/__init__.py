"""
Tractography Module

Probabilistic tractography using Runge-Kutta integration and Monte Carlo sampling.

Main components:
- RK4Integrator: Adaptive 4th-order Runge-Kutta integration with termination conditions
- SeedGenerator: Flexible seed generation strategies with deterministic RNG
- ProbabilisticTracker: Monte Carlo tractography engine with streaming to disk
- StreamlineUtils: Filtering, smoothing, clustering, and format conversion
- TractographyQC: Comprehensive quality control and validation
"""

from .rk4_integrator import RK4Integrator
from .seeding import SeedGenerator
from .probabilistic_tracker import ProbabilisticTracker
from .streamline_utils import StreamlineUtils
from .quality_control import TractographyQC

__version__ = "0.1.0"

__all__ = [
    'RK4Integrator',
    'SeedGenerator',
    'ProbabilisticTracker',
    'StreamlineUtils',
    'TractographyQC'
]
