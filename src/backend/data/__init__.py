"""
NeuroTract Data Module

This module handles all data ingestion, validation, and preprocessing operations.
Provides comprehensive tools for loading, parsing, and validating neuroimaging data.
"""

__version__ = "0.1.0"

from .data_loader import (
    DataLoader,
    DiffusionData,
    AnatomicalData,
    NIfTILoadError,
    DataValidationError
)

from .bids_handler import (
    BIDSDataset,
    BIDSLayout,
    BIDSError
)

from .metadata_extractor import (
    MetadataExtractor,
    MetadataExtractionError,
    QualityCheckError
)

from .dataset_analyzer import DatasetAnalyzer

__all__ = [
    # Data Loader
    'DataLoader',
    'DiffusionData',
    'AnatomicalData',
    'NIfTILoadError',
    'DataValidationError',

    # BIDS Handler
    'BIDSDataset',
    'BIDSLayout',
    'BIDSError',

    # Metadata Extractor
    'MetadataExtractor',
    'MetadataExtractionError',
    'QualityCheckError',

    # Dataset Analyzer
    'DatasetAnalyzer',
]
