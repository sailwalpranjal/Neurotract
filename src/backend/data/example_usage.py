#!/usr/bin/env python
"""
Example Usage of NeuroTract Data Ingestion Modules

Demonstrates how to use DataLoader, BIDSHandler, and MetadataExtractor
for comprehensive neuroimaging data processing.
"""

import logging
from pathlib import Path
import json

from data_loader import DataLoader, setup_logging as setup_loader_logging
from bids_handler import BIDSDataset, BIDSLayout, setup_logging as setup_bids_logging
from metadata_extractor import MetadataExtractor, setup_logging as setup_metadata_logging


def example_load_diffusion_data():
    """Example: Load diffusion MRI data"""
    print("=" * 60)
    print("Example 1: Loading Diffusion MRI Data")
    print("=" * 60)

    loader = DataLoader(use_mmap=True, validate=True)

    # Example paths (adjust to your data)
    nifti_path = "datasets/Stanford dataset/SUB1_b2000_150dirs_2mm.nii.gz"
    bval_path = "datasets/Stanford dataset/SUB1_b2000_150dirs_2mm.bval"
    bvec_path = "datasets/Stanford dataset/SUB1_b2000_150dirs_2mm.bvec"

    try:
        # Load diffusion data
        dwi_data = loader.load_diffusion(
            nifti_path=nifti_path,
            bval_path=bval_path,
            bvec_path=bvec_path
        )

        print(f"Successfully loaded diffusion data!")
        print(f"  Shape: {dwi_data.shape}")
        print(f"  Voxel size: {dwi_data.voxel_size} mm")
        print(f"  Number of volumes: {dwi_data.num_volumes}")
        print(f"  Number of shells: {dwi_data.num_shells}")
        print(f"  B-value shells: {dwi_data.get_shells()}")
        print(f"  File path: {dwi_data.file_path}")
        print()

        # Access metadata
        print("Metadata:")
        for key, value in dwi_data.metadata.items():
            print(f"  {key}: {value}")
        print()

    except Exception as e:
        print(f"Error loading diffusion data: {e}")
        print("(This is expected if the example data doesn't exist)")
    print()


def example_load_anatomical_data():
    """Example: Load anatomical MRI data"""
    print("=" * 60)
    print("Example 2: Loading Anatomical MRI Data")
    print("=" * 60)

    loader = DataLoader(use_mmap=True, validate=True)

    # Example path (adjust to your data)
    nifti_path = "datasets/Stanford dataset/SUB1_t1.nii.gz"

    try:
        # Load anatomical data
        anat_data = loader.load_anatomical(
            nifti_path=nifti_path,
            modality="T1w"
        )

        print(f"Successfully loaded anatomical data!")
        print(f"  Shape: {anat_data.shape}")
        print(f"  Voxel size: {anat_data.voxel_size} mm")
        print(f"  Modality: {anat_data.modality}")
        print(f"  File path: {anat_data.file_path}")
        print()

    except Exception as e:
        print(f"Error loading anatomical data: {e}")
        print("(This is expected if the example data doesn't exist)")
    print()


def example_bids_dataset():
    """Example: Work with BIDS dataset"""
    print("=" * 60)
    print("Example 3: BIDS Dataset Handling")
    print("=" * 60)

    # Example path (adjust to your data)
    bids_path = "datasets"

    try:
        # Initialize BIDS dataset
        dataset = BIDSDataset(bids_path, validate=True)

        # Get dataset information
        info = dataset.get_dataset_info()
        print(f"Dataset: {info['name']}")
        print(f"BIDS version: {info['bids_version']}")
        print(f"Number of subjects: {info['num_subjects']}")
        print(f"Has sessions: {info['has_sessions']}")
        print(f"Available modalities: {info['modalities']}")
        print(f"Has derivatives: {info['has_derivatives']}")
        print()

        # List subjects
        subjects = dataset.get_subjects()
        print(f"First 5 subjects: {subjects[:5]}")
        print()

        # Get subjects with diffusion data
        dwi_subjects = dataset.get_subjects(has_modality='dwi')
        print(f"Subjects with DWI data: {len(dwi_subjects)}")
        if dwi_subjects:
            print(f"  Examples: {dwi_subjects[:3]}")
        print()

        # Get diffusion files for first subject
        if subjects:
            subject = subjects[0]
            sessions = dataset.get_sessions(subject)
            session = sessions[0] if sessions else None

            dwi_files = dataset.get_diffusion_files(subject, session)
            if dwi_files:
                print(f"Diffusion files for {subject}:")
                for key, path in dwi_files.items():
                    print(f"  {key}: {path.name}")
            print()

        # Validate completeness
        completeness = dataset.validate_completeness(required_modalities=['dwi'])
        print(f"Complete subjects: {len(completeness['complete'])}")
        print(f"Incomplete subjects: {len(completeness['incomplete'])}")
        print()

    except Exception as e:
        print(f"Error with BIDS dataset: {e}")
        print("(This is expected if BIDS data doesn't exist)")
    print()


def example_bids_layout():
    """Example: Use BIDSLayout for queries"""
    print("=" * 60)
    print("Example 4: BIDS Layout Queries")
    print("=" * 60)

    bids_path = "datasets"

    try:
        layout = BIDSLayout(bids_path)

        # Query all diffusion files
        dwi_files = layout.get(modality='dwi', suffix='dwi', extension='.nii.gz')
        print(f"Found {len(dwi_files)} diffusion files")
        if dwi_files:
            print(f"  Example: {dwi_files[0].name}")
        print()

        # Query anatomical files
        anat_files = layout.get(modality='anat', suffix='T1w', extension='.nii.gz')
        print(f"Found {len(anat_files)} T1w files")
        if anat_files:
            print(f"  Example: {anat_files[0].name}")
        print()

    except Exception as e:
        print(f"Error with BIDS layout: {e}")
        print("(This is expected if BIDS data doesn't exist)")
    print()


def example_extract_metadata():
    """Example: Extract metadata from NIfTI file"""
    print("=" * 60)
    print("Example 5: Metadata Extraction")
    print("=" * 60)

    extractor = MetadataExtractor()

    # Example path
    nifti_path = "datasets/Stanford dataset/SUB1_b2000_150dirs_2mm.nii.gz"

    try:
        # Extract basic NIfTI metadata
        metadata = extractor.extract_nifti_metadata(nifti_path, load_data=False)

        print("Extracted metadata:")
        print(f"  File: {metadata['file_name']}")
        print(f"  Size: {metadata['file_size_mb']:.2f} MB")
        print(f"  Dimensions: {metadata['dims']}")
        print(f"  Voxel size: {metadata['voxel_size']}")
        print(f"  Orientation: {metadata['orientation']}")
        print(f"  Data type: {metadata['datatype']}")
        print()

    except Exception as e:
        print(f"Error extracting metadata: {e}")
        print("(This is expected if the example data doesn't exist)")
    print()


def example_extract_diffusion_metadata():
    """Example: Extract comprehensive diffusion metadata"""
    print("=" * 60)
    print("Example 6: Diffusion Metadata Extraction")
    print("=" * 60)

    extractor = MetadataExtractor()

    # Example paths
    nifti_path = "datasets/Stanford dataset/SUB1_b2000_150dirs_2mm.nii.gz"
    bval_path = "datasets/Stanford dataset/SUB1_b2000_150dirs_2mm.bval"
    bvec_path = "datasets/Stanford dataset/SUB1_b2000_150dirs_2mm.bvec"

    try:
        # Extract diffusion metadata
        metadata = extractor.extract_diffusion_metadata(
            nifti_path=nifti_path,
            bval_path=bval_path,
            bvec_path=bvec_path
        )

        print("Diffusion metadata:")
        print(f"  Modality: {metadata['modality']}")
        print(f"  Dimensions: {metadata['dims']}")
        print(f"  B-value shells: {metadata['bvals']['num_shells']}")
        print(f"  Shell details:")
        for shell in metadata['bvals']['shells']:
            print(f"    {shell['name']}: {shell['num_directions']} directions")
        print(f"  Gradients normalized: {metadata['bvecs']['normalized']}")
        print()

    except Exception as e:
        print(f"Error extracting diffusion metadata: {e}")
        print("(This is expected if the example data doesn't exist)")
    print()


def example_quality_validation():
    """Example: Validate data quality"""
    print("=" * 60)
    print("Example 7: Data Quality Validation")
    print("=" * 60)

    extractor = MetadataExtractor()

    # Example paths
    nifti_path = "datasets/Stanford dataset/SUB1_b2000_150dirs_2mm.nii.gz"
    bval_path = "datasets/Stanford dataset/SUB1_b2000_150dirs_2mm.bval"
    bvec_path = "datasets/Stanford dataset/SUB1_b2000_150dirs_2mm.bvec"

    try:
        # Validate quality
        quality_checks = extractor.validate_data_quality(
            nifti_path=nifti_path,
            bval_path=bval_path,
            bvec_path=bvec_path,
            strict=False
        )

        print("Quality validation results:")
        print(f"  Passed: {len(quality_checks['passed'])} checks")
        print(f"  Warnings: {len(quality_checks['warnings'])} issues")
        print(f"  Failures: {len(quality_checks['failures'])} critical issues")
        print()

        if quality_checks['passed']:
            print("Passed checks:")
            for check in quality_checks['passed'][:5]:
                print(f"  ✓ {check}")
            print()

        if quality_checks['warnings']:
            print("Warnings:")
            for warning in quality_checks['warnings']:
                print(f"  ⚠ {warning}")
            print()

        if quality_checks['failures']:
            print("Failures:")
            for failure in quality_checks['failures']:
                print(f"  ✗ {failure}")
            print()

    except Exception as e:
        print(f"Error validating quality: {e}")
        print("(This is expected if the example data doesn't exist)")
    print()


def example_generate_manifest():
    """Example: Generate scan manifest"""
    print("=" * 60)
    print("Example 8: Generate Scan Manifest")
    print("=" * 60)

    extractor = MetadataExtractor()

    # Example paths
    nifti_path = "datasets/Stanford dataset/SUB1_b2000_150dirs_2mm.nii.gz"
    bval_path = "datasets/Stanford dataset/SUB1_b2000_150dirs_2mm.bval"
    bvec_path = "datasets/Stanford dataset/SUB1_b2000_150dirs_2mm.bvec"
    output_path = "scan_manifest.json"

    try:
        # Generate comprehensive manifest
        manifest = extractor.generate_scan_manifest(
            nifti_path=nifti_path,
            bval_path=bval_path,
            bvec_path=bvec_path,
            output_path=output_path,
            validate=True
        )

        print(f"Generated comprehensive scan manifest")
        print(f"  Manifest version: {manifest['manifest_version']}")
        print(f"  Data type: {manifest['data_type']}")
        print(f"  Quality checks: {len(manifest['quality_checks']['passed'])} passed")
        print(f"  Saved to: {output_path}")
        print()

        # Print excerpt
        print("Manifest excerpt:")
        print(json.dumps({
            'data_type': manifest['data_type'],
            'metadata': {
                'file_name': manifest['metadata']['file_name'],
                'dims': manifest['metadata']['dims'],
                'voxel_size': manifest['metadata']['voxel_size'],
                'bvals': manifest['metadata']['bvals']
            }
        }, indent=2))
        print()

    except Exception as e:
        print(f"Error generating manifest: {e}")
        print("(This is expected if the example data doesn't exist)")
    print()


def main():
    """Run all examples"""
    # Setup logging
    setup_loader_logging(logging.INFO)
    setup_bids_logging(logging.WARNING)
    setup_metadata_logging(logging.INFO)

    print("\n")
    print("=" * 60)
    print("NeuroTract Data Ingestion Module Examples")
    print("=" * 60)
    print("\n")

    # Run examples
    example_load_diffusion_data()
    example_load_anatomical_data()
    example_bids_dataset()
    example_bids_layout()
    example_extract_metadata()
    example_extract_diffusion_metadata()
    example_quality_validation()
    example_generate_manifest()

    print("=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nNote: Some examples may show errors if the example data doesn't exist.")
    print("This is expected. The examples demonstrate the API usage.\n")


if __name__ == "__main__":
    main()
