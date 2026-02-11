#!/usr/bin/env python
"""
BIDS Handler Module for NeuroTract

Provides comprehensive BIDS (Brain Imaging Data Structure) dataset handling.
Supports dataset navigation, file discovery, and metadata extraction.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import re

logger = logging.getLogger(__name__)


class BIDSError(Exception):
    """Exception raised for BIDS-related errors"""
    pass


class BIDSDataset:
    """
    BIDS dataset handler

    Provides methods to navigate and query BIDS-formatted datasets.
    Supports multiple sessions, runs, and modalities.
    """

    def __init__(self, dataset_path: Union[str, Path], validate: bool = True):
        """
        Initialize BIDS dataset handler

        Parameters
        ----------
        dataset_path : str or Path
            Path to BIDS dataset root directory
        validate : bool
            Validate BIDS structure on initialization
        """
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise BIDSError(f"Dataset path does not exist: {self.dataset_path}")

        self.dataset_description = None
        self.subjects = []
        self.sessions_map = {}  # subject -> list of sessions
        self.derivatives_path = None

        # Load dataset description
        self._load_dataset_description()

        # Scan dataset structure
        self._scan_structure()

        if validate:
            self._validate_structure()

        logger.info(
            f"Initialized BIDS dataset: {self.dataset_description.get('Name', 'Unknown')} "
            f"with {len(self.subjects)} subjects"
        )

    def _load_dataset_description(self):
        """Load dataset_description.json"""
        desc_path = self.dataset_path / "dataset_description.json"

        if desc_path.exists():
            try:
                with open(desc_path, 'r') as f:
                    self.dataset_description = json.load(f)
                logger.debug(f"Loaded dataset description from {desc_path}")
            except Exception as e:
                logger.warning(f"Failed to load dataset_description.json: {e}")
                self.dataset_description = {"Name": "Unknown"}
        else:
            logger.warning("dataset_description.json not found")
            self.dataset_description = {"Name": "Unknown"}

    def _scan_structure(self):
        """Scan BIDS directory structure to find subjects and sessions"""
        # Find all subject directories
        subject_dirs = sorted([
            d for d in self.dataset_path.iterdir()
            if d.is_dir() and d.name.startswith('sub-')
        ])

        self.subjects = [d.name for d in subject_dirs]

        # Find sessions for each subject
        for subject_dir in subject_dirs:
            subject = subject_dir.name
            session_dirs = sorted([
                d for d in subject_dir.iterdir()
                if d.is_dir() and d.name.startswith('ses-')
            ])

            if session_dirs:
                self.sessions_map[subject] = [d.name for d in session_dirs]
            else:
                # No session subdirectories - single session dataset
                self.sessions_map[subject] = [None]

        # Check for derivatives
        derivatives_path = self.dataset_path / "derivatives"
        if derivatives_path.exists() and derivatives_path.is_dir():
            self.derivatives_path = derivatives_path
            logger.debug(f"Found derivatives directory: {derivatives_path}")

    def _validate_structure(self):
        """Validate basic BIDS structure"""
        # Check for required dataset_description.json
        if not (self.dataset_path / "dataset_description.json").exists():
            logger.warning("Missing dataset_description.json (required by BIDS)")

        # Check for README (recommended)
        if not (self.dataset_path / "README").exists():
            logger.debug("No README found (recommended by BIDS)")

        # Validate subject naming
        invalid_subjects = [
            s for s in self.subjects
            if not re.match(r'^sub-[a-zA-Z0-9]+$', s)
        ]
        if invalid_subjects:
            logger.warning(
                f"Invalid subject naming (should be alphanumeric): {invalid_subjects[:5]}"
            )

    def get_subjects(self, has_modality: Optional[str] = None) -> List[str]:
        """
        Get list of subject IDs

        Parameters
        ----------
        has_modality : str, optional
            Filter subjects that have specific modality (dwi, anat, func, fmap)

        Returns
        -------
        list of str
            Subject IDs (e.g., ['sub-01', 'sub-02'])
        """
        if has_modality is None:
            return self.subjects

        # Filter subjects by modality
        filtered_subjects = []
        for subject in self.subjects:
            sessions = self.get_sessions(subject)
            for session in sessions:
                modality_files = self.get_modality_files(subject, session, has_modality)
                if modality_files:
                    filtered_subjects.append(subject)
                    break

        return filtered_subjects

    def get_sessions(self, subject: str) -> List[Optional[str]]:
        """
        Get sessions for a subject

        Parameters
        ----------
        subject : str
            Subject ID (e.g., 'sub-01')

        Returns
        -------
        list of str or None
            Session IDs (e.g., ['ses-01', 'ses-02']) or [None] if no sessions
        """
        return self.sessions_map.get(subject, [])

    def get_modality_files(
        self,
        subject: str,
        session: Optional[str],
        modality: str,
        suffix: Optional[str] = None,
        extension: str = ".nii.gz"
    ) -> List[Path]:
        """
        Get files for a specific modality

        Parameters
        ----------
        subject : str
            Subject ID (e.g., 'sub-01')
        session : str or None
            Session ID (e.g., 'ses-01') or None
        modality : str
            Modality directory name (dwi, anat, func, fmap)
        suffix : str, optional
            BIDS suffix to filter (e.g., 'dwi', 'T1w', 'bold')
        extension : str
            File extension (default: '.nii.gz')

        Returns
        -------
        list of Path
            List of file paths matching criteria
        """
        # Construct path to modality directory
        subject_dir = self.dataset_path / subject
        if session:
            modality_dir = subject_dir / session / modality
        else:
            modality_dir = subject_dir / modality

        if not modality_dir.exists():
            return []

        # Find files
        pattern = f"*{extension}"
        files = sorted(modality_dir.glob(pattern))

        # Filter by suffix if specified
        if suffix:
            files = [f for f in files if f.name.endswith(f"_{suffix}{extension}")]

        return files

    def get_diffusion_files(
        self,
        subject: str,
        session: Optional[str] = None,
        run: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Get diffusion MRI files (nifti, bval, bvec) for a subject/session

        Parameters
        ----------
        subject : str
            Subject ID
        session : str, optional
            Session ID
        run : str, optional
            Run ID (e.g., 'run-01')

        Returns
        -------
        dict
            Dictionary with keys 'nifti', 'bval', 'bvec', 'json' (if available)
        """
        # Get DWI directory
        subject_dir = self.dataset_path / subject
        if session:
            dwi_dir = subject_dir / session / "dwi"
        else:
            dwi_dir = subject_dir / "dwi"

        if not dwi_dir.exists():
            return {}

        # Build filename pattern
        parts = [subject]
        if session:
            parts.append(session)
        if run:
            parts.append(run)
        parts.append("dwi")

        base_name = "_".join(parts)

        # Find files
        result = {}

        nifti_path = dwi_dir / f"{base_name}.nii.gz"
        if not nifti_path.exists():
            nifti_path = dwi_dir / f"{base_name}.nii"

        if nifti_path.exists():
            result['nifti'] = nifti_path

        bval_path = dwi_dir / f"{base_name}.bval"
        if bval_path.exists():
            result['bval'] = bval_path

        bvec_path = dwi_dir / f"{base_name}.bvec"
        if bvec_path.exists():
            result['bvec'] = bvec_path

        json_path = dwi_dir / f"{base_name}.json"
        if json_path.exists():
            result['json'] = json_path

        return result

    def get_anatomical_files(
        self,
        subject: str,
        session: Optional[str] = None,
        modality: str = "T1w",
        run: Optional[str] = None
    ) -> List[Path]:
        """
        Get anatomical MRI files for a subject/session

        Parameters
        ----------
        subject : str
            Subject ID
        session : str, optional
            Session ID
        modality : str
            Anatomical modality (T1w, T2w, FLAIR, etc.)
        run : str, optional
            Run ID

        Returns
        -------
        list of Path
            List of anatomical file paths
        """
        # Get anat directory
        subject_dir = self.dataset_path / subject
        if session:
            anat_dir = subject_dir / session / "anat"
        else:
            anat_dir = subject_dir / "anat"

        if not anat_dir.exists():
            return []

        # Build pattern
        parts = [subject]
        if session:
            parts.append(session)
        if run:
            parts.append(run)
        parts.append(modality)

        pattern = "_".join(parts) + ".nii*"

        return sorted(anat_dir.glob(pattern))

    def get_surface_files(
        self,
        subject: str,
        session: Optional[str] = None,
        space: str = "fsnative"
    ) -> Dict[str, List[Path]]:
        """
        Get surface files from derivatives (e.g., FreeSurfer)

        Parameters
        ----------
        subject : str
            Subject ID
        session : str, optional
            Session ID
        space : str
            Surface space (fsnative, fsaverage, etc.)

        Returns
        -------
        dict
            Dictionary with hemispheres as keys ('lh', 'rh') and file lists as values
        """
        if self.derivatives_path is None:
            return {}

        # Look in freesurfer derivative
        freesurfer_path = self.derivatives_path / "freesurfer" / subject

        if not freesurfer_path.exists():
            # Try alternative locations
            freesurfer_path = self.derivatives_path / subject

        if not freesurfer_path.exists():
            return {}

        # Find surface files
        surf_dir = freesurfer_path / "surf"
        if not surf_dir.exists():
            return {}

        result = {'lh': [], 'rh': []}

        # Common surface types
        surf_types = ['white', 'pial', 'inflated', 'sphere']

        for hemi in ['lh', 'rh']:
            for surf_type in surf_types:
                surf_file = surf_dir / f"{hemi}.{surf_type}"
                if surf_file.exists():
                    result[hemi].append(surf_file)

        return result

    def get_metadata(
        self,
        subject: str,
        session: Optional[str],
        modality: str,
        suffix: str
    ) -> Optional[Dict]:
        """
        Load JSON sidecar metadata for a BIDS file

        Parameters
        ----------
        subject : str
            Subject ID
        session : str or None
            Session ID
        modality : str
            Modality (dwi, anat, func, fmap)
        suffix : str
            BIDS suffix (dwi, T1w, bold, etc.)

        Returns
        -------
        dict or None
            Metadata dictionary or None if not found
        """
        # Construct path to modality directory
        subject_dir = self.dataset_path / subject
        if session:
            modality_dir = subject_dir / session / modality
        else:
            modality_dir = subject_dir / modality

        if not modality_dir.exists():
            return None

        # Build filename pattern
        parts = [subject]
        if session:
            parts.append(session)
        parts.append(suffix)

        json_name = "_".join(parts) + ".json"
        json_path = modality_dir / json_name

        if not json_path.exists():
            return None

        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata from {json_path}: {e}")
            return None

    def find_derivatives(self, pipeline_name: Optional[str] = None) -> List[Path]:
        """
        Find derivative directories

        Parameters
        ----------
        pipeline_name : str, optional
            Filter by pipeline name (e.g., 'fmriprep', 'freesurfer')

        Returns
        -------
        list of Path
            List of derivative directories
        """
        if self.derivatives_path is None:
            return []

        if pipeline_name:
            pipeline_path = self.derivatives_path / pipeline_name
            return [pipeline_path] if pipeline_path.exists() else []

        # Return all derivative pipelines
        return sorted([
            d for d in self.derivatives_path.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

    def get_dataset_info(self) -> Dict:
        """
        Get comprehensive dataset information

        Returns
        -------
        dict
            Dataset information including subjects, sessions, modalities
        """
        info = {
            'name': self.dataset_description.get('Name', 'Unknown'),
            'bids_version': self.dataset_description.get('BIDSVersion', 'Unknown'),
            'num_subjects': len(self.subjects),
            'subjects': self.subjects,
            'has_sessions': any(s[0] is not None for s in self.sessions_map.values()),
            'modalities': set(),
            'has_derivatives': self.derivatives_path is not None
        }

        # Scan for available modalities
        modality_dirs = ['dwi', 'anat', 'func', 'fmap']
        for subject in self.subjects[:10]:  # Sample first 10 subjects
            sessions = self.get_sessions(subject)
            for session in sessions:
                subject_dir = self.dataset_path / subject
                if session:
                    base_dir = subject_dir / session
                else:
                    base_dir = subject_dir

                for modality in modality_dirs:
                    modality_path = base_dir / modality
                    if modality_path.exists():
                        info['modalities'].add(modality)

        info['modalities'] = sorted(list(info['modalities']))

        return info

    def validate_completeness(
        self,
        required_modalities: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Validate dataset completeness

        Parameters
        ----------
        required_modalities : list of str, optional
            List of required modalities (e.g., ['dwi', 'anat'])

        Returns
        -------
        dict
            Dictionary with 'complete' and 'incomplete' subject lists
        """
        if required_modalities is None:
            required_modalities = ['dwi']

        complete_subjects = []
        incomplete_subjects = []

        for subject in self.subjects:
            sessions = self.get_sessions(subject)
            has_all_modalities = True

            for session in sessions:
                for modality in required_modalities:
                    files = self.get_modality_files(subject, session, modality)
                    if not files:
                        has_all_modalities = False
                        break

                if has_all_modalities:
                    break

            if has_all_modalities:
                complete_subjects.append(subject)
            else:
                incomplete_subjects.append(subject)

        return {
            'complete': complete_subjects,
            'incomplete': incomplete_subjects
        }

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"BIDSDataset(name='{self.dataset_description.get('Name', 'Unknown')}', "
            f"subjects={len(self.subjects)}, path='{self.dataset_path}')"
        )


class BIDSLayout:
    """
    Simplified BIDS layout manager

    Provides high-level interface for querying BIDS datasets.
    """

    def __init__(self, dataset_path: Union[str, Path]):
        """
        Initialize BIDS layout

        Parameters
        ----------
        dataset_path : str or Path
            Path to BIDS dataset root
        """
        self.dataset = BIDSDataset(dataset_path)

    def get(
        self,
        subject: Optional[str] = None,
        session: Optional[str] = None,
        modality: Optional[str] = None,
        suffix: Optional[str] = None,
        extension: str = ".nii.gz",
        return_type: str = "file"
    ) -> List[Union[Path, str]]:
        """
        Query BIDS dataset with flexible filters

        Parameters
        ----------
        subject : str, optional
            Subject ID filter
        session : str, optional
            Session ID filter
        modality : str, optional
            Modality filter (dwi, anat, func, fmap)
        suffix : str, optional
            BIDS suffix filter
        extension : str
            File extension
        return_type : str
            'file' returns Path objects, 'filename' returns strings

        Returns
        -------
        list
            List of matching files
        """
        results = []

        # Determine subjects to search
        subjects = [subject] if subject else self.dataset.get_subjects()

        for subj in subjects:
            # Determine sessions to search
            if session:
                sessions = [session]
            else:
                sessions = self.dataset.get_sessions(subj)

            for sess in sessions:
                if modality:
                    files = self.dataset.get_modality_files(
                        subj, sess, modality, suffix, extension
                    )
                    results.extend(files)

        if return_type == "filename":
            results = [str(f) for f in results]

        return results


def setup_logging(level: int = logging.INFO):
    """Configure logging for BIDS handler"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


if __name__ == "__main__":
    # Example usage
    setup_logging(logging.DEBUG)

    # Example: Initialize BIDS dataset
    # dataset = BIDSDataset("path/to/bids/dataset")
    # print(dataset.get_dataset_info())

    # Example: Get diffusion files
    # dwi_files = dataset.get_diffusion_files("sub-01", "ses-01")
    # print(dwi_files)

    print("BIDSHandler module ready")
