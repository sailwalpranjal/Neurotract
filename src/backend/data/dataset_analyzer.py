#!/usr/bin/env python
"""
Dataset Analyzer for NeuroTract

Scans and analyzes available datasets, validates structure, and generates reports.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib


class DatasetAnalyzer:
    """Analyzes neuroimaging datasets and generates structured reports"""

    def __init__(self, dataset_root: str = "datasets"):
        self.dataset_root = Path(dataset_root)
        self.report = {
            "total_subjects": 0,
            "datasets": {},
            "diffusion_subjects": [],
            "anatomical_subjects": [],
            "total_size_bytes": 0,
            "issues": []
        }

    def scan_all(self) -> Dict:
        """Scan all datasets and generate comprehensive report"""
        print("="*60)
        print("NeuroTract Dataset Analyzer")
        print("="*60)
        print()

        # Check Stanford dataset
        stanford_path = self.dataset_root / "Stanford dataset"
        if stanford_path.exists():
            print("[Stanford] Analyzing Stanford dataset...")
            self.analyze_stanford(stanford_path)

        # Check BIDS datasets (OpenNeuro)
        bids_subjects = list(self.dataset_root.glob("sub-*"))
        if bids_subjects:
            print(f"[BIDS] Analyzing BIDS dataset ({len(bids_subjects)} subjects)...")
            self.analyze_bids(self.dataset_root)

        # Calculate total size
        self.report["total_size_bytes"] = self._get_directory_size(self.dataset_root)
        self.report["total_size_gb"] = round(
            self.report["total_size_bytes"] / (1024**3), 2
        )

        self._print_summary()
        return self.report

    def analyze_stanford(self, path: Path):
        """Analyze Stanford HARDI dataset structure"""
        dataset_info = {
            "name": "Stanford HARDI",
            "type": "multi-shell diffusion",
            "subjects": [],
            "has_diffusion": False,
            "has_anatomical": False,
            "has_bvals": False,
            "has_bvecs": False,
            "shells": []
        }

        # Find subjects
        nii_files = list(path.glob("*.nii.gz"))
        subjects = set()
        for f in nii_files:
            if "SUB" in f.name:
                sub = f.name.split("_")[0]
                subjects.add(sub)

        dataset_info["subjects"] = sorted(list(subjects))

        # Check for diffusion shells
        for sub in subjects:
            b1000 = list(path.glob(f"{sub}_b1000*.nii.gz"))
            b2000 = list(path.glob(f"{sub}_b2000*.nii.gz"))
            b4000 = list(path.glob(f"{sub}_b4000*.nii.gz"))
            t1 = list(path.glob(f"{sub}_t1.nii.gz"))

            if b1000 or b2000 or b4000:
                dataset_info["has_diffusion"] = True
                self.report["diffusion_subjects"].append(f"Stanford-{sub}")

            if t1:
                dataset_info["has_anatomical"] = True
                self.report["anatomical_subjects"].append(f"Stanford-{sub}")

            # Track shells
            if b1000:
                dataset_info["shells"].append("b1000")
            if b2000:
                dataset_info["shells"].append("b2000")
            if b4000:
                dataset_info["shells"].append("b4000")

        dataset_info["shells"] = sorted(list(set(dataset_info["shells"])))

        # Check for bval/bvec
        bval_files = list(path.glob("*.bval"))
        bvec_files = list(path.glob("*.bvec"))
        dataset_info["has_bvals"] = len(bval_files) > 0
        dataset_info["has_bvecs"] = len(bvec_files) > 0

        if not dataset_info["has_bvals"]:
            self.report["issues"].append(
                "WARNING: Stanford dataset missing .bval files - may need extraction or conversion"
            )
        if not dataset_info["has_bvecs"]:
            self.report["issues"].append(
                "WARNING: Stanford dataset missing .bvec files - may need extraction or conversion"
            )

        self.report["datasets"]["stanford"] = dataset_info
        self.report["total_subjects"] += len(subjects)

        print(f"  [OK] Found {len(subjects)} subjects: {', '.join(sorted(subjects))}")
        print(f"  [OK] Shells: {', '.join(dataset_info['shells'])}")
        print(f"  [INFO] bval files: {len(bval_files)}, bvec files: {len(bvec_files)}")
        print()

    def analyze_bids(self, path: Path):
        """Analyze BIDS-formatted dataset"""
        dataset_info = {
            "name": "OpenNeuro (BIDS)",
            "type": "BIDS multi-modal",
            "subjects": [],
            "has_diffusion": False,
            "has_anatomical": False,
            "has_functional": False,
            "has_fieldmap": False
        }

        # Find BIDS subjects
        subjects = sorted([d.name for d in path.glob("sub-*") if d.is_dir()])
        dataset_info["subjects"] = subjects[:10]  # Store first 10 for report
        dataset_info["total_subjects"] = len(subjects)

        # Sample first subject to check modalities
        if subjects:
            sample_sub = path / subjects[0]
            sessions = list(sample_sub.glob("ses-*"))

            if sessions:
                session = sessions[0]
                if (session / "dwi").exists():
                    dataset_info["has_diffusion"] = True
                if (session / "anat").exists():
                    dataset_info["has_anatomical"] = True
                if (session / "func").exists():
                    dataset_info["has_functional"] = True
                if (session / "fmap").exists():
                    dataset_info["has_fieldmap"] = True

        # Count subjects with diffusion data
        diffusion_count = 0
        for sub_dir in list(path.glob("sub-*"))[:10]:  # Sample first 10
            dwi_dir = sub_dir / "ses-02" / "dwi"
            if not dwi_dir.exists():
                dwi_dir = sub_dir / "ses-01" / "dwi"
            if dwi_dir.exists():
                diffusion_count += 1
                self.report["diffusion_subjects"].append(sub_dir.name)

        if diffusion_count > 0:
            dataset_info["has_diffusion"] = True
        else:
            self.report["issues"].append(
                "WARNING: OpenNeuro dataset appears to lack diffusion MRI data (functional MRI only)"
            )

        self.report["datasets"]["openneuro"] = dataset_info
        self.report["total_subjects"] += len(subjects)

        print(f"  [OK] Found {len(subjects)} subjects (BIDS format)")
        print(f"  [OK] Anatomical: {dataset_info['has_anatomical']}")
        print(f"  [OK] Diffusion: {dataset_info['has_diffusion']}")
        print(f"  [OK] Functional: {dataset_info['has_functional']}")
        print(f"  [OK] Fieldmap: {dataset_info['has_fieldmap']}")
        print()

    def _get_directory_size(self, path: Path) -> int:
        """Calculate total directory size"""
        total = 0
        try:
            for entry in path.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception as e:
            print(f"Warning: Could not calculate size: {e}")
        return total

    def _print_summary(self):
        """Print analysis summary"""
        print("="*60)
        print("Summary")
        print("="*60)
        print(f"Total subjects: {self.report['total_subjects']}")
        print(f"Subjects with diffusion data: {len(self.report['diffusion_subjects'])}")
        print(f"Total dataset size: {self.report['total_size_gb']} GB")
        print()

        if self.report["issues"]:
            print("WARNING - Issues Found:")
            for issue in self.report["issues"]:
                print(f"  {issue}")
            print()

        print("Recommendation:")
        if len(self.report['diffusion_subjects']) >= 2:
            print("  [OK] Sufficient data for pipeline development and testing")
        else:
            print("  [WARNING] Limited diffusion data - may need additional datasets")
        print()

    def save_report(self, output_path: str):
        """Save analysis report to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.report, f, indent=2)
        print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    import sys

    dataset_root = sys.argv[1] if len(sys.argv) > 1 else "datasets"
    analyzer = DatasetAnalyzer(dataset_root)
    report = analyzer.scan_all()

    # Save report
    output_path = "analysis_and_decisions/dataset_analysis_report.json"
    analyzer.save_report(output_path)
