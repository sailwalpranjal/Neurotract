#!/usr/bin/env python3
"""
Authoritative Demo Dataset Downloader & Verifier for NeuroTract 2.0

Bootstraps verified open scientific datasets for demonstration and validation.
Supports:
1. Stanford HARDI (Rokem et al., CC BY 3.0)
2. Sherbrooke 3-Shell HARDI (Descoteaux et al., CC BY 3.0)
3. Local dataset verification and checksum audits
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("demo_data_downloader")

DATASET_CATALOG: Dict[str, Dict[str, Any]] = {
    "stanford_hardi": {
        "name": "Stanford HARDI Single Subject",
        "source": "Stanford Center for Cognitive and Neurobiological Imaging (CNI)",
        "url": "http://purl.stanford.edu/ng782rw8378",
        "citation": "Rokem, A. et al. (2015). Evaluating the accuracy of diffusion MRI models in white matter. PLOS ONE 10(4): e0123272.",
        "license": "Creative Commons Attribution 3.0 Unported (CC BY 3.0)",
        "scanner": "GE Discovery MR750 3.0T",
        "sequence": "2D Spin Echo EPI, 150 directions, b=1000/2000/4000 s/mm^2",
        "dipy_fetcher": "fetch_stanford_hardi",
        "expected_files": ["SUB1_b1000_1.nii.gz", "SUB1_b1000_1.bvals", "SUB1_b1000_1.bvecs"],
    },
    "sherbrooke_3shell": {
        "name": "Sherbrooke 3-Shell HARDI",
        "source": "Sherbrooke Connectivity Imaging Lab (SCIL)",
        "url": "https://dipy.org",
        "citation": "Descoteaux, M. et al. (2011). Multiple q-shell HARDI in human brain. Proc. Intl. Soc. Mag. Reson. Med. 19.",
        "license": "Creative Commons Attribution 3.0 (CC BY 3.0)",
        "scanner": "Siemens 1.5T Sonata",
        "sequence": "3-shell HARDI: b=1000, 2000, 3500 s/mm^2 (64 dirs each) + 4 b0s",
        "dipy_fetcher": "fetch_sherbrooke_3shell",
        "expected_files": ["HARDI193.nii.gz", "HARDI193.bval", "HARDI193.bvec"],
    },
}


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file"""
    if not filepath.exists() or not filepath.is_file():
        return ""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_local_stanford(data_dir: Path) -> bool:
    """Verify existing Stanford dataset files in datasets/Stanford dataset"""
    logger.info(f"Verifying local Stanford dataset at: {data_dir}")
    if not data_dir.exists():
        logger.warning(f"Directory {data_dir} does not exist.")
        return False

    required = [
        "SUB1_b1000_1.nii.gz",
        "SUB1_b1000_1.bvals",
        "SUB1_b1000_1.bvecs",
        "SUB1_aparc-reduced.nii.gz",
    ]

    all_ok = True
    manifest = {}
    for fname in required:
        fpath = data_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            sha = compute_sha256(fpath)
            logger.info(f"  [OK] {fname}: {size_mb:.2f} MB (SHA-256: {sha[:12]}...)")
            manifest[fname] = {"size_bytes": fpath.stat().st_size, "sha256": sha}
        else:
            logger.error(f"  [MISSING] {fname}")
            all_ok = False

    manifest_file = data_dir / "verified_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {manifest_file}")
    return all_ok


def verify_precomputed_artifacts(output_dir: Path) -> bool:
    """Verify precomputed demo artifacts in output/SUB1"""
    logger.info(f"Verifying precomputed demo artifacts at: {output_dir}")
    sub1_dir = output_dir / "SUB1"
    if not sub1_dir.exists():
        logger.warning(f"No precomputed SUB1 directory in {output_dir}")
        return False

    key_artifacts = [
        "streamlines.trk",
        "connectome.npy",
        "metrics.json",
        "fod.nii.gz",
        "brain_mesh_step1.json",
        "dti/dti_fa.nii.gz",
    ]

    all_present = True
    for rel_path in key_artifacts:
        p = sub1_dir / rel_path
        if p.exists():
            logger.info(f"  [FOUND] {rel_path} ({p.stat().st_size / 1024:.1f} KB)")
        else:
            logger.warning(f"  [MISSING] {rel_path}")
            all_present = False

    return all_present


def fetch_dipy_dataset(dataset_key: str, target_dir: Path) -> bool:
    """Fetch official open dataset using DIPY fetchers into cache directory"""
    try:
        import dipy.data as dpd
    except ImportError:
        logger.error("DIPY is not installed. Please activate your venv.")
        return False

    if dataset_key not in DATASET_CATALOG:
        logger.error(f"Unknown dataset key: {dataset_key}")
        return False

    info = DATASET_CATALOG[dataset_key]
    fetcher_name = info["dipy_fetcher"]
    logger.info(f"Bootstrapping {info['name']} via DIPY ({fetcher_name})...")

    target_dir.mkdir(parents=True, exist_ok=True)

    fetcher_fn = getattr(dpd, fetcher_name, None)
    if not fetcher_fn:
        logger.error(f"DIPY does not expose {fetcher_name}")
        return False

    try:
        files = fetcher_fn()
        logger.info(f"Successfully fetched {info['name']}: {files}")
        return True
    except Exception as e:
        logger.error(f"Failed to fetch {info['name']}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="NeuroTract Demo Dataset Downloader & Verifier")
    parser.add_argument("--verify-local", action="store_true", help="Verify existing local datasets in datasets/")
    parser.add_argument("--verify-artifacts", action="store_true", help="Verify precomputed demo artifacts in output/SUB1")
    parser.add_argument("--fetch", choices=["stanford_hardi", "sherbrooke_3shell", "all"], help="Fetch dataset via DIPY")
    parser.add_argument("--target-dir", default="datasets/cache", help="Target cache directory for downloaded data")

    args = parser.parse_args()

    # Default action if no arguments: verify local datasets and precomputed artifacts
    if not args.verify_local and not args.verify_artifacts and not args.fetch:
        args.verify_local = True
        args.verify_artifacts = True

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "datasets" / "Stanford dataset"
    output_dir = repo_root / "output"

    if args.verify_local:
        verify_local_stanford(data_dir)

    if args.verify_artifacts:
        verify_precomputed_artifacts(output_dir)

    if args.fetch:
        target = repo_root / args.target_dir
        if args.fetch == "all":
            for k in DATASET_CATALOG:
                fetch_dipy_dataset(k, target)
        else:
            fetch_dipy_dataset(args.fetch, target)


if __name__ == "__main__":
    main()
