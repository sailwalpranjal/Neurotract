"""
NeuroTract Command-Line Interface

Main CLI for running tractography and connectivity analysis pipelines.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional

from .utils.logger import get_logger, log_decision


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="NeuroTract: Brain White Matter Tractography & Connectivity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on a subject
  neurotract run --subject subject_001 --config config.json

  # Preprocessing only
  neurotract preprocess --input data.nii.gz --bvals data.bval --bvecs data.bvec --output processed/

  # Tractography
  neurotract tractography --input processed/ --output tracts/ --seeds-per-voxel 2

  # Build connectome
  neurotract connectome --streamlines tracts.trk --parcellation atlas.nii.gz --output connectome.npy

  # Compute graph metrics
  neurotract metrics --connectome connectome.npy --output metrics.json
        """
    )

    parser.add_argument('--version', action='version', version='NeuroTract 0.1.0')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run command (full pipeline)
    run_parser = subparsers.add_parser('run', help='Run full pipeline')
    run_parser.add_argument('--subject', required=True, help='Subject ID or manifest JSON')
    run_parser.add_argument('--config', help='Pipeline configuration JSON')
    run_parser.add_argument('--mode', choices=['quick', 'full', 'lowmem'], default='full',
                           help='Processing mode')
    run_parser.add_argument('--output', '-o', required=True, help='Output directory')

    # Preprocess command
    preproc_parser = subparsers.add_parser('preprocess', help='Preprocessing pipeline')
    preproc_parser.add_argument('--input', '-i', required=True, help='Input DWI data (NIfTI)')
    preproc_parser.add_argument('--bvals', required=True, help='b-values file')
    preproc_parser.add_argument('--bvecs', required=True, help='b-vectors file')
    preproc_parser.add_argument('--t1', help='T1-weighted anatomical (optional)')
    preproc_parser.add_argument('--output', '-o', required=True, help='Output directory')
    preproc_parser.add_argument('--no-denoise', action='store_true', help='Skip denoising')
    preproc_parser.add_argument('--no-motion-correction', action='store_true',
                               help='Skip motion correction')

    # DTI command
    dti_parser = subparsers.add_parser('dti', help='Compute DTI maps')
    dti_parser.add_argument('--input', '-i', required=True, help='Preprocessed DWI')
    dti_parser.add_argument('--bvals', required=True, help='b-values file')
    dti_parser.add_argument('--bvecs', required=True, help='b-vectors file')
    dti_parser.add_argument('--mask', help='Brain mask')
    dti_parser.add_argument('--output', '-o', required=True, help='Output directory')

    # CSD/FOD command
    csd_parser = subparsers.add_parser('csd', help='Compute FOD using CSD')
    csd_parser.add_argument('--input', '-i', required=True, help='Preprocessed DWI')
    csd_parser.add_argument('--bvals', required=True, help='b-values file')
    csd_parser.add_argument('--bvecs', required=True, help='b-vectors file')
    csd_parser.add_argument('--mask', help='Brain mask')
    csd_parser.add_argument('--response', help='Response function (auto-estimated if not provided)')
    csd_parser.add_argument('--sh-order', type=int, help='SH order (auto if not specified)')
    csd_parser.add_argument('--output', '-o', required=True, help='Output FOD file')

    # Tractography command
    tract_parser = subparsers.add_parser('tractography', help='Probabilistic tractography')
    tract_parser.add_argument('--fod', required=True, help='FOD file')
    tract_parser.add_argument('--mask', help='Brain mask')
    tract_parser.add_argument('--seeds-per-voxel', type=int, default=2,
                             help='Seeds per voxel (default: 2)')
    tract_parser.add_argument('--step-size', type=float, default=0.5,
                             help='Step size in mm (default: 0.5)')
    tract_parser.add_argument('--max-angle', type=float, default=30,
                             help='Max angle per step in degrees (default: 30)')
    tract_parser.add_argument('--fa-threshold', type=float, default=0.1,
                             help='FA threshold for termination (default: 0.1)')
    tract_parser.add_argument('--fa-map', help='FA map for termination checking (NIfTI)')
    tract_parser.add_argument('--output', '-o', required=True, help='Output streamlines file')

    # Connectome command
    conn_parser = subparsers.add_parser('connectome', help='Build structural connectome')
    conn_parser.add_argument('--streamlines', required=True, help='Streamlines file (TRK/TCK)')
    conn_parser.add_argument('--parcellation', required=True, help='Parcellation NIfTI')
    conn_parser.add_argument('--weighting', choices=['count', 'length_normalized', 'mean_fa', 'hybrid'],
                            default='count', help='Edge weighting strategy')
    conn_parser.add_argument('--fa-map', help='FA map for microstructure weighting')
    conn_parser.add_argument('--output', '-o', required=True, help='Output connectome file')

    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Compute graph theory metrics')
    metrics_parser.add_argument('--connectome', required=True, help='Connectome matrix (NPY/CSV)')
    metrics_parser.add_argument('--output', '-o', required=True, help='Output metrics JSON')

    # Analysis command (pathology detection)
    analysis_parser = subparsers.add_parser('analyze', help='Pathology detection and analysis')
    analysis_parser.add_argument('--connectome', required=True, help='Connectome matrix')
    analysis_parser.add_argument('--metrics', help='Pre-computed metrics JSON')
    analysis_parser.add_argument('--model', help='Trained model file')
    analysis_parser.add_argument('--output', '-o', required=True, help='Output analysis report')

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    logger = get_logger()
    logger.setLevel(log_level)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        if args.command == 'run':
            run_full_pipeline(args)
        elif args.command == 'preprocess':
            run_preprocessing(args)
        elif args.command == 'dti':
            run_dti(args)
        elif args.command == 'csd':
            run_csd(args)
        elif args.command == 'tractography':
            run_tractography(args)
        elif args.command == 'connectome':
            build_connectome(args)
        elif args.command == 'metrics':
            compute_metrics(args)
        elif args.command == 'analyze':
            run_analysis(args)
        else:
            parser.print_help()
            sys.exit(1)

        logger.info("Command completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.debug)
        sys.exit(1)


def run_full_pipeline(args):
    """Run full NeuroTract pipeline"""
    import nibabel as nib
    import numpy as np
    from pathlib import Path
    from datetime import datetime

    from .data.data_loader import DataLoader
    from .preprocessing.pipeline import PreprocessingPipeline
    from .microstructure.dti import DTIModel
    from .microstructure.csd import CSDModel, ResponseFunction
    from .tractography.probabilistic_tracker import ProbabilisticTracker
    from .tractography.seeding import SeedGenerator
    from .connectome.construct import ConnectomeBuilder, load_parcellation
    from .connectome.graph_metrics import ConnectomeMetrics
    from .tractography.streamline_utils import StreamlineUtils

    logger = get_logger()
    logger.info("=" * 80)
    logger.info("NEUROTRACT FULL PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output: {args.output}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()

    try:
        # Load configuration
        config = {}
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            with open(args.config, 'r') as f:
                config = json.load(f)

        # Load subject manifest
        if args.subject.endswith('.json'):
            logger.info(f"Loading subject manifest from {args.subject}")
            with open(args.subject, 'r') as f:
                manifest = json.load(f)
        else:
            # Auto-detect files
            manifest = {
                'subject_id': args.subject,
                'dwi': config.get('dwi_path'),
                'bval': config.get('bval_path'),
                'bvec': config.get('bvec_path'),
                'parcellation': config.get('parcellation_path')
            }

        subject_id = manifest.get('subject_id', 'unknown')

        # Log pipeline decision
        log_decision(
            decision_id=f"PIPELINE-{subject_id}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            component="full_pipeline",
            decision=f"Running {args.mode} mode pipeline for subject {subject_id}",
            rationale=f"User requested complete tractography and connectivity analysis",
            parameters={
                'mode': args.mode,
                'subject': subject_id,
                'output': str(output_dir),
                'config': config
            }
        )

        # =====================================================================
        # STEP 1: PREPROCESSING
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: PREPROCESSING")
        logger.info("=" * 80)

        preproc_dir = output_dir / "preprocessing"

        # Adjust preprocessing based on mode
        skip_motion = (args.mode == 'quick')
        skip_bias = (args.mode == 'quick')

        preproc_pipeline = PreprocessingPipeline(
            output_dir=preproc_dir,
            skip_motion_correction=skip_motion,
            skip_brain_extraction=False,
            skip_bias_correction=skip_bias,
            save_intermediate=True,
            save_qc_reports=True
        )

        preproc_outputs = preproc_pipeline.run(
            dwi_path=manifest['dwi'],
            bval_path=manifest.get('bval'),
            bvec_path=manifest.get('bvec'),
            output_prefix=f"{subject_id}_preprocessed"
        )

        # Load preprocessed data
        loader = DataLoader(use_mmap=(args.mode == 'lowmem'), validate=True)
        dwi_data = loader.load_diffusion(
            preproc_outputs['dwi'],
            preproc_outputs['bval'],
            preproc_outputs['bvec']
        )
        mask_img = nib.load(preproc_outputs['mask'])
        mask = mask_img.get_fdata() > 0

        # =====================================================================
        # STEP 2: DTI COMPUTATION
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: DTI COMPUTATION")
        logger.info("=" * 80)

        dti_dir = output_dir / "dti"
        dti_dir.mkdir(exist_ok=True)

        dti_model = DTIModel(dwi_data.bvals, dwi_data.bvecs)
        dti_results = dti_model.fit(dwi_data.data, mask=mask, return_s0=False)

        # Save DTI maps
        for metric_name in ['fa', 'md', 'rd', 'ad']:
            metric_path = dti_dir / f"{subject_id}_{metric_name}.nii.gz"
            metric_img = nib.Nifti1Image(dti_results[metric_name], dwi_data.affine)
            nib.save(metric_img, metric_path)
            logger.info(f"Saved {metric_name.upper()} map to {metric_path}")

        # =====================================================================
        # STEP 3: CSD/FOD COMPUTATION
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: CSD/FOD COMPUTATION")
        logger.info("=" * 80)

        csd_dir = output_dir / "csd"
        csd_dir.mkdir(exist_ok=True)

        # Estimate response function
        response_estimator = ResponseFunction(method='dhollander')
        response = response_estimator.estimate(
            dwi_data.data, dwi_data.bvals, dwi_data.bvecs,
            fa_map=dti_results['fa'], mask=mask
        )

        # Compute FOD
        sh_order = 6 if args.mode == 'quick' else 8
        csd_model = CSDModel(dwi_data.bvals, dwi_data.bvecs, sh_order=sh_order)
        fod = csd_model.fit(dwi_data.data, response['wm'], mask=mask)

        # Save FOD
        fod_path = csd_dir / f"{subject_id}_fod.nii.gz"
        fod_img = nib.Nifti1Image(fod, dwi_data.affine)
        nib.save(fod_img, fod_path)
        logger.info(f"Saved FOD to {fod_path}")

        # =====================================================================
        # STEP 4: TRACTOGRAPHY
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: PROBABILISTIC TRACTOGRAPHY")
        logger.info("=" * 80)

        tract_dir = output_dir / "tractography"
        tract_dir.mkdir(exist_ok=True)

        # Generate seeds
        seeds_per_voxel = 1 if args.mode == 'quick' else 2
        seed_gen = SeedGenerator()
        seeds, seed_metadata = seed_gen.whole_brain_seeds(
            mask=mask,
            seeds_per_voxel=seeds_per_voxel,
            voxel_size=dwi_data.voxel_size,
            jitter=True
        )
        logger.info(f"Generated {len(seeds)} seeds")

        # Create direction getter from FOD using proper peak finding
        logger.info("Creating direction getter from FOD...")
        from .tractography.fod_utils import create_fod_direction_getter

        direction_getter = create_fod_direction_getter(
            fod,
            sh_order=sh_order,
            relative_peak_threshold=0.5,
            min_separation_angle=25.0,
            use_primary_only=True
        )

        # Run tractography
        tracker = ProbabilisticTracker(
            voxel_size=dwi_data.voxel_size,
            step_size=0.5,
            max_angle=30.0,
            fa_threshold=0.1,
            fod_threshold=0.1,
            n_samples_per_seed=1
        )

        streamlines_path = tract_dir / f"{subject_id}_streamlines.h5"
        streamlines, tract_metadata = tracker.track(
            seeds=seeds,
            direction_getter=direction_getter,
            fa_volume=dti_results['fa'],
            mask_volume=mask,
            affine=dwi_data.affine,
            output_file=str(streamlines_path) if args.mode == 'lowmem' else None
        )

        # Save streamlines if not already saved
        if args.mode != 'lowmem':
            StreamlineUtils.save_trk(
                streamlines,
                str(tract_dir / f"{subject_id}_streamlines.trk"),
                dwi_data.affine,
                dwi_data.voxel_size,
                np.array(dwi_data.shape[:3])
            )

        logger.info(f"Tracked {len(streamlines)} streamlines")

        # =====================================================================
        # STEP 5: CONNECTOME CONSTRUCTION
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: CONNECTOME CONSTRUCTION")
        logger.info("=" * 80)

        connectome_dir = output_dir / "connectome"
        connectome_dir.mkdir(exist_ok=True)

        # Load parcellation
        parcellation_affine = dwi_data.affine  # Default to DWI affine
        if 'parcellation' in manifest and manifest['parcellation']:
            parcellation, labels, parcellation_affine = load_parcellation(manifest['parcellation'])
        else:
            logger.warning("No parcellation provided, using simple grid parcellation")
            # Create simple grid parcellation
            parcellation = np.zeros(mask.shape, dtype=int)
            grid_size = 10
            for i in range(0, mask.shape[0], grid_size):
                for j in range(0, mask.shape[1], grid_size):
                    for k in range(0, mask.shape[2], grid_size):
                        parcel_id = (i // grid_size) * 100 + (j // grid_size) * 10 + (k // grid_size)
                        parcellation[i:i+grid_size, j:j+grid_size, k:k+grid_size] = parcel_id
            parcellation *= mask  # Apply mask
            labels = None

        # Build connectome
        builder = ConnectomeBuilder(parcellation, parcel_labels=labels, affine=parcellation_affine)
        adjacency = builder.build_connectome(
            streamlines,
            weighting='hybrid',
            microstructure_map=dti_results['fa'],
            symmetric=True,
            normalize=False
        )

        # Save connectome
        connectome_path = connectome_dir / f"{subject_id}_connectome.npy"
        np.save(connectome_path, adjacency)
        logger.info(f"Saved connectome to {connectome_path}")

        # Also save as CSV
        csv_path = connectome_dir / f"{subject_id}_connectome.csv"
        np.savetxt(csv_path, adjacency, delimiter=',', fmt='%.6f')
        logger.info(f"Saved connectome CSV to {csv_path}")

        # =====================================================================
        # STEP 6: GRAPH METRICS
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: GRAPH THEORY METRICS")
        logger.info("=" * 80)

        metrics_calculator = ConnectomeMetrics(adjacency, node_labels=labels)
        metrics = metrics_calculator.compute_all_metrics()

        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            elif isinstance(value, dict):
                metrics_serializable[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        metrics_serializable[key][k] = v.tolist()
                    elif isinstance(v, (np.integer, np.floating)):
                        metrics_serializable[key][k] = float(v)
                    else:
                        metrics_serializable[key][k] = v
            else:
                metrics_serializable[key] = value

        # Save metrics
        metrics_path = connectome_dir / f"{subject_id}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")

        # =====================================================================
        # STEP 7: GENERATE FINAL REPORT
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: GENERATING FINAL REPORT")
        logger.info("=" * 80)

        elapsed = datetime.now() - start_time

        report_path = output_dir / f"{subject_id}_pipeline_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NEUROTRACT PIPELINE REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Subject: {subject_id}\n")
            f.write(f"Mode: {args.mode}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total processing time: {elapsed}\n\n")

            f.write("-" * 80 + "\n")
            f.write("INPUT DATA\n")
            f.write("-" * 80 + "\n")
            f.write(f"DWI shape: {dwi_data.shape}\n")
            f.write(f"Voxel size: {dwi_data.voxel_size} mm\n")
            f.write(f"Number of volumes: {dwi_data.num_volumes}\n")
            f.write(f"B-value shells: {dwi_data.get_shells()}\n\n")

            f.write("-" * 80 + "\n")
            f.write("PREPROCESSING\n")
            f.write("-" * 80 + "\n")
            f.write(f"Motion correction: {'Yes' if not skip_motion else 'No'}\n")
            f.write(f"Bias correction: {'Yes' if not skip_bias else 'No'}\n")
            f.write(f"Brain extraction: Yes\n\n")

            f.write("-" * 80 + "\n")
            f.write("DTI METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean FA: {np.mean(dti_results['fa'][mask]):.3f}\n")
            f.write(f"Mean MD: {np.mean(dti_results['md'][mask]):.6f}\n")
            f.write(f"Mean RD: {np.mean(dti_results['rd'][mask]):.6f}\n")
            f.write(f"Mean AD: {np.mean(dti_results['ad'][mask]):.6f}\n\n")

            f.write("-" * 80 + "\n")
            f.write("TRACTOGRAPHY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of seeds: {len(seeds)}\n")
            f.write(f"Seeds per voxel: {seeds_per_voxel}\n")
            f.write(f"Streamlines generated: {tract_metadata['statistics']['n_streamlines_kept']}\n")
            f.write(f"Success rate: {100.0 * tract_metadata['statistics']['n_streamlines_kept'] / max(1, tract_metadata['statistics']['n_streamlines_attempted']):.1f}%\n\n")

            f.write("-" * 80 + "\n")
            f.write("CONNECTOME\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of parcels: {builder.n_parcels}\n")
            f.write(f"Number of edges: {np.count_nonzero(adjacency)}\n")
            f.write(f"Network density: {metrics_serializable['global']['density']:.3f}\n\n")

            f.write("-" * 80 + "\n")
            f.write("GRAPH METRICS (GLOBAL)\n")
            f.write("-" * 80 + "\n")
            for key, value in metrics_serializable['global'].items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.3f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")

            f.write("-" * 80 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Preprocessed DWI: {preproc_outputs['dwi']}\n")
            f.write(f"Brain mask: {preproc_outputs['mask']}\n")
            f.write(f"FA map: {dti_dir / f'{subject_id}_fa.nii.gz'}\n")
            f.write(f"FOD: {fod_path}\n")
            f.write(f"Streamlines: {streamlines_path}\n")
            f.write(f"Connectome: {connectome_path}\n")
            f.write(f"Metrics: {metrics_path}\n\n")

            f.write("=" * 80 + "\n")
            f.write("PIPELINE COMPLETED SUCCESSFULLY\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Pipeline report saved to {report_path}")

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Total processing time: {elapsed}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


def run_preprocessing(args):
    """Run preprocessing"""
    from pathlib import Path
    from .preprocessing.pipeline import PreprocessingPipeline

    logger = get_logger()
    logger.info("=" * 80)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 80)

    try:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create preprocessing pipeline
        pipeline = PreprocessingPipeline(
            output_dir=output_dir,
            skip_motion_correction=args.no_motion_correction,
            skip_brain_extraction=False,
            skip_bias_correction=False,
            motion_registration='affine',
            brain_method='median_otsu',
            bias_method='auto',
            save_intermediate=True,
            save_qc_reports=True
        )

        # Run preprocessing
        outputs = pipeline.run(
            dwi_path=args.input,
            bval_path=args.bvals,
            bvec_path=args.bvecs,
            output_prefix="preprocessed"
        )

        logger.info("\n" + "=" * 80)
        logger.info("PREPROCESSING COMPLETED")
        logger.info("=" * 80)
        logger.info("Output files:")
        for key, path in outputs.items():
            logger.info(f"  {key}: {path}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        raise


def run_dti(args):
    """Compute DTI maps"""
    import nibabel as nib
    import numpy as np
    from pathlib import Path
    from .data.data_loader import DataLoader
    from .microstructure.dti import DTIModel

    logger = get_logger()
    logger.info("=" * 80)
    logger.info("DTI COMPUTATION")
    logger.info("=" * 80)

    try:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load diffusion data
        logger.info("Loading diffusion data...")
        loader = DataLoader(use_mmap=True, validate=True)
        dwi_data = loader.load_diffusion(
            nifti_path=args.input,
            bval_path=args.bvals,
            bvec_path=args.bvecs,
            auto_detect=True
        )

        # Load mask if provided
        mask = None
        if args.mask:
            logger.info(f"Loading brain mask from {args.mask}")
            mask_img = nib.load(args.mask)
            mask = mask_img.get_fdata() > 0
        else:
            logger.info("No mask provided, processing all voxels")
            mask = np.ones(dwi_data.shape[:3], dtype=bool)

        # Create DTI model
        logger.info("Initializing DTI model...")
        dti_model = DTIModel(
            bvals=dwi_data.bvals,
            bvecs=dwi_data.bvecs,
            b_threshold=50.0
        )

        # Fit DTI
        logger.info("Fitting DTI model...")
        dti_results = dti_model.fit(
            dwi_data=dwi_data.data,
            mask=mask,
            return_s0=False
        )

        # Save DTI maps
        logger.info("Saving DTI maps...")

        # FA map
        fa_path = output_dir / "dti_fa.nii.gz"
        fa_img = nib.Nifti1Image(dti_results['fa'], dwi_data.affine, dwi_data.header)
        fa_img.header['descrip'] = b'Fractional Anisotropy (FA)'
        nib.save(fa_img, fa_path)
        logger.info(f"Saved FA map to {fa_path}")

        # MD map
        md_path = output_dir / "dti_md.nii.gz"
        md_img = nib.Nifti1Image(dti_results['md'], dwi_data.affine, dwi_data.header)
        md_img.header['descrip'] = b'Mean Diffusivity (MD)'
        nib.save(md_img, md_path)
        logger.info(f"Saved MD map to {md_path}")

        # RD map
        rd_path = output_dir / "dti_rd.nii.gz"
        rd_img = nib.Nifti1Image(dti_results['rd'], dwi_data.affine, dwi_data.header)
        rd_img.header['descrip'] = b'Radial Diffusivity (RD)'
        nib.save(rd_img, rd_path)
        logger.info(f"Saved RD map to {rd_path}")

        # AD map
        ad_path = output_dir / "dti_ad.nii.gz"
        ad_img = nib.Nifti1Image(dti_results['ad'], dwi_data.affine, dwi_data.header)
        ad_img.header['descrip'] = b'Axial Diffusivity (AD)'
        nib.save(ad_img, ad_path)
        logger.info(f"Saved AD map to {ad_path}")

        # Save eigenvalues
        eigenvalues_path = output_dir / "dti_eigenvalues.nii.gz"
        eigenvalues_img = nib.Nifti1Image(dti_results['eigenvalues'], dwi_data.affine)
        eigenvalues_img.header['descrip'] = b'DTI Eigenvalues'
        nib.save(eigenvalues_img, eigenvalues_path)
        logger.info(f"Saved eigenvalues to {eigenvalues_path}")

        # Save eigenvectors (primary eigenvector for color FA)
        v1_path = output_dir / "dti_v1.nii.gz"
        v1 = dti_results['eigenvectors'][..., :, 0]  # Primary eigenvector
        v1_img = nib.Nifti1Image(v1, dwi_data.affine)
        v1_img.header['descrip'] = b'Primary Eigenvector (V1)'
        nib.save(v1_img, v1_path)
        logger.info(f"Saved primary eigenvector to {v1_path}")

        # Generate summary statistics
        logger.info("\nDTI Summary Statistics (within mask):")
        logger.info(f"  Mean FA: {np.mean(dti_results['fa'][mask]):.3f}")
        logger.info(f"  Std FA:  {np.std(dti_results['fa'][mask]):.3f}")
        logger.info(f"  Mean MD: {np.mean(dti_results['md'][mask]):.6f} mm²/s")
        logger.info(f"  Mean RD: {np.mean(dti_results['rd'][mask]):.6f} mm²/s")
        logger.info(f"  Mean AD: {np.mean(dti_results['ad'][mask]):.6f} mm²/s")

        # Save summary
        summary_path = output_dir / "dti_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("DTI Summary Statistics\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Input: {args.input}\n")
            f.write(f"Mask: {args.mask if args.mask else 'None (all voxels)'}\n\n")
            f.write("Within-mask statistics:\n")
            f.write(f"  Mean FA: {np.mean(dti_results['fa'][mask]):.3f}\n")
            f.write(f"  Std FA:  {np.std(dti_results['fa'][mask]):.3f}\n")
            f.write(f"  Mean MD: {np.mean(dti_results['md'][mask]):.6f} mm²/s\n")
            f.write(f"  Mean RD: {np.mean(dti_results['rd'][mask]):.6f} mm²/s\n")
            f.write(f"  Mean AD: {np.mean(dti_results['ad'][mask]):.6f} mm²/s\n")
        logger.info(f"Saved summary to {summary_path}")

        logger.info("\n" + "=" * 80)
        logger.info("DTI COMPUTATION COMPLETED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"DTI computation failed: {e}", exc_info=True)
        raise


def run_csd(args):
    """Compute CSD/FOD"""
    import nibabel as nib
    import numpy as np
    from pathlib import Path
    from .data.data_loader import DataLoader
    from .microstructure.csd import CSDModel, ResponseFunction, auto_select_sh_order
    from .microstructure.dti import DTIModel

    logger = get_logger()
    logger.info("=" * 80)
    logger.info("CSD/FOD COMPUTATION")
    logger.info("=" * 80)

    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load diffusion data
        logger.info("Loading diffusion data...")
        loader = DataLoader(use_mmap=True, validate=True)
        dwi_data = loader.load_diffusion(
            nifti_path=args.input,
            bval_path=args.bvals,
            bvec_path=args.bvecs,
            auto_detect=True
        )

        # Load mask if provided
        mask = None
        if args.mask:
            logger.info(f"Loading brain mask from {args.mask}")
            mask_img = nib.load(args.mask)
            mask = mask_img.get_fdata() > 0
        else:
            logger.info("No mask provided, processing all voxels")
            mask = np.ones(dwi_data.shape[:3], dtype=bool)

        # Determine SH order
        if args.sh_order:
            sh_order = args.sh_order
            logger.info(f"Using user-specified SH order: {sh_order}")
        else:
            sh_order = auto_select_sh_order(np.max(dwi_data.bvals))
            logger.info(f"Auto-selected SH order: {sh_order} (based on bmax={np.max(dwi_data.bvals)})")

        # Estimate response function
        if args.response:
            logger.info(f"Loading response function from {args.response}")
            response_data = np.loadtxt(args.response)
            response = {'wm': response_data}
        else:
            logger.info("Estimating response function...")

            # Need FA map for response estimation
            logger.info("Computing FA for response estimation...")
            dti_model = DTIModel(dwi_data.bvals, dwi_data.bvecs)
            dti_results = dti_model.fit(dwi_data.data, mask=mask, return_s0=False)

            response_estimator = ResponseFunction(method='dhollander')
            response = response_estimator.estimate(
                dwi_data.data,
                dwi_data.bvals,
                dwi_data.bvecs,
                fa_map=dti_results['fa'],
                mask=mask
            )

            # Save response function
            response_path = output_path.parent / "response_wm.txt"
            np.savetxt(response_path, response['wm'], fmt='%.6f')
            logger.info(f"Saved response function to {response_path}")

        # Create CSD model
        logger.info(f"Initializing CSD model (SH order={sh_order})...")
        csd_model = CSDModel(
            bvals=dwi_data.bvals,
            bvecs=dwi_data.bvecs,
            sh_order=sh_order,
            lambda_=1.0,
            tau=0.1
        )

        # Compute FOD
        logger.info("Computing FOD using CSD...")
        fod = csd_model.fit(
            dwi_data=dwi_data.data,
            response=response['wm'],
            mask=mask
        )

        # Save FOD
        logger.info(f"Saving FOD to {output_path}")
        fod_img = nib.Nifti1Image(fod, dwi_data.affine, dwi_data.header)
        fod_img.header['descrip'] = b'Fiber Orientation Distribution (FOD)'
        nib.save(fod_img, output_path)

        # Generate summary statistics
        logger.info("\nFOD Summary:")
        logger.info(f"  Shape: {fod.shape}")
        logger.info(f"  SH coefficients: {fod.shape[-1]}")
        logger.info(f"  Non-zero voxels: {np.count_nonzero(np.any(fod != 0, axis=-1))}")

        # Save summary
        summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("CSD/FOD Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Input: {args.input}\n")
            f.write(f"Output: {output_path}\n")
            f.write(f"Mask: {args.mask if args.mask else 'None (all voxels)'}\n\n")
            f.write(f"SH order: {sh_order}\n")
            f.write(f"Number of SH coefficients: {fod.shape[-1]}\n")
            f.write(f"FOD shape: {fod.shape}\n")
            f.write(f"Non-zero voxels: {np.count_nonzero(np.any(fod != 0, axis=-1))}\n")
        logger.info(f"Saved summary to {summary_path}")

        logger.info("\n" + "=" * 80)
        logger.info("CSD/FOD COMPUTATION COMPLETED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"CSD computation failed: {e}", exc_info=True)
        raise


def run_tractography(args):
    """Run tractography"""
    import nibabel as nib
    import numpy as np
    from pathlib import Path
    from .tractography.probabilistic_tracker import ProbabilisticTracker
    from .tractography.seeding import SeedGenerator
    from .tractography.streamline_utils import StreamlineUtils

    logger = get_logger()
    logger.info("=" * 80)
    logger.info("PROBABILISTIC TRACTOGRAPHY")
    logger.info("=" * 80)

    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load FOD
        logger.info(f"Loading FOD from {args.fod}")
        fod_img = nib.load(args.fod)
        fod = fod_img.get_fdata()
        affine = fod_img.affine
        voxel_size = fod_img.header.get_zooms()[:3]

        logger.info(f"FOD shape: {fod.shape}")
        logger.info(f"Voxel size: {voxel_size} mm")

        # Load mask if provided
        mask = None
        if args.mask:
            logger.info(f"Loading brain mask from {args.mask}")
            mask_img = nib.load(args.mask)
            mask = mask_img.get_fdata() > 0
        else:
            logger.info("No mask provided, using non-zero FOD voxels")
            mask = np.any(fod != 0, axis=-1)

        # Generate seeds
        logger.info(f"Generating seeds ({args.seeds_per_voxel} per voxel)...")
        seed_gen = SeedGenerator()
        seeds, seed_metadata = seed_gen.whole_brain_seeds(
            mask=mask,
            seeds_per_voxel=args.seeds_per_voxel,
            voxel_size=voxel_size,
            jitter=True
        )
        logger.info(f"Generated {len(seeds)} seeds")

        # Create direction getter from FOD using proper peak finding
        logger.info("Creating direction getter from FOD...")
        from .tractography.fod_utils import create_fod_direction_getter

        # Infer SH order from number of coefficients
        n_coeffs = fod.shape[-1]
        sh_order = int((-3 + np.sqrt(1 + 8 * n_coeffs)) / 2)
        if sh_order % 2 != 0:
            sh_order -= 1

        direction_getter = create_fod_direction_getter(
            fod,
            sh_order=sh_order,
            relative_peak_threshold=0.5,
            min_separation_angle=25.0,
            use_primary_only=True
        )

        # Initialize tracker
        logger.info("Initializing probabilistic tracker...")
        tracker = ProbabilisticTracker(
            voxel_size=voxel_size,
            step_size=args.step_size,
            max_angle=args.max_angle,
            fa_threshold=args.fa_threshold,
            fod_threshold=0.1,
            max_length=200.0,
            min_length=10.0,
            n_samples_per_seed=1
        )

        # Load FA map for termination checking
        fa_volume = None
        if hasattr(args, 'fa_map') and args.fa_map:
            logger.info(f"Loading FA map from {args.fa_map}")
            fa_img = nib.load(args.fa_map)
            fa_volume = fa_img.get_fdata()

        # Run tractography
        logger.info("Running tractography...")
        logger.info(f"  Step size: {args.step_size} mm")
        logger.info(f"  Max angle: {args.max_angle}°")
        logger.info(f"  FA threshold: {args.fa_threshold}")

        streamlines, metadata = tracker.track(
            seeds=seeds,
            direction_getter=direction_getter,
            fa_volume=fa_volume,
            mask_volume=mask,
            affine=affine,
            output_file=None,
            save_rejected=False
        )

        logger.info(f"\nTracked {len(streamlines)} streamlines")

        # Print statistics
        logger.info(tracker.get_statistics_summary())

        # Save streamlines in TRK format
        logger.info(f"\nSaving streamlines to {output_path}")

        if str(output_path).endswith('.trk'):
            StreamlineUtils.save_trk(
                streamlines,
                str(output_path),
                affine,
                voxel_size,
                np.array(fod.shape[:3])
            )
        elif str(output_path).endswith('.tck'):
            StreamlineUtils.save_tck(streamlines, str(output_path))
        elif str(output_path).endswith('.vtk'):
            StreamlineUtils.save_vtk(streamlines, str(output_path))
        else:
            # Default to TRK
            output_path = output_path.with_suffix('.trk')
            StreamlineUtils.save_trk(
                streamlines,
                str(output_path),
                affine,
                voxel_size,
                np.array(fod.shape[:3])
            )

        # Compute and save bundle statistics
        bundle_stats = StreamlineUtils.compute_bundle_statistics(streamlines)

        stats_path = output_path.parent / f"{output_path.stem}_statistics.json"
        with open(stats_path, 'w') as f:
            stats_combined = {
                'bundle_statistics': bundle_stats,
                'tracking_metadata': {
                    'n_seeds': len(seeds),
                    'seeds_per_voxel': args.seeds_per_voxel,
                    'step_size': args.step_size,
                    'max_angle': args.max_angle,
                    'fa_threshold': args.fa_threshold
                },
                'tracking_statistics': metadata['statistics']
            }
            json.dump(stats_combined, f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")

        logger.info("\n" + "=" * 80)
        logger.info("TRACTOGRAPHY COMPLETED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Tractography failed: {e}", exc_info=True)
        raise


def build_connectome(args):
    """Build connectome"""
    import nibabel as nib
    import numpy as np
    from pathlib import Path
    from .connectome.construct import ConnectomeBuilder, load_parcellation
    from .tractography.streamline_utils import StreamlineUtils

    logger = get_logger()
    logger.info("=" * 80)
    logger.info("CONNECTOME CONSTRUCTION")
    logger.info("=" * 80)

    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load streamlines
        logger.info(f"Loading streamlines from {args.streamlines}")
        streamlines_path = str(args.streamlines)

        if streamlines_path.endswith('.trk'):
            streamlines, header = StreamlineUtils.load_trk(streamlines_path)
        elif streamlines_path.endswith('.tck'):
            streamlines = StreamlineUtils.load_tck(streamlines_path)
        else:
            raise ValueError(f"Unsupported streamline format: {streamlines_path}")

        logger.info(f"Loaded {len(streamlines)} streamlines")

        # Load parcellation
        logger.info(f"Loading parcellation from {args.parcellation}")
        parcellation, labels, parcellation_affine = load_parcellation(args.parcellation)
        logger.info(f"Loaded parcellation: {np.max(parcellation) + 1} parcels")

        # Load FA map if using microstructure weighting
        microstructure_map = None
        if args.weighting in ['mean_fa', 'hybrid'] and args.fa_map:
            logger.info(f"Loading FA map from {args.fa_map}")
            fa_img = nib.load(args.fa_map)
            microstructure_map = fa_img.get_fdata()

        # Build connectome (pass affine for world→voxel coordinate conversion)
        logger.info(f"Building connectome with weighting: {args.weighting}")
        builder = ConnectomeBuilder(parcellation, parcel_labels=labels, affine=parcellation_affine)

        adjacency = builder.build_connectome(
            streamlines=streamlines,
            weighting=args.weighting,
            microstructure_map=microstructure_map,
            symmetric=True,
            normalize=False
        )

        # Save connectome as NumPy array
        logger.info(f"Saving connectome to {output_path}")
        np.save(output_path, adjacency)

        # Also save as CSV
        csv_path = output_path.with_suffix('.csv')
        np.savetxt(csv_path, adjacency, delimiter=',', fmt='%.6f')
        logger.info(f"Saved CSV to {csv_path}")

        # Save labels if available
        if labels:
            labels_path = output_path.parent / f"{output_path.stem}_labels.txt"
            with open(labels_path, 'w') as f:
                for label in labels:
                    f.write(f"{label}\n")
            logger.info(f"Saved labels to {labels_path}")

        # Compute and save edge statistics
        logger.info("Computing edge statistics...")
        edge_stats = builder.compute_edge_statistics(streamlines, adjacency)

        # Save basic connectome info
        info_path = output_path.parent / f"{output_path.stem}_info.json"
        info = {
            'n_parcels': builder.n_parcels,
            'n_edges': int(np.count_nonzero(adjacency)),
            'density': float(np.count_nonzero(adjacency) / (builder.n_parcels ** 2)),
            'weighting': args.weighting,
            'total_weight': float(np.sum(adjacency)),
            'mean_edge_weight': float(np.mean(adjacency[adjacency > 0])) if np.any(adjacency > 0) else 0.0,
            'n_streamlines': len(streamlines),
            'n_edge_pairs': len(edge_stats)
        }

        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Saved connectome info to {info_path}")

        # Print summary
        logger.info("\nConnectome Summary:")
        logger.info(f"  Number of parcels: {builder.n_parcels}")
        logger.info(f"  Number of edges: {info['n_edges']}")
        logger.info(f"  Network density: {info['density']:.3f}")
        logger.info(f"  Total weight: {info['total_weight']:.2f}")
        logger.info(f"  Mean edge weight: {info['mean_edge_weight']:.3f}")

        logger.info("\n" + "=" * 80)
        logger.info("CONNECTOME CONSTRUCTION COMPLETED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Connectome construction failed: {e}", exc_info=True)
        raise


def compute_metrics(args):
    """Compute graph metrics"""
    import numpy as np
    from pathlib import Path
    from .connectome.graph_metrics import ConnectomeMetrics

    logger = get_logger()
    logger.info("=" * 80)
    logger.info("GRAPH THEORY METRICS COMPUTATION")
    logger.info("=" * 80)

    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load connectome
        logger.info(f"Loading connectome from {args.connectome}")
        connectome_path = str(args.connectome)

        if connectome_path.endswith('.npy'):
            adjacency = np.load(connectome_path)
        elif connectome_path.endswith('.csv'):
            adjacency = np.loadtxt(connectome_path, delimiter=',')
        else:
            raise ValueError(f"Unsupported connectome format: {connectome_path}")

        logger.info(f"Loaded connectome: {adjacency.shape[0]} nodes, "
                   f"{np.count_nonzero(adjacency)} edges")

        # Load labels if available
        labels_path = Path(connectome_path).parent / f"{Path(connectome_path).stem}_labels.txt"
        labels = None
        if labels_path.exists():
            logger.info(f"Loading node labels from {labels_path}")
            with open(labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]

        # Initialize metrics calculator
        logger.info("Computing comprehensive graph metrics...")
        metrics_calc = ConnectomeMetrics(adjacency, node_labels=labels)

        # Compute all metrics
        metrics = metrics_calc.compute_all_metrics()

        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            elif isinstance(value, dict):
                metrics_serializable[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        metrics_serializable[key][k] = v.tolist()
                    elif isinstance(v, (np.integer, np.floating)):
                        metrics_serializable[key][k] = float(v)
                    else:
                        metrics_serializable[key][k] = v
            elif isinstance(value, (np.integer, np.floating)):
                metrics_serializable[key] = float(value)
            else:
                metrics_serializable[key] = value

        # Save metrics
        logger.info(f"Saving metrics to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

        # Print summary of global metrics
        logger.info("\nGlobal Network Metrics:")
        logger.info("-" * 60)
        for key, value in metrics_serializable['global'].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

        # Save human-readable summary
        summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Graph Theory Metrics Summary\n")
            f.write("=" * 70 + "\n\n")

            f.write("Network Information:\n")
            f.write(f"  Number of nodes: {adjacency.shape[0]}\n")
            f.write(f"  Number of edges: {np.count_nonzero(adjacency)}\n\n")

            f.write("Global Metrics:\n")
            f.write("-" * 70 + "\n")
            for key, value in metrics_serializable['global'].items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

            if 'communities' in metrics_serializable:
                f.write("Community Detection:\n")
                f.write("-" * 70 + "\n")
                for key, value in metrics_serializable['communities'].items():
                    if not isinstance(value, list):
                        f.write(f"  {key}: {value}\n")
                f.write("\n")

            f.write("Node-level Metrics:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Degree: min={np.min(metrics['node_degree'])}, "
                   f"max={np.max(metrics['node_degree'])}, "
                   f"mean={np.mean(metrics['node_degree']):.2f}\n")
            f.write(f"  Strength: min={np.min(metrics['node_strength']):.2f}, "
                   f"max={np.max(metrics['node_strength']):.2f}, "
                   f"mean={np.mean(metrics['node_strength']):.2f}\n")
            f.write(f"  Betweenness: min={np.min(metrics['betweenness_centrality']):.4f}, "
                   f"max={np.max(metrics['betweenness_centrality']):.4f}, "
                   f"mean={np.mean(metrics['betweenness_centrality']):.4f}\n")

        logger.info(f"Saved summary to {summary_path}")

        logger.info("\n" + "=" * 80)
        logger.info("METRICS COMPUTATION COMPLETED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Metrics computation failed: {e}", exc_info=True)
        raise


def run_analysis(args):
    """Run pathology analysis"""
    import numpy as np
    from pathlib import Path
    from datetime import datetime
    from .connectome.graph_metrics import compute_network_resilience

    logger = get_logger()
    logger.info("=" * 80)
    logger.info("PATHOLOGY DETECTION AND ANALYSIS")
    logger.info("=" * 80)

    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load connectome
        logger.info(f"Loading connectome from {args.connectome}")
        connectome_path = str(args.connectome)

        if connectome_path.endswith('.npy'):
            adjacency = np.load(connectome_path)
        elif connectome_path.endswith('.csv'):
            adjacency = np.loadtxt(connectome_path, delimiter=',')
        else:
            raise ValueError(f"Unsupported connectome format: {connectome_path}")

        logger.info(f"Loaded connectome: {adjacency.shape[0]} nodes")

        # Load metrics if provided
        metrics = None
        if args.metrics:
            logger.info(f"Loading pre-computed metrics from {args.metrics}")
            with open(args.metrics, 'r') as f:
                metrics = json.load(f)
        else:
            logger.info("Computing metrics for analysis...")
            from .connectome.graph_metrics import ConnectomeMetrics
            metrics_calc = ConnectomeMetrics(adjacency)
            metrics_raw = metrics_calc.compute_all_metrics()

            # Convert for serialization
            metrics = {}
            for key, value in metrics_raw.items():
                if isinstance(value, np.ndarray):
                    metrics[key] = value.tolist()
                elif isinstance(value, dict):
                    metrics[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            metrics[key][k] = v.tolist()
                        elif isinstance(v, (np.integer, np.floating)):
                            metrics[key][k] = float(v)
                        else:
                            metrics[key][k] = v
                else:
                    metrics[key] = value

        # =====================================================================
        # STATISTICAL ANALYSIS
        # =====================================================================
        logger.info("\nPerforming statistical analysis...")

        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'connectome_path': str(connectome_path),
            'network_size': adjacency.shape[0],
            'network_edges': int(np.count_nonzero(adjacency)),
            'analyses': {}
        }

        # 1. Network Efficiency Analysis
        logger.info("  Computing network efficiency...")
        global_efficiency = metrics.get('global', {}).get('global_efficiency', 0.0)
        analysis_results['analyses']['efficiency'] = {
            'global_efficiency': global_efficiency,
            'interpretation': (
                'High' if global_efficiency > 0.5 else
                'Moderate' if global_efficiency > 0.3 else
                'Low'
            )
        }

        # 2. Small-World Analysis
        logger.info("  Analyzing small-world properties...")
        sigma = metrics.get('global', {}).get('small_world_sigma', 0.0)
        analysis_results['analyses']['small_world'] = {
            'sigma': sigma,
            'is_small_world': sigma > 1.0,
            'interpretation': (
                'Network exhibits small-world properties' if sigma > 1.0 else
                'Network does not exhibit small-world properties'
            )
        }

        # 3. Modularity Analysis
        logger.info("  Analyzing modularity...")
        if 'communities' in metrics and 'louvain_modularity' in metrics['communities']:
            modularity = metrics['communities']['louvain_modularity']
            n_communities = len(set(metrics['communities']['louvain_partition']))

            analysis_results['analyses']['modularity'] = {
                'modularity_q': modularity,
                'n_communities': n_communities,
                'interpretation': (
                    'Strong modular structure' if modularity > 0.4 else
                    'Moderate modular structure' if modularity > 0.3 else
                    'Weak modular structure'
                )
            }

        # 4. Hub Identification
        logger.info("  Identifying network hubs...")
        if 'node_degree' in metrics and 'betweenness_centrality' in metrics:
            degree = np.array(metrics['node_degree'])
            betweenness = np.array(metrics['betweenness_centrality'])

            # Hubs: high degree AND high betweenness
            degree_threshold = np.percentile(degree, 75)
            betweenness_threshold = np.percentile(betweenness, 75)

            hub_indices = np.where(
                (degree > degree_threshold) & (betweenness > betweenness_threshold)
            )[0]

            analysis_results['analyses']['hubs'] = {
                'n_hubs': int(len(hub_indices)),
                'hub_indices': hub_indices.tolist(),
                'hub_degrees': degree[hub_indices].tolist(),
                'hub_betweenness': betweenness[hub_indices].tolist(),
                'interpretation': f'Identified {len(hub_indices)} hub nodes'
            }

        # 5. Network Resilience
        logger.info("  Computing network resilience...")
        resilience_random = compute_network_resilience(
            adjacency, attack_type='random', n_steps=10
        )
        resilience_targeted = compute_network_resilience(
            adjacency, attack_type='targeted', n_steps=10
        )

        analysis_results['analyses']['resilience'] = {
            'random_attack': {
                'efficiency_decline': resilience_random['efficiency'].tolist(),
                'final_efficiency': float(resilience_random['efficiency'][-1]),
                'efficiency_at_50pct_loss': float(resilience_random['efficiency'][5])
            },
            'targeted_attack': {
                'efficiency_decline': resilience_targeted['efficiency'].tolist(),
                'final_efficiency': float(resilience_targeted['efficiency'][-1]),
                'efficiency_at_50pct_loss': float(resilience_targeted['efficiency'][5])
            },
            'interpretation': (
                'Network shows high resilience to random failures'
                if resilience_random['efficiency'][5] > 0.5 else
                'Network shows moderate resilience to random failures'
            )
        }

        # 6. Pathology Detection (simplified heuristics)
        logger.info("  Detecting potential pathological patterns...")

        pathology_indicators = []

        # Check for abnormally low efficiency
        if global_efficiency < 0.3:
            pathology_indicators.append({
                'indicator': 'Low global efficiency',
                'value': global_efficiency,
                'severity': 'moderate',
                'description': 'Network shows reduced overall connectivity efficiency'
            })

        # Check for excessive modularity (potential network fragmentation)
        if 'modularity' in analysis_results['analyses']:
            mod_q = analysis_results['analyses']['modularity']['modularity_q']
            if mod_q > 0.6:
                pathology_indicators.append({
                    'indicator': 'Excessive modularity',
                    'value': mod_q,
                    'severity': 'low',
                    'description': 'Network shows strong segregation, possibly indicating reduced integration'
                })

        # Check for hub vulnerability
        if 'resilience' in analysis_results['analyses']:
            targeted_eff = analysis_results['analyses']['resilience']['targeted_attack']['efficiency_at_50pct_loss']
            if targeted_eff < 0.3:
                pathology_indicators.append({
                    'indicator': 'High hub vulnerability',
                    'value': targeted_eff,
                    'severity': 'moderate',
                    'description': 'Network highly vulnerable to targeted hub attacks'
                })

        analysis_results['analyses']['pathology_detection'] = {
            'n_indicators': len(pathology_indicators),
            'indicators': pathology_indicators,
            'overall_assessment': (
                'No significant pathological patterns detected' if len(pathology_indicators) == 0 else
                f'{len(pathology_indicators)} potential pathological indicators detected'
            )
        }

        # =====================================================================
        # SAVE RESULTS
        # =====================================================================
        logger.info(f"\nSaving analysis results to {output_path}")

        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)

        # Generate human-readable report
        report_path = output_path.parent / f"{output_path.stem}_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NEUROTRACT PATHOLOGY DETECTION AND ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Connectome: {connectome_path}\n")
            f.write(f"Network size: {analysis_results['network_size']} nodes, "
                   f"{analysis_results['network_edges']} edges\n\n")

            # Efficiency
            f.write("-" * 80 + "\n")
            f.write("NETWORK EFFICIENCY\n")
            f.write("-" * 80 + "\n")
            eff = analysis_results['analyses']['efficiency']
            f.write(f"Global efficiency: {eff['global_efficiency']:.3f} ({eff['interpretation']})\n\n")

            # Small-world
            f.write("-" * 80 + "\n")
            f.write("SMALL-WORLD PROPERTIES\n")
            f.write("-" * 80 + "\n")
            sw = analysis_results['analyses']['small_world']
            f.write(f"Small-world coefficient (σ): {sw['sigma']:.3f}\n")
            f.write(f"Result: {sw['interpretation']}\n\n")

            # Modularity
            if 'modularity' in analysis_results['analyses']:
                f.write("-" * 80 + "\n")
                f.write("MODULARITY\n")
                f.write("-" * 80 + "\n")
                mod = analysis_results['analyses']['modularity']
                f.write(f"Modularity Q: {mod['modularity_q']:.3f}\n")
                f.write(f"Number of communities: {mod['n_communities']}\n")
                f.write(f"Assessment: {mod['interpretation']}\n\n")

            # Hubs
            if 'hubs' in analysis_results['analyses']:
                f.write("-" * 80 + "\n")
                f.write("HUB NODES\n")
                f.write("-" * 80 + "\n")
                hubs = analysis_results['analyses']['hubs']
                f.write(f"{hubs['interpretation']}\n")
                f.write(f"Hub node indices: {hubs['hub_indices']}\n\n")

            # Resilience
            f.write("-" * 80 + "\n")
            f.write("NETWORK RESILIENCE\n")
            f.write("-" * 80 + "\n")
            res = analysis_results['analyses']['resilience']
            f.write(f"Random attack - final efficiency: {res['random_attack']['final_efficiency']:.3f}\n")
            f.write(f"Targeted attack - final efficiency: {res['targeted_attack']['final_efficiency']:.3f}\n")
            f.write(f"Assessment: {res['interpretation']}\n\n")

            # Pathology
            f.write("-" * 80 + "\n")
            f.write("PATHOLOGY DETECTION\n")
            f.write("-" * 80 + "\n")
            path = analysis_results['analyses']['pathology_detection']
            f.write(f"{path['overall_assessment']}\n\n")

            if path['n_indicators'] > 0:
                f.write("Detected indicators:\n")
                for idx, indicator in enumerate(path['indicators'], 1):
                    f.write(f"\n{idx}. {indicator['indicator']}\n")
                    f.write(f"   Value: {indicator['value']:.3f}\n")
                    f.write(f"   Severity: {indicator['severity']}\n")
                    f.write(f"   Description: {indicator['description']}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Saved report to {report_path}")

        # Print summary
        logger.info("\nAnalysis Summary:")
        logger.info(f"  Global efficiency: {analysis_results['analyses']['efficiency']['global_efficiency']:.3f}")
        logger.info(f"  Small-world σ: {analysis_results['analyses']['small_world']['sigma']:.3f}")
        logger.info(f"  Pathology indicators: {analysis_results['analyses']['pathology_detection']['n_indicators']}")

        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
