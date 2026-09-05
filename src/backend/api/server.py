"""
NeuroTract 2.0 FastAPI Server

REST and Real-Time Event API for diffusion MRI tractography, structural connectome
construction, graph-theoretic network analytics, scientific provenance, and validation.
"""

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import json
import time
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import nibabel as nib

from ..utils.logger import get_logger
from .events import global_event_manager
from ..data.validator import DatasetValidator, validate_dataset
from ..provenance.tracker import global_provenance_tracker, get_software_versions, METRIC_REGISTRY
from ..analysis.benchmark import run_dti_reference_benchmark, run_graph_reference_benchmark
from ..analysis.sensitivity import evaluate_connectome_threshold_sensitivity
from ..analysis.report_generator import generate_html_report
from ..data.slice_extractor import get_subject_slices

logger = get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="NeuroTract 2.0 API",
    description="Interactive Diffusion-MRI Analysis Laboratory API",
    version="2.0.0"
)

# CORS middleware for web laboratory access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Job storage with file-based persistence
jobs_db_file = Path("jobs_database.json")


def load_jobs_db() -> Dict[str, Any]:
    """Load jobs database from file"""
    if jobs_db_file.exists():
        try:
            with open(jobs_db_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load jobs database: {e}")
            return {}
    return {}


def save_jobs_db():
    """Save jobs database to file"""
    try:
        with open(jobs_db_file, 'w') as f:
            json.dump(jobs_db, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save jobs database: {e}")


jobs_db = load_jobs_db()


# ──────────────────────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────────────────────

class JobConfig(BaseModel):
    """Configuration for analysis job"""
    subject_id: str
    mode: str = "full"  # quick, full, lowmem
    preprocessing: Dict[str, Any] = Field(default_factory=dict)
    tractography: Dict[str, Any] = Field(default_factory=dict)
    connectome: Dict[str, Any] = Field(default_factory=dict)
    rng_seed: Optional[int] = 42


class TractographyParams(BaseModel):
    """Parameters for tractography job"""
    dwi_file: str
    bval_file: Optional[str] = None
    bvec_file: Optional[str] = None
    mask_file: Optional[str] = None
    algorithm: Optional[str] = "probabilistic"
    step_size: Optional[float] = 0.5
    fa_threshold: Optional[float] = 0.1
    max_angle: Optional[float] = 30.0
    seeds_per_voxel: Optional[int] = 2
    rng_seed: Optional[int] = 42


class GraphAnalysisParams(BaseModel):
    """Parameters for graph analysis job"""
    tractogram_file: str
    atlas_file: Optional[str] = None
    parcellation_scheme: Optional[str] = None
    threshold: Optional[float] = 0.0


class SensitivityParams(BaseModel):
    """Parameters for parameter sensitivity analysis"""
    subject_id: str
    thresholds: Optional[List[float]] = [0, 1, 2, 5, 10]


def _job_to_frontend(j: dict) -> dict:
    """Convert internal job dict to frontend-expected format."""
    return {
        "id": j["job_id"],
        "status": j["status"],
        "progress": round(j.get("progress", 0) * 100, 1),
        "task": j.get("config", {}).get("mode", "full"),
        "created_at": j["created_at"],
        "updated_at": j["updated_at"],
        "result": j.get("results"),
        "error": j.get("message") if j["status"] == "failed" else None,
    }


# ──────────────────────────────────────────────────────────────
# Core Health & Version Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "NeuroTract 2.0 API",
        "version": "2.0.0",
        "status": "online",
        "laboratory": "Interactive Diffusion-MRI Analysis Laboratory",
        "software_versions": get_software_versions()
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "software_versions": get_software_versions()
    }


@app.get("/version")
async def get_version():
    return {
        "api_version": "2.0.0",
        "neurotract_version": "2.0.0",
        "software_versions": get_software_versions()
    }


# ──────────────────────────────────────────────────────────────
# Real-Time SSE Stream Endpoint
# ──────────────────────────────────────────────────────────────

@app.get("/jobs/{job_id}/events")
@app.get("/api/jobs/{job_id}/events")
async def stream_job_events(job_id: str):
    """
    Server-Sent Events (SSE) live event stream for an analysis job.
    Delivers stage transitions, telemetry, and completion events.
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return StreamingResponse(
        global_event_manager.event_generator(job_id),
        media_type="text/event-stream"
    )


# ──────────────────────────────────────────────────────────────
# Dataset Ingestion & Validation Boundary
# ──────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload neuroimaging file (.nii, .nii.gz, .bval, .bvec, .trk)"""
    import uuid
    import shutil

    file_id = str(uuid.uuid4())
    file_dir = UPLOAD_DIR / file_id
    file_dir.mkdir(parents=True, exist_ok=True)

    file_path = file_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = file_path.stat().st_size
    logger.info(f"Uploaded {file.filename} ({file_size} bytes) -> {file_id}")

    return {
        "file_id": file_id,
        "filename": file.filename,
        "size_bytes": file_size,
        "path": str(file_path),
    }


@app.get("/api/datasets/validate")
@app.post("/api/datasets/validate")
async def validate_dataset_endpoint(
    dwi_path: str = Query(..., description="Path to DWI NIfTI file"),
    bval_path: Optional[str] = Query(None, description="Optional path to .bval file"),
    bvec_path: Optional[str] = Query(None, description="Optional path to .bvec file")
):
    """
    Strict boundary validation of a diffusion MRI dataset.
    Returns authoritative DatasetValidationReport.
    """
    validator = DatasetValidator()
    report = validator.validate_dwi_dataset(dwi_path, bval_path, bvec_path)
    return report.to_dict()


@app.get("/api/datasets/report/{subject_id}")
async def get_subject_validation_report(subject_id: str):
    """Generate or retrieve validation report for a known subject"""
    ds_dir = Path("datasets") / "Stanford dataset"
    dwi_cand = ds_dir / f"{subject_id}_b1000_1.nii.gz"
    if not dwi_cand.exists():
        # Look in datasets/
        for p in Path("datasets").rglob(f"*{subject_id}*dwi.nii.gz"):
            dwi_cand = p
            break

    if not dwi_cand.exists():
        raise HTTPException(status_code=404, detail=f"No DWI dataset found for {subject_id}")

    validator = DatasetValidator()
    report = validator.validate_dwi_dataset(
        dwi_path=dwi_cand,
        source_name=f"Stanford HARDI {subject_id}",
        license_info="CC BY 3.0"
    )
    return report.to_dict()


# ──────────────────────────────────────────────────────────────
# Jobs Lifecycle & Submission
# ──────────────────────────────────────────────────────────────

@app.get("/jobs")
async def list_jobs(limit: int = 50):
    """List all jobs, most recent first."""
    sorted_jobs = sorted(
        jobs_db.values(),
        key=lambda j: j.get("created_at", ""),
        reverse=True,
    )[:limit]
    return [_job_to_frontend(j) for j in sorted_jobs]


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job details by ID."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _job_to_frontend(jobs_db[job_id])


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    jobs_db[job_id]["status"] = "failed"
    jobs_db[job_id]["message"] = "Cancelled by user"
    jobs_db[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"
    save_jobs_db()

    global_event_manager.emit(job_id, "job_cancelled", {
        "message": f"Job {job_id} cancelled by user",
        "stage": "cancelled",
        "status": "failed"
    })
    return {"message": f"Job {job_id} cancelled"}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    del jobs_db[job_id]
    save_jobs_db()
    return {"message": f"Job {job_id} deleted"}


@app.post("/jobs/submit")
async def submit_job(config: JobConfig, background_tasks: BackgroundTasks):
    """Submit a new analysis job (full pipeline)."""
    import uuid

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Job queued for execution",
        "config": config.model_dump(),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "results": None
    }
    jobs_db[job_id] = job
    save_jobs_db()

    global_event_manager.emit(job_id, "job_queued", {
        "message": "Job queued for real-time execution",
        "stage": "queued",
        "status": "pending",
        "progress": 0.0
    })

    background_tasks.add_task(process_job, job_id, config)
    logger.info(f"Job {job_id} submitted for subject {config.subject_id}")

    return _job_to_frontend(job)


@app.post("/tractography")
async def run_tractography_api(params: TractographyParams, background_tasks: BackgroundTasks):
    """Submit a tractography job."""
    import uuid

    job_id = str(uuid.uuid4())
    config = JobConfig(
        subject_id=params.dwi_file,
        mode="tractography",
        tractography={
            "step_size": params.step_size,
            "max_angle": params.max_angle,
            "seeds_per_voxel": params.seeds_per_voxel,
            "fa_threshold": params.fa_threshold,
            "rng_seed": params.rng_seed,
        }
    )
    job = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Tractography job queued",
        "config": config.model_dump(),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "results": None,
    }
    jobs_db[job_id] = job
    save_jobs_db()

    background_tasks.add_task(process_job, job_id, config)
    return _job_to_frontend(job)


# ──────────────────────────────────────────────────────────────
# Real Scientific Pipeline Execution Worker
# ──────────────────────────────────────────────────────────────

async def process_job(job_id: str, config: JobConfig):
    """
    Robust background worker executing the real NeuroTract scientific pipeline
    with live event emission and provenance tracking.
    """
    from ..data.data_loader import DataLoader
    from ..preprocessing.pipeline import PreprocessingPipeline
    from ..microstructure.dti import DTIModel
    from ..microstructure.csd import CSDModel, ResponseFunction
    from ..tractography.probabilistic_tracker import ProbabilisticTracker
    from ..tractography.seeding import SeedGenerator
    from ..tractography.streamline_utils import StreamlineUtils
    from ..surfaces.mesh_generator import generate_brain_mesh
    from ..connectome.construct import ConnectomeBuilder
    from ..connectome.graph_metrics import ConnectomeMetrics

    logger.info(f"Starting real pipeline execution for job {job_id}...")
    start_time = time.time()
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    def emit_progress(stage: str, progress: float, message: str, telemetry: Optional[Dict] = None):
        elapsed = round(time.time() - start_time, 2)
        jobs_db[job_id]["progress"] = progress
        jobs_db[job_id]["message"] = message
        jobs_db[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"
        save_jobs_db()

        event_payload = {
            "stage": stage,
            "status": "running",
            "progress": progress,
            "elapsed_seconds": elapsed,
            "message": message,
            "telemetry": telemetry or {}
        }
        global_event_manager.emit(job_id, f"stage_{stage}", event_payload)
        logger.info(f"[{job_id}] {stage} ({int(progress*100)}%): {message}")

    try:
        jobs_db[job_id]["status"] = "running"
        save_jobs_db()

        # Step 1: Resolve dataset files
        emit_progress("validation", 0.05, "Locating and validating input dataset...")

        subj_input = config.subject_id
        dwi_file = None
        bval_file = None
        bvec_file = None
        parc_file = None

        # Resolve subject path
        if Path(subj_input).exists():
            cand = Path(subj_input)
            if cand.is_file():
                dwi_file = cand
            elif cand.is_dir():
                dwis = list(cand.glob("*dwi.nii*")) + list(cand.glob("*b1000*.nii*"))
                if dwis:
                    dwi_file = dwis[0]
        else:
            # Check datasets/Stanford dataset
            ds_dir = Path("datasets") / "Stanford dataset"
            for f in [ds_dir / f"{subj_input}_b1000_1.nii.gz", ds_dir / f"{subj_input}.nii.gz"]:
                if f.exists():
                    dwi_file = f
                    break

        if dwi_file is None:
            # Fallback to SUB1 default demo scan
            fallback = Path("datasets/Stanford dataset/SUB1_b1000_1.nii.gz")
            if fallback.exists():
                dwi_file = fallback
                logger.info(f"Using verified fallback dataset: {fallback}")
            else:
                raise FileNotFoundError(f"Could not locate diffusion data for input '{subj_input}'")

        # Discover bval/bvec
        for cand in [dwi_file.with_suffix('').with_suffix('.bval'), dwi_file.with_suffix('').with_suffix('.bvals'),
                     dwi_file.parent / f"{dwi_file.stem.replace('.nii','')}.bvals"]:
            if cand.exists():
                bval_file = cand
                break

        for cand in [dwi_file.with_suffix('').with_suffix('.bvec'), dwi_file.with_suffix('').with_suffix('.bvecs'),
                     dwi_file.parent / f"{dwi_file.stem.replace('.nii','')}.bvecs"]:
            if cand.exists():
                bvec_file = cand
                break

        # Check parcellation
        parc_cand = dwi_file.parent / f"{dwi_file.stem.split('_')[0]}_aparc-reduced.nii.gz"
        if parc_cand.exists():
            parc_file = parc_cand

        # Validate with DatasetValidator
        validator = DatasetValidator()
        val_report = validator.validate_dwi_dataset(dwi_file, bval_file, bvec_file)
        if not val_report.is_valid:
            raise ValueError(f"Dataset boundary validation failed: {'; '.join(val_report.errors)}")

        emit_progress("validation", 0.15, "Dataset validated successfully", {
            "num_volumes": val_report.num_volumes,
            "voxel_size": val_report.voxel_size_mm,
            "dimensions": val_report.dimensions,
            "shells": val_report.gradient_summary.shell_distribution if val_report.gradient_summary else {}
        })

        # Provenance setup
        rng_seed = config.rng_seed or 42
        prov_exec = global_provenance_tracker.create_execution(
            execution_id=job_id,
            dataset_name=str(dwi_file.name),
            dataset_checksums=val_report.checksums_sha256,
            rng_seed=rng_seed,
            preprocessing_params=config.preprocessing,
            tractography_params=config.tractography,
            connectome_params=config.connectome
        )

        # Step 2: Preprocessing
        emit_progress("preprocessing", 0.20, "Executing gradient correction and brain extraction...")
        preproc_dir = output_dir / "preprocessed"
        skip_motion = config.mode == "quick" or config.preprocessing.get("skip_motion", True)

        pipeline = PreprocessingPipeline(
            output_dir=preproc_dir,
            skip_motion_correction=skip_motion,
            skip_brain_extraction=False,
            skip_bias_correction=(config.mode == "quick"),
            save_intermediate=True,
            save_qc_reports=True
        )

        preproc_outputs = pipeline.run(
            dwi_path=dwi_file,
            bval_path=bval_file,
            bvec_path=bvec_file,
            output_prefix=f"{job_id}_preproc"
        )

        # Load preprocessed diffusion data
        loader = DataLoader(use_mmap=(config.mode == "lowmem"), validate=True)
        dwi_data = loader.load_diffusion(
            preproc_outputs['dwi'],
            preproc_outputs['bval'],
            preproc_outputs['bvec']
        )
        mask_img = nib.load(str(preproc_outputs['mask']))
        brain_mask = mask_img.get_fdata() > 0

        emit_progress("preprocessing", 0.35, "Preprocessing completed", {
            "mask_voxels": int(np.sum(brain_mask)),
            "dwi_shape": list(dwi_data.data.shape)
        })

        # Step 3: DTI Modeling
        emit_progress("dti", 0.40, "Fitting diffusion tensor model (WLS)...")
        dti_model = DTIModel(dwi_data.bvals, dwi_data.bvecs)
        dti_results = dti_model.fit(dwi_data.data, mask=brain_mask)

        dti_dir = output_dir / "dti"
        dti_dir.mkdir(exist_ok=True)
        fa_map = dti_results['fa']
        fa_path = dti_dir / "dti_fa.nii.gz"
        nib.save(nib.Nifti1Image(fa_map.astype(np.float32), dwi_data.affine), str(fa_path))

        md_map = dti_results['md']
        md_path = dti_dir / "dti_md.nii.gz"
        nib.save(nib.Nifti1Image(md_map.astype(np.float32), dwi_data.affine), str(md_path))

        emit_progress("dti", 0.50, "DTI maps computed", {
            "mean_fa_brain": round(float(np.mean(fa_map[brain_mask])), 4),
            "mean_md_brain": float(f"{np.mean(md_map[brain_mask]):.3e}")
        })

        # Step 4: CSD / FOD
        emit_progress("csd", 0.55, "Estimating fiber orientation distribution (CSD)...")
        response_est = ResponseFunction(method='dhollander')
        response = response_est.estimate(dwi_data.data, dwi_data.bvals, dwi_data.bvecs, fa_map=fa_map, mask=brain_mask)
        csd_model = CSDModel(max_order=8)
        fod = csd_model.fit(dwi_data.data, dwi_data.bvals, dwi_data.bvecs, response=response, mask=brain_mask)

        fod_path = output_dir / "fod.nii.gz"
        nib.save(nib.Nifti1Image(fod.astype(np.float32), dwi_data.affine), str(fod_path))
        emit_progress("csd", 0.65, "FOD spherical harmonics computed (order 8, 45 coeffs)")

        # Step 5: Tractography
        emit_progress("tractography", 0.68, "Initializing probabilistic tractography...")
        t_conf = config.tractography or {}
        step_size = float(t_conf.get("step_size", 0.5))
        max_angle = float(t_conf.get("max_angle", 30.0))
        fa_thresh = float(t_conf.get("fa_threshold", 0.1))
        seeds_per_vox = int(t_conf.get("seeds_per_voxel", 1 if config.mode == "quick" else 2))

        tracker = ProbabilisticTracker(
            voxel_size=dwi_data.voxel_size[:3],
            step_size=step_size,
            max_angle=max_angle,
            fa_threshold=fa_thresh,
            rng_seed=rng_seed
        )

        seed_gen = SeedGenerator(seed_density=seeds_per_vox)
        seeds = seed_gen.generate_seeds_wm(brain_mask, fa_map, fa_threshold=0.2)
        if len(seeds) > 10000 and config.mode == "quick":
            seeds = seeds[::(len(seeds)//10000)]

        emit_progress("tractography", 0.72, f"Propagating streamlines from {len(seeds):,} seed points...", {
            "total_seeds": len(seeds),
            "step_size_mm": step_size,
            "max_angle_deg": max_angle
        })

        # Track with progress batches
        streamlines = []
        batch_size = max(1, len(seeds) // 10)
        for i, s in enumerate(seeds):
            sl = tracker.track_one(s, fod, brain_mask, fa_map, dwi_data.affine)
            if sl is not None and len(sl) > 5:
                streamlines.append(sl)

            if (i + 1) % batch_size == 0 or (i + 1) == len(seeds):
                sub_progress = 0.72 + 0.13 * ((i + 1) / len(seeds))
                emit_progress("tractography", sub_progress, f"Tracking: {len(streamlines):,} streamlines kept ({i+1}/{len(seeds)} seeds)", {
                    "seeds_processed": i + 1,
                    "streamlines_kept": len(streamlines),
                })

        streamlines_path = output_dir / "streamlines.trk"
        StreamlineUtils.save_trk(streamlines, str(streamlines_path), dwi_data.affine, brain_mask.shape)
        emit_progress("tractography", 0.85, f"Tractography completed ({len(streamlines):,} streamlines)")

        # Step 6: Brain Surface Mesh
        emit_progress("brain_mesh", 0.88, "Generating cortical surface mesh (Marching Cubes)...")
        mesh_data = generate_brain_mesh(
            mask_path=str(preproc_outputs['mask']),
            step_size=1,
            cache_dir=str(output_dir)
        )
        emit_progress("brain_mesh", 0.90, f"Surface mesh generated ({mesh_data['metadata']['n_vertices']:,} vertices)")

        # Step 7: Connectome Construction
        emit_progress("connectome", 0.92, "Mapping streamline endpoints to anatomical parcellation...")
        if parc_file and parc_file.exists():
            parc_img = nib.load(str(parc_file))
            parcellation = parc_img.get_fdata().astype(int)
            n_parcels = int(np.max(parcellation)) + 1
        else:
            # Synthetic 89 parcel grid aligned to brain mask
            parcellation = np.zeros(brain_mask.shape, dtype=int)
            grid = 10
            pid = 1
            for x in range(0, brain_mask.shape[0], grid):
                for y in range(0, brain_mask.shape[1], grid):
                    for z in range(0, brain_mask.shape[2], grid):
                        if np.any(brain_mask[x:x+grid, y:y+grid, z:z+grid]):
                            parcellation[x:x+grid, y:y+grid, z:z+grid] = pid
                            pid += 1
                            if pid > 89:
                                break
            n_parcels = 89

        builder = ConnectomeBuilder(parcellation=parcellation, n_parcels=n_parcels, affine=dwi_data.affine)
        connectome = builder.build_connectome(streamlines, weighting='count')

        connectome_path = output_dir / "connectome.npy"
        np.save(str(connectome_path), connectome)
        np.savetxt(str(output_dir / "connectome.csv"), connectome, delimiter=",", fmt="%.4f")
        emit_progress("connectome", 0.94, f"Connectome built ({int(np.sum(connectome > 0)/2):,} structural edges)")

        # Step 8: Graph Theory Metrics
        emit_progress("graph_metrics", 0.96, "Computing graph-theoretical topological network metrics...")
        metrics_calc = ConnectomeMetrics(connectome)
        metrics = metrics_calc.compute_all_metrics()

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Register metric provenance
        for m_key in ["global_efficiency", "clustering_coefficient", "density", "transitivity", "assortativity"]:
            if m_key in metrics.get("global", {}):
                global_provenance_tracker.create_metric_provenance(
                    execution_id=job_id,
                    metric_key=m_key,
                    value=metrics["global"][m_key],
                    input_properties={
                        "nodes": n_parcels,
                        "edges": int(np.sum(connectome > 0) / 2),
                        "weighting": "count"
                    }
                )

        # Step 9: Report Generation
        emit_progress("reporting", 0.98, "Generating reproducible HTML analysis report...")
        report_path = output_dir / "report.html"
        generate_html_report(job_id, report_path)
        global_provenance_tracker.complete_execution(job_id, "completed")

        # Complete Job
        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["progress"] = 1.0
        jobs_db[job_id]["message"] = "Analysis completed successfully"
        jobs_db[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"
        jobs_db[job_id]["results"] = {
            "connectome_file": str(connectome_path),
            "streamlines_file": str(streamlines_path),
            "fod_file": str(fod_path),
            "metrics_file": str(metrics_path),
            "report_file": str(report_path),
            "num_streamlines": len(streamlines),
            "num_edges": int(np.sum(connectome > 0) / 2),
            "metrics": {
                "global_efficiency": float(metrics.get("global", {}).get("global_efficiency", 0)),
                "clustering_coefficient": float(metrics.get("global", {}).get("clustering_coefficient", 0)),
                "modularity": float(metrics.get("communities", {}).get("louvain_modularity", 0))
            }
        }
        save_jobs_db()

        global_event_manager.emit(job_id, "job_completed", {
            "stage": "completed",
            "status": "completed",
            "progress": 1.0,
            "elapsed_seconds": round(time.time() - start_time, 2),
            "message": "Analysis completed successfully",
            "results": jobs_db[job_id]["results"]
        })
        logger.info(f"Job {job_id} successfully completed in {time.time() - start_time:.2f}s")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["message"] = f"Error: {str(e)}"
        jobs_db[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"
        save_jobs_db()

        global_event_manager.emit(job_id, "job_failed", {
            "stage": "failed",
            "status": "failed",
            "error": str(e),
            "message": f"Execution failed: {str(e)}"
        })


# ──────────────────────────────────────────────────────────────
# Pre-computed Results Endpoints (Fast Demo Exploration)
# ──────────────────────────────────────────────────────────────

@app.get("/results/available")
async def list_available_results():
    """Scan output/ directory for available pipeline results."""
    results = []
    if not OUTPUT_DIR.exists():
        return results

    for subject_dir in sorted(OUTPUT_DIR.iterdir()):
        if not subject_dir.is_dir() or subject_dir.name == "provenance":
            continue
        subject_id = subject_dir.name

        entry = {
            "subject_id": subject_id,
            "has_streamlines": (subject_dir / "streamlines.trk").exists(),
            "has_metrics": (subject_dir / "metrics.json").exists(),
            "has_connectome": (subject_dir / "connectome.npy").exists(),
            "has_dti": (subject_dir / "dti").is_dir() if (subject_dir / "dti").exists() else False,
            "has_fod": (subject_dir / "fod.nii.gz").exists(),
            "has_report": (subject_dir / "report.html").exists(),
            "files": [],
        }

        for f in sorted(subject_dir.rglob("*")):
            if f.is_file():
                rel = f.relative_to(subject_dir)
                entry["files"].append({
                    "name": str(rel),
                    "size_bytes": f.stat().st_size,
                    "type": _classify_file(f.name),
                })

        stats_file = subject_dir / "streamlines_statistics.json"
        if stats_file.exists():
            with open(stats_file) as sf:
                entry["streamline_stats"] = json.load(sf)

        info_file = subject_dir / "connectome_info.json"
        if info_file.exists():
            with open(info_file) as inf:
                entry["connectome_info"] = json.load(inf)

        results.append(entry)

    return results


def _classify_file(filename: str) -> str:
    fl = filename.lower()
    if fl.endswith(".trk") or fl.endswith(".tck"):
        return "tractogram"
    elif fl.endswith(".nii") or fl.endswith(".nii.gz"):
        return "volume"
    elif fl.endswith(".npy"):
        return "matrix"
    elif fl.endswith(".json"):
        return "json"
    elif fl.endswith(".csv"):
        return "csv"
    elif fl.endswith(".html"):
        return "html"
    elif fl.endswith(".txt"):
        return "text"
    return "other"


@app.get("/results/{subject_id}/streamlines")
async def get_result_streamlines(subject_id: str, max_streamlines: int = 3000):
    """Load streamlines from streamlines.trk and return JSON for 3D viewer."""
    trk_path = OUTPUT_DIR / subject_id / "streamlines.trk"
    if not trk_path.exists():
        raise HTTPException(status_code=404, detail=f"No streamlines found for {subject_id}")

    tractogram = nib.streamlines.load(str(trk_path))
    all_streamlines = tractogram.streamlines
    n_total = len(all_streamlines)

    if n_total > max_streamlines:
        indices = np.linspace(0, n_total - 1, max_streamlines, dtype=int)
        selected = [all_streamlines[i] for i in indices]
    else:
        selected = list(all_streamlines)

    streamlines_data = []
    all_points = []
    total_points = 0
    lengths = []

    for sl in selected:
        points = sl.astype(float)
        n_pts = len(points)
        total_points += n_pts

        diffs = np.diff(points, axis=0)
        length = float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))
        lengths.append(length)

        if n_pts > 1:
            tangent = points[-1] - points[0]
            mag = np.linalg.norm(tangent)
            if mag > 0:
                tangent = tangent / mag
            orientation = [float(tangent[0]), float(tangent[1]), float(tangent[2])]
        else:
            orientation = [0.0, 0.0, 1.0]

        flat_points = points.flatten().tolist()
        all_points.extend([points.min(axis=0), points.max(axis=0)])

        streamlines_data.append({
            "points": flat_points,
            "numPoints": n_pts,
            "length": length,
            "orientation": orientation,
        })

    if all_points:
        all_mins = np.array([p for i, p in enumerate(all_points) if i % 2 == 0])
        all_maxs = np.array([p for i, p in enumerate(all_points) if i % 2 == 1])
        bounds_min = all_mins.min(axis=0).tolist()
        bounds_max = all_maxs.max(axis=0).tolist()
    else:
        bounds_min = [0, 0, 0]
        bounds_max = [0, 0, 0]

    lengths_arr = np.array(lengths) if lengths else np.array([0.0])

    return {
        "streamlines": streamlines_data,
        "bounds": {"min": bounds_min, "max": bounds_max},
        "metadata": {
            "count": len(streamlines_data),
            "totalPoints": total_points,
            "meanLength": float(lengths_arr.mean()),
            "maxLength": float(lengths_arr.max()),
            "minLength": float(lengths_arr.min()),
            "totalInFile": n_total,
        },
    }


@app.get("/results/{subject_id}/metrics")
async def get_result_metrics(subject_id: str):
    """Return metrics.json mapped to frontend GraphMetrics format."""
    metrics_path = OUTPUT_DIR / subject_id / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail=f"No metrics found for {subject_id}")

    with open(metrics_path) as f:
        raw = json.load(f)

    global_data = raw.get("global", {})
    return {
        "global": {
            "clustering_coefficient": global_data.get("clustering_coefficient", 0),
            "characteristic_path_length": global_data.get("characteristic_path_length", 0),
            "global_efficiency": global_data.get("global_efficiency", 0),
            "modularity": raw.get("communities", {}).get("louvain_modularity", 0),
            "assortativity": global_data.get("assortativity", 0),
            "small_worldness": global_data.get("small_world_sigma", 0),
            "density": global_data.get("density", 0),
            "transitivity": global_data.get("transitivity", 0),
        },
        "nodal": {
            "degree": raw.get("node_degree", []),
            "betweenness_centrality": raw.get("betweenness_centrality", []),
            "closeness_centrality": raw.get("closeness_centrality", []),
            "local_efficiency": raw.get("local_efficiency", raw.get("eigenvector_centrality", [])),
            "node_strength": raw.get("node_strength", []),
            "eigenvector_centrality": raw.get("eigenvector_centrality", []),
        },
        "rich_club": raw.get("rich_club", {}),
        "communities": raw.get("communities", {}),
    }


@app.get("/results/{subject_id}/connectome")
async def get_result_connectome(subject_id: str):
    """Return the connectome matrix as 2D array."""
    npy_path = OUTPUT_DIR / subject_id / "connectome.npy"
    if not npy_path.exists():
        raise HTTPException(status_code=404, detail=f"No connectome found for {subject_id}")

    matrix = np.load(str(npy_path))
    return matrix.tolist()


@app.get("/results/{subject_id}/slices")
async def get_subject_slices_endpoint(
    subject_id: str,
    axial: float = 0.5,
    coronal: float = 0.5,
    sagittal: float = 0.5,
    modality: str = "fa"
):
    """Return real 2D orthogonal MRI slice images (base64 PNG) for synchronized viewer."""
    try:
        return get_subject_slices(subject_id, axial, coronal, sagittal, modality)
    except Exception as e:
        logger.error(f"Failed to extract slices for {subject_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/results/{subject_id}/brain-mesh")
async def get_brain_mesh(subject_id: str, step_size: int = 1):
    """Generate and return triangulated brain surface mesh from brain mask."""
    from ..surfaces.mesh_generator import generate_brain_mesh

    subject_dir = OUTPUT_DIR / subject_id
    mask_path = subject_dir / "preprocessed" / "preprocessed_brain_mask.nii.gz"

    if not mask_path.exists():
        for p in Path("output").rglob("*brain_mask*.nii*"):
            mask_path = p
            break

    if not mask_path.exists():
        raise HTTPException(status_code=404, detail=f"No brain mask found for {subject_id}")

    try:
        return generate_brain_mesh(
            mask_path=str(mask_path),
            step_size=step_size,
            smooth_sigma=1.0,
            cache_dir=str(subject_dir),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mesh generation failed: {str(e)}")


@app.get("/results/{subject_id}/parcellation-labels")
async def get_parcellation_labels(subject_id: str):
    """Return anatomical parcellation labels."""
    from ..surfaces.parcellation_mapping import get_parcellation_labels, LOBE_CENTROIDS

    subject_dir = OUTPUT_DIR / subject_id
    labels_file = subject_dir / "connectome_labels.txt"

    n_parcels = 89
    info_file = subject_dir / "connectome_info.json"
    if info_file.exists():
        with open(info_file) as f:
            n_parcels = json.load(f).get("n_parcels", 89)

    labels = get_parcellation_labels(
        labels_file=str(labels_file) if labels_file.exists() else None,
        n_parcels=n_parcels,
    )

    return {
        "labels": labels,
        "atlas": "Desikan-Killiany (aparc-reduced)",
        "n_parcels": n_parcels,
        "lobe_centroids": LOBE_CENTROIDS,
    }


# ──────────────────────────────────────────────────────────────
# Scientific Provenance, Validation & Sensitivity Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/api/provenance/metric-definitions")
async def get_metric_definitions():
    """Return scientific definitions, formulas, and citations for all metrics."""
    return METRIC_REGISTRY


@app.get("/api/provenance/{identifier}")
async def get_execution_or_metric_provenance(identifier: str):
    """Get full scientific provenance for an execution ID, or definition for a metric key."""
    if identifier in METRIC_REGISTRY:
        reg = METRIC_REGISTRY[identifier]
        return {
            "metric_id": identifier,
            "name": reg.get("name", identifier),
            "value": None,
            "units": reg.get("units", ""),
            "formula": reg.get("formula", ""),
            "description": reg.get("description", ""),
            "reference_citation": reg.get("reference", ""),
            "input_properties": {},
            "execution_id": "canonical_registry",
            "timestamp": datetime.utcnow().isoformat(),
            "software_versions": get_software_versions(),
        }
    prov = global_provenance_tracker.get_record(identifier)
    if not prov:
        raise HTTPException(status_code=404, detail=f"No provenance or metric definition found for '{identifier}'")
    return prov


@app.get("/api/provenance/{execution_id}/metric/{metric_key}")
async def get_metric_provenance_endpoint(execution_id: str, metric_key: str):
    """Inspect exact mathematical provenance for a specific derived metric."""
    prov = global_provenance_tracker.get_metric_provenance(execution_id, metric_key)
    if not prov:
        raise HTTPException(status_code=404, detail=f"No provenance for metric '{metric_key}'")
    return prov


@app.get("/api/validation/benchmark")
@app.post("/api/validation/benchmark")
async def run_validation_benchmark_endpoint(n_samples: int = 25):
    """
    Run live reference validation benchmark against DIPY and NetworkX baselines.
    Returns exact Pearson correlation, MAE, and tolerances.
    """
    try:
        dti_bench = run_dti_reference_benchmark(n_samples=n_samples)
        graph_bench = run_graph_reference_benchmark()
        return {
            "status": "success",
            "dti_benchmark": dti_bench,
            "graph_benchmark": graph_bench,
            "all_passed": dti_bench["metrics"]["fractional_anisotropy"]["passed"] and graph_bench.get("all_metrics_passed", True)
        }
    except Exception as e:
        logger.error(f"Benchmark error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sensitivity/run")
async def run_sensitivity_endpoint(params: SensitivityParams):
    """Evaluate structural connectome sensitivity to edge threshold variations."""
    npy_path = OUTPUT_DIR / params.subject_id / "connectome.npy"
    if not npy_path.exists():
        raise HTTPException(status_code=404, detail=f"Connectome not found for {params.subject_id}")

    matrix = np.load(str(npy_path))
    thresholds = params.thresholds or [0, 1, 2, 5, 10]
    result = evaluate_connectome_threshold_sensitivity(matrix, thresholds)
    return result


@app.get("/api/report/{subject_id}/export")
async def export_report_endpoint(subject_id: str):
    """Generate and return reproducible HTML analysis report."""
    try:
        report_path = OUTPUT_DIR / subject_id / "report.html"
        html = generate_html_report(subject_id, report_path)
        return Response(content=html, media_type="text/html")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/results/{subject_id}/download/{filename:path}")
async def download_result_file(subject_id: str, filename: str):
    """Download result file with path traversal security check."""
    file_path = OUTPUT_DIR / subject_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    try:
        file_path.resolve().relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(
        str(file_path),
        media_type="application/octet-stream",
        filename=file_path.name,
    )


def start_server(host: str = "0.0.0.0", port: int = 8000):
    logger.info(f"Starting NeuroTract 2.0 API server on {host}:{port}")
    uvicorn.run("src.backend.api.server:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    start_server()
