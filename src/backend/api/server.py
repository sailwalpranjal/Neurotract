"""
NeuroTract FastAPI Server

REST API for tractography and connectivity analysis.
"""

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import json
from pathlib import Path
import logging
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="NeuroTract API",
    description="Brain White Matter Tractography & Connectivity Analysis API",
    version="0.1.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Job storage with file-based persistence
jobs_db_file = Path("jobs_database.json")

def load_jobs_db():
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


class JobConfig(BaseModel):
    """Configuration for analysis job"""
    subject_id: str
    mode: str = "full"  # quick, full, lowmem
    preprocessing: Dict = {}
    tractography: Dict = {}
    connectome: Dict = {}


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


class GraphAnalysisParams(BaseModel):
    """Parameters for graph analysis job"""
    tractogram_file: str
    atlas_file: Optional[str] = None
    parcellation_scheme: Optional[str] = None
    threshold: Optional[float] = 0.0


def _job_to_frontend(j: dict) -> dict:
    """Convert internal job dict to frontend-expected format."""
    return {
        "id": j["job_id"],
        "status": j["status"],
        "progress": j.get("progress", 0) * 100,
        "task": j.get("config", {}).get("mode", "full"),
        "created_at": j["created_at"],
        "updated_at": j["updated_at"],
        "result": j.get("results"),
        "error": j["message"] if j["status"] == "failed" else None,
    }


# ──────────────────────────────────────────────────────────────
# Core endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "NeuroTract API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/version")
async def get_version():
    """Get API version"""
    return {
        "api_version": "0.1.0",
        "neurotract_version": "0.1.0"
    }


# ──────────────────────────────────────────────────────────────
# File upload  (frontend calls POST /upload)
# ──────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a neuroimaging file (.nii, .nii.gz, .trk, .tck, .dcm).
    Returns file_id + filename for use in subsequent pipeline calls.
    """
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


@app.post("/upload/dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_type: str = "dwi"
):
    """Upload dataset file (legacy endpoint)."""
    import shutil

    temp_dir = Path("uploads") / "datasets"
    temp_dir.mkdir(parents=True, exist_ok=True)

    file_path = temp_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"Uploaded {file.filename} ({dataset_type}) to {file_path}")

    return {
        "filename": file.filename,
        "file_path": str(file_path),
        "dataset_type": dataset_type,
        "size_bytes": file_path.stat().st_size,
        "message": "File uploaded successfully"
    }


# ──────────────────────────────────────────────────────────────
# Jobs  (frontend calls GET /jobs, GET /jobs/{id}, POST /jobs/{id}/cancel)
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
    jobs_db[job_id]["updated_at"] = datetime.now().isoformat()
    save_jobs_db()
    return {"message": f"Job {job_id} cancelled"}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    import os
    job = jobs_db[job_id]
    if job.get('results'):
        for key, val in job['results'].items():
            if isinstance(val, str) and os.path.exists(val):
                try:
                    os.remove(val)
                except Exception:
                    pass

    del jobs_db[job_id]
    save_jobs_db()
    return {"message": f"Job {job_id} deleted"}


# Legacy endpoint (kept for backwards compatibility)
@app.post("/jobs/submit")
async def submit_job(config: JobConfig, background_tasks: BackgroundTasks):
    """Submit a new analysis job (full pipeline)."""
    import uuid

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Job queued",
        "config": config.model_dump(),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "results": None
    }
    jobs_db[job_id] = job
    save_jobs_db()

    background_tasks.add_task(process_job, job_id, config)
    logger.info(f"Job {job_id} submitted for subject {config.subject_id}")

    return _job_to_frontend(job)


@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get status of a job (legacy endpoint)."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _job_to_frontend(jobs_db[job_id])


# ──────────────────────────────────────────────────────────────
# Job results  (metrics, connectome, file download)
# ──────────────────────────────────────────────────────────────

@app.get("/jobs/{job_id}/metrics")
async def get_job_metrics(job_id: str):
    """Return graph metrics for a completed job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    j = jobs_db[job_id]
    if j["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    results = j.get("results", {})
    metrics_file = results.get("metrics_file")
    if metrics_file and Path(metrics_file).exists():
        with open(metrics_file, "r") as f:
            return json.load(f)

    return results.get("metrics", {})


@app.get("/jobs/{job_id}/connectome")
async def get_job_connectome(job_id: str):
    """Return the connectome matrix for a completed job."""
    import numpy as np

    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    j = jobs_db[job_id]
    if j["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    results = j.get("results", {})
    connectome_file = results.get("connectome_file")
    if connectome_file and Path(connectome_file).exists():
        matrix = np.load(connectome_file)
        return matrix.tolist()

    raise HTTPException(status_code=404, detail="Connectome not found")


@app.get("/jobs/{job_id}/results/{result_type}")
async def get_job_result_file(job_id: str, result_type: str):
    """Download a result file (connectome, streamlines, metrics, fod)."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    j = jobs_db[job_id]
    if j["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    results = j.get("results", {})
    file_path = results.get(f"{result_type}_file")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail=f"Result '{result_type}' not found")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=Path(file_path).name,
    )


@app.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """Get all results of a completed job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    j = jobs_db[job_id]
    if j["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed (status: {j['status']})")

    return {"job_id": job_id, "results": j["results"]}


# ──────────────────────────────────────────────────────────────
# Pipeline endpoints  (tractography, graph-analysis)
# ──────────────────────────────────────────────────────────────

@app.post("/tractography")
async def run_tractography_api(params: TractographyParams, background_tasks: BackgroundTasks):
    """Submit a tractography job."""
    import uuid

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Tractography job queued",
        "config": {"mode": "tractography", **params.model_dump()},
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "results": None,
    }
    jobs_db[job_id] = job
    save_jobs_db()

    background_tasks.add_task(process_job, job_id, JobConfig(
        subject_id=params.dwi_file,
        mode="tractography",
        tractography={
            "step_size": params.step_size,
            "max_angle": params.max_angle,
            "seeds_per_voxel": params.seeds_per_voxel,
            "fa_threshold": params.fa_threshold,
        },
    ))

    return _job_to_frontend(job)


@app.post("/graph-analysis")
async def run_graph_analysis_api(params: GraphAnalysisParams, background_tasks: BackgroundTasks):
    """Submit a graph analysis job."""
    import uuid

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Graph analysis job queued",
        "config": {"mode": "graph-analysis", **params.model_dump()},
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "results": None,
    }
    jobs_db[job_id] = job
    save_jobs_db()

    background_tasks.add_task(process_job, job_id, JobConfig(
        subject_id=params.tractogram_file,
        mode="graph-analysis",
        connectome={"parcellation": params.atlas_file, "threshold": params.threshold},
    ))

    return _job_to_frontend(job)


# ──────────────────────────────────────────────────────────────
# Reference data endpoints  (atlases, algorithms, datasets)
# ──────────────────────────────────────────────────────────────

@app.get("/atlases")
async def list_atlases():
    """List available brain atlases/parcellations."""
    return [
        {"name": "aparc", "description": "Desikan-Killiany Atlas (FreeSurfer aparc)"},
        {"name": "aparc-reduced", "description": "Reduced Desikan-Killiany Atlas"},
        {"name": "schaefer_100", "description": "Schaefer 100 Parcels"},
        {"name": "schaefer_200", "description": "Schaefer 200 Parcels"},
        {"name": "schaefer_400", "description": "Schaefer 400 Parcels"},
        {"name": "aal", "description": "Automated Anatomical Labeling (AAL)"},
    ]


@app.get("/algorithms")
async def list_algorithms():
    """List available tractography algorithms."""
    return [
        {"name": "probabilistic", "description": "Probabilistic tractography with FOD sampling (default)"},
        {"name": "deterministic", "description": "Deterministic tractography following peak directions"},
    ]


@app.get("/datasets/list")
async def list_datasets():
    """List available datasets."""
    report_path = Path("analysis_and_decisions/dataset_analysis_report.json")
    if not report_path.exists():
        return {"datasets": [], "message": "No datasets found"}

    with open(report_path, 'r') as f:
        report = json.load(f)

    return {
        "total_subjects": report.get("total_subjects", 0),
        "total_size_gb": report.get("total_size_gb", 0),
        "datasets": report.get("datasets", []),
        "diffusion_subjects": report.get("diffusion_subjects", 0),
    }


# ──────────────────────────────────────────────────────────────
# Background job processing
# ──────────────────────────────────────────────────────────────

async def process_job(job_id: str, config: JobConfig):
    """
    Background task to process a job using the real NeuroTract pipeline.
    """
    from pathlib import Path
    import nibabel as nib
    import numpy as np
    from ..data.data_loader import DataLoader
    from ..preprocessing.pipeline import PreprocessingPipeline
    from ..microstructure.dti import DTIModel
    from ..microstructure.csd import CSDModel, ResponseFunction
    from ..tractography.probabilistic_tracker import ProbabilisticTracker
    from ..tractography.seeding import SeedGenerator
    from ..tractography.streamline_utils import StreamlineUtils
    from ..connectome.construct import ConnectomeBuilder
    from ..connectome.graph_metrics import ConnectomeMetrics

    logger.info(f"Processing job {job_id}...")
    output_dir = Path(f"output/{job_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    def update_progress(progress: float, message: str):
        jobs_db[job_id]["progress"] = progress
        jobs_db[job_id]["message"] = message
        jobs_db[job_id]["updated_at"] = datetime.now().isoformat()
        save_jobs_db()
        logger.info(f"Job {job_id}: {message} ({int(progress * 100)}%)")

    try:
        jobs_db[job_id]["status"] = "running"
        save_jobs_db()
        update_progress(0.0, "Loading data...")

        subject_path = Path(f"datasets/{config.subject_id}")

        data_loader = DataLoader()
        dwi_files = list(subject_path.glob("*dwi.nii.gz"))
        if not dwi_files:
            raise FileNotFoundError(f"No DWI data found for subject {config.subject_id}")

        dwi_file = dwi_files[0]
        bval_file = dwi_file.with_suffix('').with_suffix('.bval')
        bvec_file = dwi_file.with_suffix('').with_suffix('.bvec')

        data = data_loader.load_diffusion_data(str(dwi_file), str(bval_file), str(bvec_file))
        update_progress(0.1, "Data loaded")

        # Preprocessing
        update_progress(0.15, "Running preprocessing...")
        pipeline = PreprocessingPipeline(config=config.preprocessing or {})
        preprocessed = pipeline.run(data.volume, data.bvals, data.bvecs)

        dwi_corrected = preprocessed["dwi_corrected"]
        brain_mask = preprocessed["brain_mask"]
        bvals_corrected = preprocessed["bvals_corrected"]
        bvecs_corrected = preprocessed["bvecs_corrected"]
        update_progress(0.3, "Preprocessing completed")

        # DTI
        update_progress(0.35, "Computing DTI maps...")
        dti_model = DTIModel()
        tensors = dti_model.fit(dwi_corrected, bvals_corrected, bvecs_corrected, mask=brain_mask)
        fa_map = dti_model.compute_fa(tensors)
        update_progress(0.45, "DTI completed")

        # CSD/FOD
        update_progress(0.5, "Computing FOD...")
        csd_model = CSDModel(sh_order=8)
        response = ResponseFunction.estimate_from_data(dwi_corrected, bvals_corrected, bvecs_corrected, fa_map)
        fod = csd_model.fit(dwi_corrected, bvals_corrected, bvecs_corrected, response, mask=brain_mask)

        fod_path = output_dir / "fod.nii.gz"
        nib.save(nib.Nifti1Image(fod, data.affine), str(fod_path))
        update_progress(0.6, "FOD completed")

        # Tractography
        update_progress(0.65, "Running tractography...")
        tracker_config = config.tractography or {}
        tracker = ProbabilisticTracker(
            step_size=tracker_config.get('step_size', 0.5),
            max_angle=tracker_config.get('max_angle', 30.0),
            max_length=tracker_config.get('max_length', 200.0),
            min_length=tracker_config.get('min_length', 10.0)
        )

        seed_generator = SeedGenerator()
        seeds = seed_generator.generate_seeds(brain_mask, seeds_per_voxel=tracker_config.get('seeds_per_voxel', 2))

        streamlines = []
        total_seeds = len(seeds)
        for idx, seed in enumerate(seeds):
            streamline = tracker.track_from_seed(seed, fod, brain_mask, fa_map)
            if streamline is not None:
                streamlines.append(streamline)
            if idx % max(1, total_seeds // 10) == 0:
                progress = 0.65 + 0.2 * (idx / total_seeds)
                update_progress(progress, f"Tractography: {len(streamlines)} streamlines")

        streamlines_path = output_dir / "streamlines.trk"
        StreamlineUtils.save_trk(streamlines, str(streamlines_path), data.affine, brain_mask.shape)
        update_progress(0.85, f"Tractography completed: {len(streamlines)} streamlines")

        # Connectome
        update_progress(0.9, "Building connectome...")
        connectome_config = config.connectome or {}
        parcellation_file = connectome_config.get('parcellation')
        if parcellation_file and Path(parcellation_file).exists():
            parcellation_img = nib.load(parcellation_file)
            parcellation = parcellation_img.get_fdata()
        else:
            parcellation = np.zeros(brain_mask.shape, dtype=int)
            grid_size = 10
            for i, x in enumerate(range(0, brain_mask.shape[0], grid_size)):
                for j, y in enumerate(range(0, brain_mask.shape[1], grid_size)):
                    for k, z in enumerate(range(0, brain_mask.shape[2], grid_size)):
                        parcellation[x:x+grid_size, y:y+grid_size, z:z+grid_size] = i * 100 + j * 10 + k

        builder = ConnectomeBuilder()
        connectome = builder.build(
            streamlines, parcellation,
            weighting=connectome_config.get('weighting', 'count'),
            fa_map=fa_map
        )

        connectome_path = output_dir / "connectome.npy"
        np.save(str(connectome_path), connectome)
        update_progress(0.95, "Connectome built")

        # Metrics
        update_progress(0.97, "Computing graph metrics...")
        metrics_calculator = ConnectomeMetrics()
        metrics = metrics_calculator.compute_all(connectome)

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Mark completed
        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["progress"] = 1.0
        jobs_db[job_id]["message"] = "Processing completed successfully"
        jobs_db[job_id]["updated_at"] = datetime.now().isoformat()
        jobs_db[job_id]["results"] = {
            "connectome_file": str(connectome_path),
            "streamlines_file": str(streamlines_path),
            "fod_file": str(fod_path),
            "metrics_file": str(metrics_path),
            "num_streamlines": len(streamlines),
            "metrics": {
                "global_efficiency": float(metrics.get("global_efficiency", 0)),
                "modularity": float(metrics.get("modularity", 0)),
                "clustering_coefficient": float(metrics.get("clustering_coefficient", 0))
            }
        }
        save_jobs_db()
        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["message"] = f"Error: {str(e)}"
        jobs_db[job_id]["updated_at"] = datetime.now().isoformat()
        save_jobs_db()


# ──────────────────────────────────────────────────────────────
# Pre-computed results endpoints  (serve CLI pipeline output)
# ──────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("output")

@app.get("/results/available")
async def list_available_results():
    """
    Scan output/ directory for pre-computed pipeline results.
    Returns a list of subjects with their available files.
    """
    results = []
    if not OUTPUT_DIR.exists():
        return results

    for subject_dir in sorted(OUTPUT_DIR.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name

        entry = {
            "subject_id": subject_id,
            "has_streamlines": (subject_dir / "streamlines.trk").exists(),
            "has_metrics": (subject_dir / "metrics.json").exists(),
            "has_connectome": (subject_dir / "connectome.npy").exists(),
            "has_dti": (subject_dir / "dti").is_dir() if (subject_dir / "dti").exists() else False,
            "has_fod": (subject_dir / "fod.nii.gz").exists(),
            "files": [],
        }

        # List all files with sizes
        for f in sorted(subject_dir.rglob("*")):
            if f.is_file():
                rel = f.relative_to(subject_dir)
                entry["files"].append({
                    "name": str(rel),
                    "size_bytes": f.stat().st_size,
                    "type": _classify_file(f.name),
                })

        # Load statistics if available
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
    """Classify a file by its extension."""
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
    elif fl.endswith(".txt"):
        return "text"
    elif fl.endswith(".bval") or fl.endswith(".bvals"):
        return "bval"
    elif fl.endswith(".bvec") or fl.endswith(".bvecs"):
        return "bvec"
    else:
        return "other"


@app.get("/results/{subject_id}/streamlines")
async def get_result_streamlines(subject_id: str, max_streamlines: int = 3000):
    """
    Read TRK file from output/{subject_id}/streamlines.trk and return
    streamlines as JSON for the 3D viewer. Subsamples to max_streamlines
    to keep the response size manageable for the browser.
    """
    import nibabel as nib
    import numpy as np

    trk_path = OUTPUT_DIR / subject_id / "streamlines.trk"
    if not trk_path.exists():
        raise HTTPException(status_code=404, detail=f"No streamlines found for {subject_id}")

    logger.info(f"Loading streamlines from {trk_path}...")
    tractogram = nib.streamlines.load(str(trk_path))
    all_streamlines = tractogram.streamlines

    n_total = len(all_streamlines)
    logger.info(f"Loaded {n_total} streamlines, subsampling to {max_streamlines}")

    # Subsample if needed
    if n_total > max_streamlines:
        indices = np.linspace(0, n_total - 1, max_streamlines, dtype=int)
        selected = [all_streamlines[i] for i in indices]
    else:
        selected = list(all_streamlines)

    # Convert to JSON-serializable format
    streamlines_data = []
    all_points = []
    total_points = 0
    lengths = []

    for sl in selected:
        points = sl.astype(float)  # Nx3 array
        n_pts = len(points)
        total_points += n_pts

        # Calculate length
        diffs = np.diff(points, axis=0)
        length = float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))
        lengths.append(length)

        # Calculate mean orientation
        if n_pts > 1:
            tangent = points[-1] - points[0]
            mag = np.linalg.norm(tangent)
            if mag > 0:
                tangent = tangent / mag
            orientation = [float(tangent[0]), float(tangent[1]), float(tangent[2])]
        else:
            orientation = [0.0, 0.0, 1.0]

        # Flatten points to [x,y,z,x,y,z,...]
        flat_points = points.flatten().tolist()
        all_points.extend([points.min(axis=0), points.max(axis=0)])

        streamlines_data.append({
            "points": flat_points,
            "numPoints": n_pts,
            "length": length,
            "orientation": orientation,
        })

    # Calculate bounds
    if all_points:
        all_mins = np.array([p for i, p in enumerate(all_points) if i % 2 == 0])
        all_maxs = np.array([p for i, p in enumerate(all_points) if i % 2 == 1])
        bounds_min = all_mins.min(axis=0).tolist()
        bounds_max = all_maxs.max(axis=0).tolist()
    else:
        bounds_min = [0, 0, 0]
        bounds_max = [0, 0, 0]

    lengths_arr = np.array(lengths) if lengths else np.array([0.0])

    result = {
        "streamlines": streamlines_data,
        "bounds": {
            "min": bounds_min,
            "max": bounds_max,
        },
        "metadata": {
            "count": len(streamlines_data),
            "totalPoints": total_points,
            "meanLength": float(lengths_arr.mean()),
            "maxLength": float(lengths_arr.max()),
            "minLength": float(lengths_arr.min()),
            "totalInFile": n_total,
        },
    }

    logger.info(f"Returning {len(streamlines_data)} streamlines ({total_points} points)")
    return result


@app.get("/results/{subject_id}/metrics")
async def get_result_metrics(subject_id: str):
    """
    Return metrics.json for a subject, mapped to the frontend GraphMetrics format.
    """
    metrics_path = OUTPUT_DIR / subject_id / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail=f"No metrics found for {subject_id}")

    with open(metrics_path) as f:
        raw = json.load(f)

    # Map to frontend GraphMetrics format
    global_data = raw.get("global", {})
    result = {
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

    return result


@app.get("/results/{subject_id}/connectome")
async def get_result_connectome(subject_id: str):
    """Return the connectome matrix as a 2D array for a subject."""
    import numpy as np

    npy_path = OUTPUT_DIR / subject_id / "connectome.npy"
    if not npy_path.exists():
        raise HTTPException(status_code=404, detail=f"No connectome found for {subject_id}")

    matrix = np.load(str(npy_path))
    return matrix.tolist()


@app.get("/results/{subject_id}/info")
async def get_result_info(subject_id: str):
    """Return all available info/statistics files for a subject."""
    subject_dir = OUTPUT_DIR / subject_id
    if not subject_dir.exists():
        raise HTTPException(status_code=404, detail=f"Subject {subject_id} not found")

    info = {"subject_id": subject_id}

    for json_file in subject_dir.glob("*.json"):
        key = json_file.stem
        with open(json_file) as f:
            info[key] = json.load(f)

    for txt_file in subject_dir.glob("*.txt"):
        key = txt_file.stem
        info[key] = txt_file.read_text()

    return info


@app.get("/results/{subject_id}/dti-maps")
async def get_dti_maps_info(subject_id: str):
    """Return list of available DTI maps with their info."""
    dti_dir = OUTPUT_DIR / subject_id / "dti"
    if not dti_dir.exists():
        raise HTTPException(status_code=404, detail=f"No DTI maps for {subject_id}")

    maps = []
    for f in sorted(dti_dir.iterdir()):
        if f.is_file() and f.suffix in ('.gz', '.nii'):
            maps.append({
                "name": f.stem.replace('.nii', ''),
                "filename": f.name,
                "size_bytes": f.stat().st_size,
            })

    return maps


@app.get("/results/{subject_id}/brain-mesh")
async def get_brain_mesh(subject_id: str, step_size: int = 1):
    """
    Generate and return a triangulated brain surface mesh from the brain mask.
    Uses marching cubes algorithm. Result is cached on disk.
    """
    from ..surfaces.mesh_generator import generate_brain_mesh

    subject_dir = OUTPUT_DIR / subject_id
    mask_path = subject_dir / "preprocessed" / "preprocessed_brain_mask.nii.gz"

    if not mask_path.exists():
        # Try alternate locations
        dataset_dir = Path("datasets")
        for d in dataset_dir.iterdir():
            if d.is_dir():
                alt = d / f"{subject_id}_brain_mask.nii.gz"
                if alt.exists():
                    mask_path = alt
                    break

    if not mask_path.exists():
        raise HTTPException(status_code=404, detail=f"No brain mask found for {subject_id}")

    try:
        result = generate_brain_mesh(
            mask_path=str(mask_path),
            step_size=step_size,
            smooth_sigma=1.0,
            cache_dir=str(subject_dir),
        )
        logger.info(f"Brain mesh: {result['metadata']['n_vertices']} vertices, {result['metadata']['n_faces']} faces")
        return result
    except Exception as e:
        logger.error(f"Failed to generate brain mesh: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Mesh generation failed: {str(e)}")


@app.get("/results/{subject_id}/parcellation-labels")
async def get_parcellation_labels(subject_id: str):
    """
    Return anatomical parcellation labels mapped from the Desikan-Killiany atlas.
    """
    from ..surfaces.parcellation_mapping import get_parcellation_labels, LOBE_CENTROIDS

    subject_dir = OUTPUT_DIR / subject_id
    labels_file = subject_dir / "connectome_labels.txt"

    # Determine number of parcels from connectome if available
    n_parcels = 89
    info_file = subject_dir / "connectome_info.json"
    if info_file.exists():
        with open(info_file) as f:
            info = json.load(f)
            n_parcels = info.get("n_parcels", 89)

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


@app.get("/results/{subject_id}/download/{filename:path}")
async def download_result_file(subject_id: str, filename: str):
    """Download any result file from a subject's output directory."""
    file_path = OUTPUT_DIR / subject_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    # Security: ensure path doesn't escape the output dir
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
    """Start the FastAPI server."""
    logger.info(f"Starting NeuroTract API server on {host}:{port}")
    uvicorn.run(
        "src.backend.api.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()
