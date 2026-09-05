# NeuroTract 2.0 — Architecture & Data Flow

## System Architecture

NeuroTract 2.0 is structured as a decoupled scientific visualization laboratory comprising three primary layers:
1. **Scientific Computing Core** (Python / DIPY / Nibabel / NetworkX / SciPy)
2. **API & Real-Time Orchestration Layer** (FastAPI / SSE Event Bus / Provenance Store)
3. **Interactive Analysis Workstation** (Next.js 14 / React 18 / Three.js / Plotly.js / Zustand)

```
┌────────────────────────────────────────────────────────────────────────┐
│               NeuroTract 2.0 Laboratory (Frontend)                     │
│  ┌──────────────────────┬──────────────────────┬────────────────────┐  │
│  │  Live Observatory    │  3D Brain & Slices   │ Connectome Matrix  │  │
│  │  (SSE Event Stream)  │  (Three.js Viewport) │ & Graph Analytics  │  │
│  └──────────┬───────────┴──────────┬───────────┴────────────┬───────┘  │
│             │                      │                        │          │
│  ┌──────────┴──────────────────────┴────────────────────────┴───────┐  │
│  │         State Management (Zustand Store) & API Client            │  │
│  │  - Synchronized parcel/streamline selection                      │  │
│  │  - Provenance Inspector Drawer ("Where did this number come from?")│
│  └───────────────────────────────┬──────────────────────────────────┘  │
└──────────────────────────────────┼─────────────────────────────────────┘
                                   │ HTTP / SSE
┌──────────────────────────────────┼─────────────────────────────────────┐
│  FastAPI Backend Layer           │                                     │
│  ┌───────────────────────────────┴──────────────────────────────────┐  │
│  │  API Endpoints: Ingestion, Jobs, Results, Provenance, Validation │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │  Real-Time Execution Engine (SSE Event Broadcaster)              │  │
│  │  - Background worker thread with queue & cancellation            │  │
│  │  - Genuine telemetry: seeds processed, rejection counts, timings │  │
│  └───────────────────────────────┬──────────────────────────────────┘  │
│                                  │                                     │
│  Scientific Pipeline Engine      │                                     │
│  ┌───────────────────────────────┴──────────────────────────────────┐  │
│  │ 1. Data Ingestion & Boundary Validation                          │  │
│  │ 2. Preprocessing (Gradient QC, Brain Masking, Bias Correction)    │  │
│  │ 3. Microstructure Modeling (DTI: FA/MD/RD/AD & CSD: FOD SH)      │  │
│  │ 4. Probabilistic Tractography (RK4 Integration, FOD Sampling)    │  │
│  │ 5. Parcellation Mapping & Connectome Construction                │  │
│  │ 6. Graph Theory Network Metrics & Spectral Analysis              │  │
│  │ 7. Provenance Tracking & Reproducible Report Bundler             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

## Data Flow & Processing Stages

### 1. Ingestion Boundary
- Input: DWI NIfTI (`.nii.gz`), b-values (`.bval`), b-vectors (`.bvec`), parcellation (`.nii.gz`).
- `DatasetValidator`: Checks 4D shape compatibility, gradient table dimension matching, unit vector normalization, presence of $b=0$ volumes, and generates a canonical `DatasetValidationReport`.

### 2. Preprocessing
- Gradient correction: Unit normalization and orientation check.
- Brain extraction: DIPY `median_otsu` to compute binary mask.
- Bias field correction: Intensity normalization across coil sensitivities.
- Saves intermediate NIfTI masks and QC reports with SHA-256 checksums.

### 3. Microstructure Modeling
- **DTI (Diffusion Tensor Imaging)**: Weighted Least Squares fit yielding diffusion tensor $D$, eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3$, eigenvectors, Fractional Anisotropy (FA), Mean Diffusivity (MD), Radial Diffusivity (RD), and Axial Diffusivity (AD).
- **CSD (Constrained Spherical Deconvolution)**: Real symmetric spherical harmonics (order $L_{max}=8$, 45 coefficients) for fiber orientation distribution (FOD) estimation.

### 4. Tractography & Surface Extraction
- Probabilistic Monte Carlo tracking using 4th-order Runge-Kutta (RK4) integration.
- Stopping criteria: FA $< 0.1$, FOD amplitude $< 0.1$, turning angle $> 30^\circ$, or exiting brain mask.
- Marching Cubes algorithm generates high-resolution triangulated surface mesh from subject brain mask.

### 5. Structural Connectome & Graph Theory
- Endpoints of reconstructed streamlines mapped to anatomical parcels (Desikan-Killiany aparc atlas, 89 parcels).
- Edge weights: Streamline count, length-normalized count, or microstructural (mean FA).
- NetworkX graph metrics: Density, clustering coefficient, characteristic path length, global efficiency, modularity (Louvain), degree centrality, betweenness centrality.

### 6. Real-Time Event Stream
The backend emits Server-Sent Events (SSE) via `GET /api/jobs/{id}/events`:
- Event Types: `stage_started`, `stage_progress`, `stage_completed`, `telemetry_update`, `pipeline_completed`, `pipeline_failed`.
- Telemetry includes: `stage`, `elapsed_seconds`, `progress`, `seeds_evaluated`, `streamlines_accepted`, `streamlines_rejected_by_reason`, `memory_mb`.

### 7. Dual Mode Execution
- **Demo Mode**: Loads pre-computed, verified subject artifacts (`SUB1`, `SUB2`) instantly from disk for responsive exploration.
- **Compute Mode**: Runs the genuine scientific pipeline on uploaded or selected datasets, streaming live events and saving verifiable output artifacts.
