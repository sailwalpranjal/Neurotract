# NeuroTract Agent Map

Welcome to **NeuroTract 2.0** — an interactive, transparent diffusion-MRI analysis laboratory.

## Repository Navigation Map

| Domain | Key Paths & Files | Primary Purpose |
|---|---|---|
| **Documentation & Standards** | [`docs/`](file:///f:/NeuroTract/docs) | Authoritative knowledge base, scientific contracts, specs, and status |
| **Scientific Backend** | [`src/backend/`](file:///f:/NeuroTract/src/backend) | Python scientific computing, DTI, CSD, tractography, connectome, metrics |
| **API & Real-Time Engine** | [`src/backend/api/`](file:///f:/NeuroTract/src/backend/api) | FastAPI server, SSE event bus, dataset validator, provenance endpoints |
| **Web Laboratory (Frontend)** | [`src/frontend/`](file:///f:/NeuroTract/src/frontend) | Next.js 14, Three.js 3D viewer, synchronized 2D slices, analysis dashboard |
| **Datasets & Bootstrapping** | [`datasets/`](file:///f:/NeuroTract/datasets), [`scripts/`](file:///f:/NeuroTract/scripts) | Stanford HARDI dataset, validation tools, data acquisition scripts |
| **Precomputed Artifacts** | [`output/`](file:///f:/NeuroTract/output) | Verified output bundles (SUB1, SUB2) for immediate exploration |
| **Test Suites** | [`tests/`](file:///f:/NeuroTract/tests) | Pytest suite: unit, integration, validation, mathematical contracts |

## Essential Documentation Links

- **Current Status & Audit**: [`docs/PROJECT_STATUS.md`](file:///f:/NeuroTract/docs/PROJECT_STATUS.md)
- **Architecture & Data Flow**: [`docs/ARCHITECTURE.md`](file:///f:/NeuroTract/docs/ARCHITECTURE.md)
- **Scientific Contract & Math**: [`docs/SCIENTIFIC_CONTRACT.md`](file:///f:/NeuroTract/docs/SCIENTIFIC_CONTRACT.md)
- **Datasets & Validation Specs**: [`docs/DATASETS.md`](file:///f:/NeuroTract/docs/DATASETS.md)
- **Validation & Benchmarks**: [`docs/VALIDATION.md`](file:///f:/NeuroTract/docs/VALIDATION.md)
- **Workstation UX Specification**: [`docs/UX_SPEC.md`](file:///f:/NeuroTract/docs/UX_SPEC.md)
- **Execution Plan & Progress**: [`docs/EXECUTION_PLAN.md`](file:///f:/NeuroTract/docs/EXECUTION_PLAN.md)

## Development Workflows

```bash
# Run all Python tests
.\.venv\Scripts\pytest.exe -v

# Run FastAPI backend with live SSE streaming
.\.venv\Scripts\python.exe -m uvicorn src.backend.api.server:app --reload --port 8000

# Frontend development
cd src/frontend
npm run dev

# Frontend type check & build
npm run type-check
npm run build
```

## Non-Negotiable Engineering Rules

1. **No Fake Data or Simulated Computations**: Every chart, streamline, and metric must map to real backend data or live worker execution.
2. **Scientific Integrity**: Reconstructed streamlines are algorithmic trajectories, never literal physical axons. No medical diagnosis or fake health scores.
3. **Traceable Provenance**: All numbers must be traceable to input files, preprocessing, algorithms, parameters, RNG seeds, and software versions.
