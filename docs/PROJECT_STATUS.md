# NeuroTract 2.0 — Project Status

**Date**: September 6, 2026  
**Auditor**: Principal Scientific Software Engineer, Antigravity AI  
**Repository State**: In Transformation to NeuroTract 2.0

---

## Executive Summary

NeuroTract is an open-source diffusion MRI tractography and structural connectome analysis laboratory. A thorough code and scientific audit conducted in Phase 0 revealed significant scientific assets—notably real Stanford HARDI diffusion data (`SUB1`, `SUB2`), a working DIPY-based processing pipeline in the CLI, and a Three.js 3D viewer—alongside critical bugs, scientific errors, and architectural discrepancies that must be eliminated to achieve production quality.

---

## Findings Matrix

| Component | Status Before Audit | Reality & Bugs Uncovered | NeuroTract 2.0 Remediation |
|---|---|---|---|
| **DTI Model** (`dti.py`) | ⚠️ Broken Tests (2/11 failed) | Fractional Anisotropy formula used `0.5 * sum((evals - md)**2)` instead of `1.5 * sum(...)`, reducing FA by $\sqrt{3} \approx 1.732$. | Fixed standard Basser/Pierpaoli formula; all DTI tests pass with strict tolerances. |
| **API Server** (`server.py`) | ❌ Runtime Failure on Submit | `process_job` calling conventions mismatched: passed volume instead of path to `PreprocessingPipeline`, omitted `bvals`/`bvecs` in `DTIModel()`, omitted `voxel_size` in `ProbabilisticTracker()`, omitted `parcellation` in `ConnectomeBuilder()`. | Standardized service layer connecting real pipeline with typed parameters and graceful error handling. |
| **Execution Engine** | ⚠️ Polling Only | Web UI polled `/jobs/{id}` every 2 seconds. No event-driven updates, no live telemetry, no stream of stage milestones. | Built real-time Server-Sent Events (SSE) streaming engine with genuine telemetry (processed seeds, kept streamlines, stage durations). |
| **Scientific Truth** | ❌ Unscientific UI Elements | `BrainHealthSummary.tsx` calculated a fake "Health Score %", diagnosed subjects, and flagged normal Stanford scans as "concerning" due to arbitrary hard-coded thresholds. | Removed all diagnostic claims and arbitrary normative ranges. Streamlines labeled strictly as algorithmic reconstructions. |
| **Anatomical Visualization** | ⚠️ Hybrid Real / Stock | 3D viewer supported marching cubes surface mesh from real brain mask, but defaulted to stock GLB model (`brain_hologram.glb`). 2D slice controls existed in settings but were never wired to real slice volume data. | Removed stock GLBs; render real subject marching cubes mesh and real tractogram. Integrated synchronized 2D orthographic slices (axial, coronal, sagittal). |
| **Metric Provenance** | ❌ Absent | Metrics displayed as naked numbers with zero traceability to software versions, execution IDs, preprocessing transforms, or seeds. | Built "Where did this number come from?" Provenance Inspector tracing every metric to its exact mathematical formula, input graph, and parameters. |
| **Dataset Ingestion** | ⚠️ Minimal Boundary Checks | API accepted uploads without validating gradient table dimensions, unit norm of vectors, or non-zero file sizes (dummy 0-byte files existed in repo). | Created `DatasetValidator` generating structured `DatasetValidationReport` with SHA-256 checksums and gradient checks. |
| **Validation & Benchmarks** | ❌ No Reference Testing | No automated comparison against reference toolkits or known mathematical tensors. | Built Validation Center comparing NeuroTract DTI and graph metrics against reference baselines with documented tolerances. |

---

## Software & Dependency Baseline

- **Python**: 3.11.9
- **DIPY**: 1.11.0 (Diffusion Imaging in Python)
- **Nibabel**: 5.3.3 (Neuroimaging file formats)
- **NetworkX**: 3.6.1 (Graph theory metrics)
- **SciPy**: 1.17.0
- **Scikit-learn**: 1.8.0
- **FastAPI**: 0.128.7
- **Next.js**: 14.2.35
- **React**: 18.2.0
- **Three.js**: 0.160.0 (@react-three/fiber 8.15.0, @react-three/drei 9.92.0)

---

## Active Roadmap

1. [x] Phase 0: Repository Audit & Knowledge Base
2. [x] Phase 1: Scientific Truth Audit (FA formula correction)
3. [x] Phase 2: Data Standardization & Validation Layer
4. [x] Phase 3: Authoritative Demo Dataset Management (Stanford HARDI SUB1 & SUB2)
5. [x] Phase 4: Scientific Pipeline Hardening & Provenance Tracking
6. [x] Phase 5: Real-Time SSE Execution Engine (Starlette threadpool background worker)
7. [x] Phase 6: Live Analysis Observatory with 1-click execution trigger
8. [x] Phase 7: Synchronized 2D/3D Anatomical & Connectome Viewer
9. [x] Phase 8-10: Advanced Scientific Visualizations & Provenance Inspector
10. [x] Phase 11-12: Parameter Sensitivity Lab & Validation Center (DIPY / NetworkX numerical concordance)
11. [x] Phase 13-14: Performance Optimization & Reproducible Reporting (HTML export)
12. [x] Phase 15-23: Frontend/Backend Hardening, Run Comparison Suite (A vs B), E2E Tests, and Final Verification

