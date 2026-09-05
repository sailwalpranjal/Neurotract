# NeuroTract 2.0 — Execution Plan & Progress Tracker

**Current Phase**: Phase 1 Complete $\to$ Proceeding to Implementation & Integration  
**Last Updated**: September 6, 2026

---

## Phase Status Summary

| Phase | Description | Status | Key Deliverables & Decisions |
|---|---|---|---|
| **Phase 0** | Repository & Architecture Audit | ✅ Completed | Comprehensive audit completed. `AGENTS.md` and `docs/` created. |
| **Phase 1** | Scientific Truth Audit & Math Fixes | ✅ Completed | Identified and corrected FA math error; removed unscientific health scores. |
| **Phase 2** | Data Standardization & Ingestion | 🔄 In Progress | `DatasetValidator` with shape, gradient, and checksum verification. |
| **Phase 3** | Authoritative Demo Datasets | 🔄 In Progress | Reproducible download script & Stanford HARDI bootstrap. |
| **Phase 4** | Scientific Pipeline & Provenance | 🔄 In Progress | Fix `server.py` signatures; integrate `ProvenanceTracker`. |
| **Phase 5** | Real-Time Execution Engine (SSE) | ⏳ Next | Event-driven SSE streaming endpoint & worker task management. |
| **Phase 6** | Live Analysis Observatory | ⏳ Queued | Milestone execution timeline, telemetry cards, and log viewer. |
| **Phase 7** | Synchronized 2D/3D Brain Experience | ⏳ Queued | Orthogonal slice viewer synchronized with 3D tractogram and surface. |
| **Phase 8** | Advanced Scientific Visualization | ⏳ Queued | Distribution charts, connectome heatmap, degree distributions. |
| **Phase 9** | Microinteractions & Workstation UX | ⏳ Queued | Polished cross-filtering, keyboard shortcuts, accessible tooltips. |
| **Phase 10** | "Where Did This Number Come From?" | ⏳ Queued | Interactive Provenance Inspector drawer. |
| **Phase 11** | Parameter Sensitivity Lab | ⏳ Queued | Multi-run parameter variation and edge stability comparisons. |
| **Phase 12** | Validation Center | ⏳ Queued | DIPY reference baseline agreement and error metrics. |
| **Phase 13** | Performance Optimization | ⏳ Queued | Subsampling, level-of-detail, binary transfers. |
| **Phase 14** | Reproducible Report Generation | ⏳ Queued | Exportable HTML/JSON analysis bundle with provenance. |
| **Phase 15** | Frontend Redesign & Polishing | ⏳ Queued | Workstation layout, clean dark theme, semantic indicators. |
| **Phase 16** | Backend Robustness & Schemas | ⏳ Queued | Pydantic v2 schemas, structured error handling, clean shutdowns. |
| **Phase 17** | Testing Battery | ⏳ Queued | Unit, integration, scientific, and E2E pipeline tests. |
| **Phase 18** | Failure Handling | ⏳ Queued | Deliberate bad-input handling and graceful error surfaces. |
| **Phase 19** | Security & Data Privacy | ⏳ Queued | Safe file handling, path traversal prevention, research disclaimer. |
| **Phase 20** | Authoritative Documentation | ⏳ Queued | README.md update and full API reference. |
| **Phase 21** | Code Quality & Dead Code Removal | ⏳ Queued | Code cleanup, typing annotations, dead code pruning. |
| **Phase 22** | Coherent Incremental Commits | ⏳ Queued | Discrete, validated git commits (no push). |
| **Phase 23** | Verification Loop & Final Report | ⏳ Queued | Full test run, dev server verification, console checks. |

---

## Technical Decisions Log

1. **DTI Fractional Anisotropy Formula**:
   - *Problem*: Formula was calculating $\sqrt{0.5 \sum (\lambda_i - \bar{\lambda})^2} / \sqrt{\sum \lambda_i^2}$, under-estimating FA by factor of $\sqrt{3}$.
   - *Resolution*: Replaced with standard Basser & Pierpaoli formulation using $1.5$ factor for variance from mean eigenvalue.
2. **Real-Time Streaming**:
   - *Decision*: Adopt Server-Sent Events (SSE) via Starlette/FastAPI `StreamingResponse` for unidirectional real-time telemetry, avoiding WebSocket connection setup overhead while offering native browser reconnection.
3. **Synchronized 2D Orthogonal Slices**:
   - *Decision*: Provide slice generation from DTI FA/MD NIfTI volumes via backend API endpoint serving fast base64/PNG or raw arrays, synchronized with 3D crosshairs in Three.js viewport.
4. **Removal of Fake Health Scores**:
   - *Decision*: Remove `BrainHealthSummary.tsx` completely; replace with `ProvenanceInspector` and `ValidationCenter`.
