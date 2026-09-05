# NeuroTract 2.0 — Workstation UX Specification

## Design Philosophy

NeuroTract 2.0 is an advanced scientific visualization workstation. It rejects superficial SaaS styling, cartoon gradients, and misleading glowing badges in favor of:
- **High Information Density**: Precise numerical displays, inspectable tables, and contextual tooltips.
- **Synchronized Multi-View Exploration**: Bidirectional cross-filtering between 3D brain geometry, 2D orthogonal MRI slices, connectome graphs, and connectivity matrices.
- **Meaningful State Communication**: Purposeful semantic color coding rather than decorative animations:
  - `Gray`: Pending / Queued
  - `Blue`: Active / Running computation
  - `Green`: Completed / Validated
  - `Amber`: Scientific Warning / High uncertainty
  - `Red`: Failed / Rejected boundary check
  - `Purple / Cyan`: Reconstructed streamline / Inferred tractography trajectory

---

## Core Layout & Workstation Views

### 1. Ingestion & Live Observatory (Dashboard)
- **Dataset Inspector**: Live drag-and-drop or subject selector with immediate boundary validation (dimensions, voxel size, gradient count, SNR).
- **Pipeline Runner**: Parameter control panel (algorithm, step size, seed density, curvature threshold, random seed).
- **Real-Time Execution Engine**:
  - Live milestone timeline: Ingestion $\to$ Preprocessing $\to$ DTI $\to$ CSD $\to$ Tractography $\to$ Parcellation $\to$ Connectome $\to$ Graph Metrics.
  - Active telemetry card: Real-time progress %, elapsed duration, seeds evaluated, streamlines kept/rejected with live breakdown, memory usage.
  - Live execution log console.

### 2. 3D/2D Anatomical & Connectome Viewer
- **Synchronized Views**:
  1. 3D Viewport: Real subject marching cubes cortical surface + streamlines color-coded by local orientation (Red = L/R, Green = A/P, Blue = I/S) or FA.
  2. 2D Orthogonal Slices: Axial, Coronal, and Sagittal cross-sections computed from real FA/MD volumes, with interactive position crosshairs.
  3. Interactive Region Selection: Clicking an anatomical region highlights all connected streamlines, selects the parcel in the 3D surface, centers the orthogonal slices, and filters the connectome matrix.

### 3. Analytics & Provenance Laboratory
- **Graph Metrics Panel**: Global and nodal network metrics (clustering, path length, efficiency, modularity, rich club, degree distribution).
- **Connectome Matrix Heatmap**: Interactive $89 \times 89$ matrix with zoom, parcel reordering by lobe/system, and edge weight inspection.
- **"Where Did This Number Come From?" Inspector**:
  - Reusable slide-out provenance drawer: Click any metric to inspect exact formula, input graph, node count, edge count, weighting, software versions, execution ID, and timestamp.
- **Parameter Sensitivity Lab**:
  - Side-by-side comparison of tracking runs with differing parameters (e.g. angle $30^\circ$ vs $45^\circ$, seed density 2 vs 5).
  - Highlights stable edges, lost connections, and metric divergence.
- **Validation Center**:
  - Direct benchmark comparisons against reference implementations (DIPY).
  - Scatter plots, Bland-Altman agreement plots, correlation coefficients, and numerical error metrics.
- **Reproducible Report Exporter**:
  - One-click export of complete analysis bundle: HTML report with embedded figures, JSON provenance bundle, and CSV connectome matrices.
