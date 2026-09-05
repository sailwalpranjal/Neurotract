# NeuroTract 2.0 — Datasets & Ingestion Specifications

## 1. Supported Input Formats

NeuroTract 2.0 strictly defines valid input combinations. Arbitrary unsupported files are rejected at the boundary with actionable error reports:

1. **Diffusion NIfTI + Gradient Tables**:
   - 4D Volume: `.nii` or `.nii.gz`
   - b-values: `.bval` or `.bvals` (space or tab-delimited ASCII)
   - b-vectors: `.bvec` or `.bvecs` ($3 \times N$ or $N \times 3$ ASCII)
2. **Anatomical Reference**:
   - T1-weighted or T2-weighted structural volume: `.nii` or `.nii.gz`
3. **Parcellation Atlas**:
   - Voxel-based integer parcellation: `.nii` or `.nii.gz` matching or coregistered with anatomical volume (e.g. FreeSurfer Desikan-Killiany aparc, Schaefer 100/200/400).
4. **BIDS (Brain Imaging Data Structure)**:
   - Valid subject directory hierarchy: `sub-<id>/ses-<ses>/dwi/sub-<id>_dwi.nii.gz` with sidecar `.json`, `.bval`, and `.bvec`.

---

## 2. Benchmark Datasets in Repository

### A. Stanford HARDI Dataset (Primary Verified Demo Data)
- **Source**: Stanford Center for Cognitive and Neurobiological Imaging (CNI)
- **URL**: http://purl.stanford.edu/ng782rw8378
- **License**: Creative Commons Attribution 3.0 (CC BY 3.0)
- **Acquisition Parameters**:
  - Scanner: GE Discovery MR750 3.0 Tesla
  - Sequence: 2D Spin Echo EPI, TR = 8200 ms, TE = 96.8 ms
  - Voxel size: $2.0 \times 2.0 \times 2.0\text{ mm}^3$ (matrix $128 \times 128 \times 60$)
  - Shells: Multi-shell HARDI:
    - $b = 0\text{ s/mm}^2$ (10 baseline volumes)
    - $b = 1000\text{ s/mm}^2$ (150 non-collinear directions)
    - $b = 2000\text{ s/mm}^2$ (150 non-collinear directions)
    - $b = 4000\text{ s/mm}^2$ (150 non-collinear directions)
  - Subjects available locally in `datasets/Stanford dataset/`:
    - `SUB1`: Full multi-shell scans, T1 anatomical, aparc-reduced parcellation (89 regions), corpus callosum mask.
    - `SUB2`: Full multi-shell scans, T1 anatomical, classification maps.

### B. Precomputed Verified Demo Artifacts
Located in `output/SUB1/`:
- Streamlines: `streamlines.trk` (22,392 streamlines, 8.5 MB)
- Brain surface: `brain_mesh_step1.json` (marching cubes from mask, 3.3 MB)
- Connectome: `connectome.npy` (89x89 matrix, 788 edges)
- Metrics: `metrics.json` (complete global & nodal graph theory measures)
- DTI scalar maps: `dti_fa.nii.gz`, `dti_md.nii.gz`, `dti_rd.nii.gz`, `dti_ad.nii.gz`
- CSD: `fod.nii.gz` (spherical harmonic FOD volume, 45 coefficients)

---

## 3. Dataset Validation Checks

The `DatasetValidator` enforces the following rules prior to execution:

| Check | Requirement | Action on Failure |
|---|---|---|
| **Dimension Match** | 4th dimension of 4D DWI == number of entries in `.bval` == number of columns/rows in `.bvec` | Halt with `MismatchedDimensionsError` |
| **Gradient Table** | Gradient vectors must be approximately unit norm ($\|g\| \approx 1.0 \pm 0.05$) for $b > 50$ | Re-normalize if minor discrepancy; halt if corrupt |
| **b0 Count** | Dataset must contain at least 1 volume with $b \le 50\text{ s/mm}^2$ | Halt with `MissingB0Error` |
| **File Integrity** | File size must be $> 0$; gzip magic number valid | Halt with `CorruptFileError` |
| **Coordinate Affine** | Affine matrix determinant must be non-zero and invertible | Halt with `InvalidAffineError` |
