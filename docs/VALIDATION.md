# NeuroTract 2.0 — Validation & Benchmark Framework

## 1. Truth & Verification Principles

1. **No Fabricated Benchmarks**: Never claim "100% accuracy" where physical or mathematical ground truth does not support it.
2. **Reference Comparison**: Compare NeuroTract outputs against established open-source tools (DIPY reference implementations, NetworkX graph baselines) with explicit numerical tolerances.
3. **Statistical Reporting**: Report Pearson correlation ($r$), Mean Absolute Error (MAE), Relative Error (RE), and Jaccard edge overlap rather than hand-waving claims.

---

## 2. Validation Test Battery

### Test Suite 1: Synthetic Tensor Analytical Verification
Tests DTI tensor fitting against synthetically generated signals from known diffusion tensors:
- **Cylindrical Tensor**: $\lambda_1 = 1.5 \times 10^{-3}, \lambda_2 = \lambda_3 = 0.4 \times 10^{-3}\text{ mm}^2/\text{s}$.
  - Expected FA: $\approx 0.6862$
  - Computed FA tolerance: $|FA_{\text{computed}} - 0.6862| < 0.005$.
- **Planar Tensor**: $\lambda_1 = \lambda_2 = 1.0 \times 10^{-3}, \lambda_3 = 0.2 \times 10^{-3}\text{ mm}^2/\text{s}$.
  - Expected FA: $\approx 0.6547$.
- **Isotropic Tensor**: $\lambda_1 = \lambda_2 = \lambda_3 = 0.7 \times 10^{-3}\text{ mm}^2/\text{s}$.
  - Expected FA: $0.0000$ (tolerance $< 0.01$).

### Test Suite 2: Reference Implementation Agreement (DIPY Baseline)
Using Stanford HARDI SUB1:
- Fit standard DIPY `TensorModel` vs NeuroTract `DTIModel` on identical voxels.
- Compute FA Pearson correlation: expected $r > 0.999$, MAE $< 10^{-4}$.
- Compute MD Pearson correlation: expected $r > 0.999$, MAE $< 10^{-6}\text{ mm}^2/\text{s}$.

### Test Suite 3: Graph Theory Metric Consistency
Using the 89-node Stanford connectome:
- Verify that global efficiency calculated via inverse shortest paths strictly satisfies $0 \le E_{\text{glob}} \le 1$.
- Verify that node degrees sum to $2 \times |E|$.
- Verify that modularity $Q \in [-0.5, 1.0]$.
- Compare NetworkX reference metrics with internal calculations (MAE $< 10^{-7}$).

### Test Suite 4: Boundary & Degradation Testing
- Verify rejection of 0-byte NIfTI files.
- Verify rejection of mismatched b-vector lengths.
- Verify robust handling of zero/negative signal voxels without NaN propagation.
