
### [gradient_correction] Automated Decision
**Timestamp**: 2026-04-26 07:41:44
**Component**: preprocessing.gradient_correction
**Decision Maker**: automation
**Status**: implemented

**Decision**: Gradient table validated and corrected

**Rationale**: Applied normalization: True, Applied flips: x=False, y=False, z=False

**Parameters & Thresholds**:
- b0_threshold = 50.0
- normalize_bvecs = True
- flip_x = False
- flip_y = False
- flip_z = False
- num_volumes = 160
- num_shells = 1
- shells = [2000]

---

### [brain_extraction_method] Automated Decision
**Timestamp**: 2026-04-26 07:41:44
**Component**: preprocessing.brain_extraction
**Decision Maker**: automation
**Status**: implemented

**Decision**: Using median_otsu method for brain extraction

**Rationale**: Method selected based on initialization. median_otsu is optimized for DWI data.

**Parameters & Thresholds**:
- method = median_otsu
- median_radius = 4
- num_pass = 4
- autocrop = False
- dilate = 1

---

### [bias_correction_method] Automated Decision
**Timestamp**: 2026-04-26 07:42:01
**Component**: preprocessing.bias_correction
**Decision Maker**: automation
**Status**: implemented

**Decision**: Using polynomial method for bias correction

**Rationale**: Method selected: auto, ANTs available: False

**Parameters & Thresholds**:
- method = polynomial
- convergence_threshold = 0.001
- max_iterations = 50
- n4_shrink_factor = 4

---

### [preprocessing_pipeline_complete] Automated Decision
**Timestamp**: 2026-04-26 07:42:07
**Component**: preprocessing.pipeline
**Decision Maker**: automation
**Status**: implemented

**Decision**: Preprocessing pipeline completed successfully

**Rationale**: All steps executed: motion=No, brain=Yes, bias=Yes

**Parameters & Thresholds**:
- output_dir = output/stanford_test/preprocessed
- output_files = {'dwi': 'output/stanford_test/preprocessed/preprocessed_dwi.nii.gz', 'bval': 'output/stanford_test/preprocessed/preprocessed_dwi.bval', 'bvec': 'output/stanford_test/preprocessed/preprocessed_dwi.bvec', 'mask': 'output/stanford_test/preprocessed/preprocessed_brain_mask.nii.gz', 'checksums': 'output/stanford_test/preprocessed/preprocessed_checksums.json', 'qc_report': 'output/stanford_test/preprocessed/qc/preprocessed_qc_report.txt'}
- pipeline_metrics = {'gradient': {'num_volumes': 160, 'num_b0': 10, 'num_dwi': 150, 'num_shells': 1, 'shells': [2000], 'norm_mean': 0.9999995792895648, 'norm_std': 2.8961722432049147e-06, 'norm_min': 0.9999922788701922, 'norm_max': 1.0000064760290306, 'num_duplicates': 0, 'min_angle_deg': 1.1654683597156625, 'max_angle_deg': 89.9965942769843, 'mean_angle_deg': 5.7040445520405925, 'shell_2000_count': 150}, 'brain_mask': {'total_voxels': 652536, 'brain_voxels': 186830, 'brain_fraction': 0.2863137052974855, 'mean_signal_inside': 203.57645797650272, 'mean_signal_outside': 20.513835539481985, 'std_signal_inside': 73.81013298873418, 'std_signal_outside': 34.32950252557539, 'cnr': 5.332516027595784, 'num_components': 1, 'largest_component_fraction': 1.0}, 'bias_correction': {'cv_before': 0.7457324972316287, 'cv_after': 0.7457324972316287, 'cv_improvement_percent': 0.0, 'mean_intensity_before': 938.2047377599794, 'mean_intensity_after': 938.2047377599794, 'std_intensity_before': 699.6497620042948, 'std_intensity_after': 699.6497620042948}}
- checksums = {'input_dwi': '3762f358606ee2c092ddadddcde5bf0d', 'bias_corrected': '78d05f0d09944a115c8d12ab023cfc0e'}
- processing_time = 0:00:26.179412

---

### [tractography_20260426_081152] Automated Decision
**Timestamp**: 2026-04-26 08:11:52
**Component**: probabilistic_tracking
**Decision Maker**: automation
**Status**: implemented

**Decision**: Tracked 30141 streamlines from 373660 seeds

**Rationale**: Probabilistic tracking with 1 samples per seed. Parameters automatically selected based on data quality. RNG seed 2137564784 ensures reproducibility.

**Parameters & Thresholds**:
- n_seeds = 373660
- n_samples_per_seed = 1
- n_streamlines_kept = 30141
- success_rate = 0.08066424021838034
- step_size = 0.625
- max_angle = 30.0
- fa_threshold = 0.1
- fod_threshold = 0.1
- rng_seed = 2137564784
- tracking_time_seconds = 226.96496176719666

---
