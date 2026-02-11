# NeuroTract Dataset Requirements

This document specifies all datasets required for NeuroTract development and production use. **IMPORTANT**: Most datasets require manual registration and download due to data use agreements and privacy policies.

---

## ⚠️ MANUAL DOWNLOAD REQUIRED

The following datasets **CANNOT** be automatically downloaded and require manual registration, application, or credential-based access. You must download these manually before proceeding with pipeline execution.

---

## 1. Human Connectome Project (HCP) - PRIMARY DATASET

**Status**: ⚠️ **REQUIRES MANUAL REGISTRATION & DOWNLOAD**

### Access Requirements
- **Registration URL**: https://db.humanconnectome.org/
- **Access Level**: Open Access (requires free registration and data use agreement acceptance)
- **Credentials**: ConnectomeDB account required

### Dataset Specifications
- **Recommended Subset**: HCP 1200 Subjects Release - Diffusion MRI subset
- **Target**: ~100-200 subjects for development, full 1200 for production
- **Storage Required**: ~500 GB for recommended subset
- **Data Type**: Minimally preprocessed diffusion MRI (dMRI)

### Files Needed (per subject)
```
<subject_id>/
├── T1w/
│   ├── Diffusion/
│   │   ├── data.nii.gz          # 4D diffusion-weighted images
│   │   ├── bvals                # b-values (diffusion weighting)
│   │   ├── bvecs                # gradient directions
│   │   ├── nodif_brain_mask.nii.gz  # brain mask
│   │   └── grad_dev.nii.gz      # gradient deviation (optional)
│   └── T1w_acpc_dc_restore_brain.nii.gz  # Anatomical reference
└── MNINonLinear/
    └── fsaverage_LR32k/
        ├── <subject>.L.midthickness.32k_fs_LR.surf.gii  # Left cortical surface
        └── <subject>.R.midthickness.32k_fs_LR.surf.gii  # Right cortical surface
```

### Example Subject IDs
- 100307, 100408, 101107, 101309, 101410 (use for testing)

### Download Instructions
1. Create account at https://db.humanconnectome.org/
2. Accept Data Use Terms
3. Navigate to "WU-Minn HCP Data - 1200 Subjects"
4. Select subjects and download via:
   - **Aspera Connect** (fastest, recommended for large downloads)
   - **AWS S3** (requires AWS credentials)
   - **Direct download** (slower)
5. Place downloaded data in: `datasets/HCP/`

### Checksum Verification
After download, verify file integrity:
```bash
md5sum datasets/HCP/100307/T1w/Diffusion/data.nii.gz
# Expected: <will be provided after manual download>
```

---

## 2. ADNI (Alzheimer's Disease Neuroimaging Initiative)

**Status**: ⚠️ **REQUIRES APPLICATION & APPROVAL**

### Access Requirements
- **Application URL**: https://adni.loni.usc.edu/data-samples/access-data/
- **Access Level**: Requires formal application and approval (2-4 weeks processing time)
- **Credentials**: LONI IDA account required after approval
- **Data Use Agreement**: Must be signed and approved

### Dataset Specifications
- **Target**: ADNI2 and ADNI3 diffusion MRI scans
- **Subjects Needed**: ~50-100 subjects across diagnostic groups:
  - Cognitively Normal (CN): 20-30 subjects
  - Mild Cognitive Impairment (MCI): 20-30 subjects
  - Alzheimer's Disease (AD): 20-30 subjects
- **Storage Required**: ~100-200 GB
- **Data Type**: Raw diffusion MRI + clinical metadata

### Files Needed (per subject/timepoint)
```
<subject_id>/
├── <scan_date>/
│   ├── DIFFUSION/
│   │   ├── *.dcm or *.nii.gz    # DICOM or NIfTI diffusion scans
│   │   ├── *.bval
│   │   └── *.bvec
│   └── MPRAGE/
│       └── *.nii.gz              # T1-weighted anatomical
└── clinical_data.csv             # Diagnosis, cognitive scores, demographics
```

### Download Instructions
1. Submit application at https://adni.loni.usc.edu/data-samples/access-data/
2. Wait for approval email (2-4 weeks)
3. Log in to LONI Image & Data Archive (IDA)
4. Use "Advanced Search" to filter:
   - Imaging Protocol: DTI or dMRI
   - Image Type: DIFFUSION
5. Download selected subjects to: `datasets/ADNI/`

### Clinical Data
- Download clinical assessments from ADNI Study Data
- Required variables: Diagnosis (DX), MMSE, CDR, age, sex, education, APOE genotype

---

## 3. UK Biobank (Optional)

**Status**: ⚠️ **REQUIRES FORMAL APPLICATION (Restricted Access)**

### Access Requirements
- **Application URL**: https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access
- **Access Level**: Restricted - requires research proposal and institutional approval
- **Processing Time**: 2-3 months
- **Cost**: Application and data access fees apply

### Dataset Specifications
- **Target**: Brain MRI subset with diffusion imaging
- **Subjects**: 40,000+ with brain MRI (subset with dMRI available)
- **Storage Required**: Variable, potentially TBs for full access
- **Note**: Due to access restrictions, **UK Biobank is OPTIONAL** for NeuroTract

### Alternative Recommendation
If UK Biobank access is not feasible, use:
- **IXI Dataset** (https://brain-development.org/ixi-dataset/) - Public, no registration
- **OpenNeuro datasets** (https://openneuro.org/) - Public diffusion MRI datasets

---

## 4. Vercel Skill Agent Rules

**Status**: ⚠️ **LOCATION UNKNOWN - REQUIRES CLARIFICATION**

### Purpose
Frontend build, code style, and deployment conventions for the web UI component.

### Expected Content
- Build configuration guidelines
- Code style rules
- Deployment best practices
- React/Next.js conventions

### Action Required
**QUESTION FOR USER**: Please provide:
1. URL or path to Vercel skill agent markdown rules
2. If no such document exists, we will use standard Vercel/Next.js best practices

---

## 5. Public Test Datasets (No Registration Required)

For **immediate development and testing** without waiting for approvals, use these public datasets:

### A. Stanford HARDI Dataset
- **URL**: http://purl.stanford.edu/ng782rw8378
- **Access**: Direct download, no registration
- **Size**: ~500 MB
- **Content**: Single subject HARDI data (165 directions)
- **Use**: Algorithm development and testing

### B. OpenNeuro Dataset ds000221 (MRI)
- **URL**: https://openneuro.org/datasets/ds000221
- **Access**: Direct download via openneuro-cli or browser
- **Size**: Variable
- **Content**: Multi-subject structural and diffusion MRI
- **Use**: Pipeline testing

### C. DIPY Sample Data
- **URL**: Included with DIPY installation
- **Access**: `dipy.data.fetch_*()` functions
- **Size**: ~100 MB total
- **Content**: Small test datasets (ScanSTan, Sherbrooke, etc.)
- **Use**: Unit testing and algorithm validation

### Download Commands
```python
# In Python with DIPY installed
from dipy.data import fetch_stanford_hardi, fetch_sherbrooke_3shell
stanford_data = fetch_stanford_hardi()
sherbrooke_data = fetch_sherbrooke_3shell()
```

---

## Dataset Directory Structure

After manual downloads, organize datasets as follows:

```
datasets/
├── README.md (this file)
├── HCP/
│   ├── 100307/
│   ├── 100408/
│   └── ...
├── ADNI/
│   ├── subject_001/
│   ├── subject_002/
│   └── clinical_data.csv
├── public_test/
│   ├── stanford_hardi/
│   ├── dipy_data/
│   └── openneuro/
└── checksums.md (file hashes for verification)
```

---

## Checksums and Verification

After downloading, create `datasets/checksums.md` with file hashes:

```bash
# Generate checksums for verification
cd datasets/
find . -type f -name "*.nii.gz" -o -name "*.bval" -o -name "*.bvec" | xargs md5sum > checksums.md
```

---

## Storage Requirements Summary

| Dataset | Size | Access | Priority | Status |
|---------|------|--------|----------|--------|
| HCP (subset) | ~500 GB | Manual | HIGH | ⚠️ Required |
| ADNI | ~100-200 GB | Manual | HIGH | ⚠️ Required |
| UK Biobank | TBs | Manual | LOW | Optional |
| Public Test | ~1 GB | Auto | HIGH | ✓ Automated |

**Total Required**: ~600-700 GB for development and production

---

## Next Steps

1. **IMMEDIATE**: Register for HCP and ADNI access
2. **Download**: Start with small test datasets (5-10 subjects) for development
3. **Verify**: Run checksum verification on all downloaded files
4. **Organize**: Follow directory structure above
5. **Notify**: Confirm "downloaded" when manual datasets are ready

---

## Automated Dataset Fetcher (for public data only)

A dataset fetcher script will be provided in `src/backend/data/fetch_datasets.py` that can automatically download public test datasets. This script will:
- Download Stanford HARDI
- Fetch DIPY sample data
- Validate file integrity
- **NOT** attempt to download HCP/ADNI (requires manual access)

---

## Questions or Issues?

- **HCP Access**: support@humanconnectome.org
- **ADNI Access**: adni@loni.usc.edu
- **UK Biobank**: access@ukbiobank.ac.uk
- **NeuroTract Issues**: See repository issue tracker
