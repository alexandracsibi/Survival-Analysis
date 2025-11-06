# Datasets

This directory lists and documents the datasets used in the thesis project.

---

## Overview
The datasets in this project support research on integrating semi-supervised learning approaches into survival analysis.  
The focus is on leveraging censored and partially labeled data to improve prediction accuracy and robustness in event-time modeling.

---

## Datasets Used

### 1. MIMIC-Eye (Restricted Access)
Multimodal clinical dataset containing structured hospital data, imaging (CXR), and radiology reports.  
Access provided through PhysioNet credentialed user agreement.  
No data is stored in this repository; only scripts reference the approved local dataset.

#### Selected Files

| Folder | File | Purpose |
|---------|------|----------|
| `Hosp/` | `admissions.csv`, `diagnoses_icd.csv`, `labevents.csv` | Demographics, diagnoses, and lab data for survival feature extraction. |
| `ICU/` | `icustays.csv` | ICU stay metadata and outcomes. |
| `ED/` | `triage.csv`, `vitalsign.csv` | Vital signs and acuity scores at admission. |
| `CXR-JPG/` | `cxr_meta.csv`, `cxr_chexpert.csv`, `{dicom_id}.jpg` | Chest X-ray images with metadata and image-level labels. |
| `CXR-DICOM/` | `{study_id}.txt` | Associated radiology reports. |

These components form the structured, imaging, and textual feature bases used for multimodal survival modeling.

**Source:** [https://physionet.org/content/mimic-eye](https://physionet.org/content/mimic-eye)

---

### 2. SUPPORT2 Dataset (Open Access)
Clinical survival dataset containing demographic, physiological, and outcome data from hospitalized patients.  
Used for baseline model benchmarking.

**Source:** [https://hbiostat.org/data/](https://hbiostat.org/data/)

---

### 3. DeepHit Synthetic Dataset
Synthetic survival dataset for controlled model validation and reproducibility testing.  
Used to evaluate semi-supervised learning performance under simulated censoring conditions.

**Source:** [https://github.com/chl8856/DeepHit](https://github.com/chl8856/DeepHit)
