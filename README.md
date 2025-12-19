# Thesis Survival Experiments

This repository provides a modular PyTorch framework for training **DeepSurv**, **DeepHit**, and **multimodal survival models** on multiple datasets (Synthetic, SUPPORT2, MNB, MIMIC-Eye).  
Experiments are fully configuration-driven through YAML files.

---

## 0. Setup

---

## 1. Project Structure

alexa_thesis/
│
├── train.py                     # Main experiment runner
├── README.md
│
├── configs/                     # YAML experiment configurations
│   ├── support2_deepsurv.yaml
│   ├── synthetic_deephit.yaml
│   ├── mnb_binary_deepsurv.yaml
│   ├── mimiceye_tabular_deepsurv.yaml
│   └── mimiceye_multimodal_deepsurv.yaml
│
├── datasets/                    # Dataset loaders
│   ├── base.py
│   ├── synthetic.py
│   ├── support2.py
│   ├── mnb.py
│   ├── mimic_eye_tabular.py
│   ├── mimic_eye_multimodal.py
│   └── __init__.py
│
├── models/                      # DeepSurv / DeepHit / multimodal models
│   ├── mlp.py
│   ├── deepcox.py
│   ├── deephit.py
│   ├── multimodal.py
│   └── __init__.py
│
├── trainers/                    # Training logic for Cox & DeepHit
│   ├── supervised_trainer.py
│   └── deephit_trainer.py
│
├── losses/                      # Loss implementations
│   ├── cox.py
│   └── deephit.py
│
└── metrics/                     # Evaluation metrics
    ├── survival_metrics.py      # C-index
    └── classification_metrics.py# PR-AUC, F1-score from risk scores

---

## 2. Data Structure

All datasets must follow the directory structure below:

data/
  synthetic/
    train.csv
    val.csv
    test.csv
    synthetic_true.csv

  support2/
    train.csv
    val.csv
    test.csv

  MNB/
    train.csv
    val.csv
    test.csv

  mimic-eye/
    tabular/
      train.csv
      val.csv
      test.csv
      category_mappings.json
      mimic_eye_survival_admissions.csv

    multimodal/
      train.csv
      val.csv
      test.csv
      multimodal_category_mappings.json
      mimic_eye_multimodal_admissions.csv

Each dataset loader automatically infers feature columns while respecting dataset-specific time/event column names.

---

## 3. How to Run Experiments

Experiments are fully driven by YAML config files inside configs/.

Run any experiment with:

```bash

python train.py --config configs/<config_name>.yaml

```

## 4. Metrics

Each trainer computes and logs **survival** and **classification** metrics.

### Survival Metrics
- **Concordance Index (C-index)**  
  Measures how well the model ranks patients by risk (standard survival evaluation).

### Classification Metrics
Derived from model risk scores (`log_risk` for Cox, `-expected_time` for DeepHit):

- **PR-AUC** (Precision–Recall Area Under Curve)
- **F1-score**
- **Decision threshold** used for F1 (default: median predicted risk)

These metrics complement survivial ranking metrics and provide a binary "event vs. no event" interpretation.

---

## 5. Model Families Supported

### 1. DeepSurv (Cox Proportional Hazards Neural Network)
- MLP backbone for tabular data  
- Optional multimodal fusion with ResNet18 for image inputs (MIMIC-Eye)

### 2. DeepHit
- Supports single-event and competing-risks modeling  
- Uses discrete time bins (`TimeDiscretizer`)  
- Loss: negative log-likelihood over time bins

### 3. Multimodal Cox
- Tabular branch: MLP  
- Image branch: ResNet18  
- Late-fusion Cox head for risk prediction

---

## 6. Configuration System (YAML)

Each config file specifies:

 - dataset settings
 - model architecture
 - optimizer hyperparameters
 - training schedule
 - optional SSL flags (for later extension) ------------------------------------------------------------

Configs enable fully reproducible experiments without modifying code.

---

## 7. Checkpoints and Outputs

During training:

 - Best validation model is tracked via C-index
 - Saved to:
    `runs/<experiment_name>_best.pt`
Test-set metrics are printed after training.