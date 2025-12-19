# Thesis Survival Experiments

This repository contains a modular PyTorch framework for training and evaluating neural survival models on multiple real-world and synthetic datasets.
All experiments are fully configuration-driven using YAML files, enabling reproducible research.

Supported datasets: Synthetic (DeepHit), SUPPORT2, MNB (financial risk), MIMIC-Eye (tabular & multimodal)

---

## 1. Data Structure

All datasets must follow the directory structure below:

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
      multimodal/
        train.csv
        val.csv
        test.csv
        multimodal_category_mappings.jso

Each dataset loader:

 - infers feature columns automatically
 - respects dataset-specific time and event definitions
 - handles censoring consistently across models

---

## 3. How to Run Experiments

Experiments are fully driven by YAML config files inside configs/.

Run any experiment with:

```bash

python train.py --config configs/<config_name>.yaml

```

## 4. Models Implemented

### DeepSurv

Neural Cox proportional hazards model:

 - MLP backbone for tabular features
 - Outputs a continuous risk score
 - Optimized via partial log-likelihood

### DeepHit

Discrete-time survival model:

- Supports single-event and competing risks
- Learns event-time distributions over time bins
- Optimized via negative log-likelihood

### Multimodal Cox

- Tabular branch: MLP
- Image branch: ResNet-18
- Late-fusion Cox head for risk prediction

## 5. Evaluation Metrics

Each trainer computes and logs and **survival** metrics.

### Primary Metric
- **Concordance Index (C-index)**  
 Measures how well predicted risks rank individuals by event time.

### Additional Metrics

 - **Time-dependent AUC**
 - **Integrated AUC (iAUC)**
 - **Integrated Brier Score (IBS))**

These metrics complement survivial ranking metrics and and used for auxiliary analysis, not model selection.

---

## 6. Configuration System (YAML)

Each config file specifies:

 - dataset and preprocessing options
 - model architecture
 - optimizer hyperparameters
 - training schedule
 - evaluation horizon and metrics
 - optional semi-supervised flags

Configs enable fully reproducible experiments without modifying code.

---

## 7. Checkpoints and Outputs

During training:

 - Best model is selected based on validation C-index
 - Saved to:
    `runs/<experiment_name>_best.pt`
Test-set metrics are printed after training.
