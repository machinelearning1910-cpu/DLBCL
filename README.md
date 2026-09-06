# FairLoRA-Enabled Federated Multi-Task Learning for Intelligent Sepsis Management

This repository provides the implementation associated with the paper:

**FairLoRA-Enabled Federated Multi-Task Learning for Intelligent Sepsis Management**

The framework is designed for intelligent sepsis management in distributed healthcare environments by combining **Transformer-based temporal clinical modeling**, **Federated Learning**, **fairness-aware Low-Rank Adaptation (FairLoRA)**, and **Multi-Task Learning** within a unified architecture.

---

## Overview

Sepsis prediction from intensive care unit (ICU) records is challenging because clinical measurements are temporal, patient populations differ across hospitals, and healthcare data cannot be freely centralized because of privacy concerns. In addition, conventional deep-learning models can introduce substantial communication and computational overhead when deployed in federated settings.

The proposed framework addresses these challenges through:

1. **Transformer-based temporal modeling** of longitudinal ICU records.
2. **Federated learning with FedAvg** for collaborative training across distributed hospitals without sharing raw patient data.
3. **FairLoRA** for fairness-aware and parameter-efficient model adaptation.
4. **Multi-task learning** for simultaneous sepsis prediction, recovery forecasting, and organ failure risk assessment.
5. **Fairness evaluation** across age groups, gender categories, and participating hospitals.

---

<p align="center">
  <img src="figures/overall_workflow.png" width="900">
</p>

<p align="center">
  <b>Overall workflow of the proposed FairLoRA-enabled federated multi-task sepsis framework.</b>
</p>

---

## Methodology

### 1. Clinical Data Preprocessing

The framework operates on multivariate ICU time-series records from the PhysioNet/Computing in Cardiology Challenge 2019 dataset.

The preprocessing pipeline includes:

- missing-value handling using median imputation,
- Z-score normalization of continuous clinical variables,
- retention of 37 clinical features after preprocessing,
- construction of 24-hour temporal patient sequences,
- preparation of labels for the three clinical prediction tasks.

### 2. Transformer-Based Temporal Encoder

A Transformer encoder is used to model temporal dependencies across longitudinal ICU observations. The encoder learns patient representations from 24-hour clinical sequences and captures changes associated with disease progression.

### 3. FairLoRA Adaptation

FairLoRA introduces low-rank trainable adaptation parameters while keeping the main model parameters largely fixed. This reduces the number of trainable parameters and lowers communication overhead during federated optimization while supporting fairness-aware learning.

### 4. Federated Learning

The federated environment consists of **five heterogeneous hospitals** with non-IID demographic distributions.

Each hospital:

- maintains its patient data locally,
- receives the current global model,
- performs local model training,
- transmits model updates rather than raw clinical records.

The central server aggregates local updates using **Federated Averaging (FedAvg)** to produce the updated global model.

### 5. Multi-Task Prediction

A shared patient representation is used to jointly support three clinically relevant tasks:

- **Sepsis prediction**
- **Recovery forecasting**
- **Organ failure risk assessment**

The multi-task formulation enables the model to learn shared clinical information while maintaining task-specific prediction outputs.

### 6. Fairness Evaluation

Fairness is assessed across heterogeneous demographic groups and participating hospitals using:

- Demographic Parity Difference (DPD)
- Equal Opportunity Difference (EOD)
- False Positive Rate Gap (FPR Gap)
- AUROC disparity analysis

---

## Dataset

Experiments were conducted using the **PhysioNet/Computing in Cardiology Challenge 2019** sepsis dataset.

The original dataset contains multivariate ICU time-series measurements, including physiological variables, laboratory biomarkers, demographic information, and hospitalization-related attributes. The proposed framework retains **37 clinical features** after preprocessing and represents each patient using a **24-hour temporal sequence**.

Dataset:

https://physionet.org/content/challenge-2019/1.0.0/

---

## Federated Experimental Setup

| Parameter | Setting |
|---|---|
| Federated framework | FedAvg |
| Number of hospitals | 5 |
| Temporal encoder | Transformer Encoder |
| Fairness module | FairLoRA |
| Number of tasks | 3 |
| Clinical features | 37 |
| Sequence length | 24 hours |
| Local epochs | 30 |
| Communication rounds | 10 |
| Batch size | 128 |
| Optimizer | AdamW |
| Learning-rate scheduler | Cosine Annealing |
| Loss formulation | Multi-Task Loss |

---

## Results

The proposed framework achieved the following overall predictive performance:

| Metric | Performance |
|---|---:|
| Accuracy | **93.26%** |
| AUROC | **0.824** |
| AUPRC | **0.978** |
| Precision | **0.882** |
| Recall | **0.915** |
| F1-score | **0.898** |

### Cross-Hospital Performance

The model maintained stable performance across five hospitals with heterogeneous non-IID demographic distributions. The largest reported cross-hospital AUROC difference was **0.060**.

<p align="center">
  <img src="figures/cross_hospital_roc.png" width="760">
</p>

<p align="center">
  <b>Cross-hospital ROC analysis of the proposed FairLoRA-enabled framework.</b>
</p>

### Parameter Efficiency

FairLoRA reduced the number of trainable parameters from **1,216,135** to **123,271**, corresponding to an **89.86% reduction** relative to standard federated training.

<p align="center">
  <img src="figures/parameter_efficiency.png" width="700">
</p>

<p align="center">
  <b>Trainable parameter comparison between standard FedAvg and FairLoRA.</b>
</p>

### Fairness Analysis

The framework was evaluated across age groups, gender categories, and participating hospitals to examine demographic consistency and subgroup disparities.

<p align="center">
  <img src="figures/fairness_metrics.png" width="760">
</p>

<p align="center">
  <b>Fairness-gap analysis across demographic attributes.</b>
</p>

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Running the Framework

Place the PhysioNet Sepsis Challenge data in your project data directory and update the dataset path in your training script.

Example:

```bash
python main.py
```

The implementation should follow the experimental configuration reported in the paper: five non-IID hospitals, 30 local epochs, 10 communication rounds, batch size 128, AdamW optimization, and cosine-annealing learning-rate scheduling.

---

## Suggested Repository Structure

```text
YOUR_REPOSITORY_NAME/
│
├── README.md
├── requirements.txt
├── main.py
│
├── figures/
│   ├── overall_workflow.png
│   ├── cross_hospital_roc.png
│   ├── parameter_efficiency.png
│   └── fairness_metrics.png
│
└── data/
    └── README.md
```

The PhysioNet dataset itself should not be uploaded to the repository unless its license and distribution terms explicitly permit redistribution. The repository can instead provide the official dataset link and instructions for obtaining it.

---

## Key Features

- Privacy-preserving federated training across multiple hospitals
- Transformer-based temporal representation learning
- FairLoRA-based parameter-efficient adaptation
- Demographic fairness evaluation
- Non-IID federated healthcare simulation
- Simultaneous three-task clinical prediction
- Cross-hospital generalization analysis
- Parameter-efficiency evaluation

---

## Citation

If you use this work, please cite the associated paper. Replace the placeholder author information below with the final publication details when available.

```bibtex
@article{fairlora_sepsis_2026,
  title   = {FairLoRA-Enabled Federated Multi-Task Learning for Intelligent Sepsis Management},
  author  = {Authors},
  year    = {2026}
}
```

---

## Acknowledgment

This work uses data from the **PhysioNet/Computing in Cardiology Challenge 2019** for early prediction of sepsis from clinical data.
