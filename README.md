<div align="center">

# EEG-AAD 2026: Auditory Attention Decoding — Track 1 (Cross-Subject)
### Team Euler — IIT (BHU) Varanasi × Luleå University of Technology

[![Challenge](https://img.shields.io/badge/Challenge-ICASSP%202026%20EEG--AAD-blue)](https://fchest.github.io/icassp-aad/)
[![Track 1](https://img.shields.io/badge/Track%201%20Cross--Subject-Rank%209%20%7C%2052.12%25-green)]()

</div>

---

## Overview

This repository contains the Track 1 (Cross-Subject) implementation for the [IEEE ICASSP 2026 EEG-AAD Auditory Attention Decoding Challenge](https://fchest.github.io/icassp-aad/), developed by **Team Euler** under the supervision of **Dr. Rajkumar Saini** at Luleå University of Technology.

**Task:** Given EEG signals from a subject never seen during training, decode the spatial orientation (left/right) of their auditory attention to one of two competing speakers.

**Team Euler — Track 1 result: 52.12% accuracy, Rank 9 out of 43 teams.**

---

## Leaderboard (Track 1 — Cross-Subject, 43 teams)

| Rank | Team | Score (%) | Std |
|---|---|---|---|
| 1 | TencentAILAB-IACAS | 56.50 | 5.02 |
| 2 | poly | 54.41 | 5.16 |
| 3 | Invincible | 53.79 | 5.62 |
| 4 | CPInS Lab NUST | 53.65 | 3.19 |
| 5 | HS_WC | 53.62 | 4.22 |
| 6 | Lab9 Tencent | 52.50 | 10.06 |
| 7 | CityUDG HANI Lab | 52.42 | 2.94 |
| 8 | SmileHnu | 52.35 | 2.20 |
| **9** | **Euler (ours)** | **52.12** | **6.66** |
| 10 | AAD_Decoding | 51.94 | 6.72 |
| 11 | PeRCeiVe Lab | 51.85 | 7.91 |
| 12 | Gradient Blade | 51.62 | 5.49 |
| 13 | XDataD Sync Neural | 51.60 | 4.31 |
| 14 | LuckyEEG | 51.57 | 3.87 |
| 15 | CLEEG | 51.43 | 3.96 |
| 16 | HUST-BCI | 51.32 | 2.46 |
| 17 | NeuEEG | 50.97 | 7.93 |
| 18 | CPRL | 50.91 | 5.97 |
| … | … | … | … |
| 37 | **Baseline (official)** | 49.63 | 2.89 |
| … | … | … | … |
| 43 | ECNU | 49.06 | 5.41 |

Team Euler outperforms the official challenge baseline by **+2.49%** on the held-out test set.

---

## Dataset

The **MM-AAD (Multi-Modal Auditory Attention Decoding)** dataset consists of EEG recordings from 40 subjects in an audio-only competitive listening scenario:
- Subjects attended to one of two competing voices at ±90° (left/right)
- ~55 minutes per subject → ~73.3 hours total
- 6,580 decision windows per subject after windowing

### Preprocessing

- Bandpass filter: 0.1 – 50 Hz
- 50 Hz power-line noise removal (notch filter)
- Ocular artifact removal via ICA
- Downsampling to 128 Hz

### Data Split

| Split | Subject IDs | Windows |
|---|---|---|
| Training | 1–30 (excl. val) | ~26 × 6,580 |
| Validation (held-out) | 1, 2, 3, 6 | ~4 × 6,580 |
| Test (unseen) | 31–40 | ~10 × 6,580 |

---

## Architecture: DARNet

DARNet (Dual Attention Refinement Network) processes EEG windows through spatial-temporal encoding followed by two stacked attention-refinement modules.

```
INPUT: [batch, 32 channels, 128 timesteps]
        │
  TokenEmbedding
  ├─ Conv2D #1: 1 → 64 channels  (kernel 1×8, temporal encoding)
  ├─ Conv2D #2: 64 → 16 channels (kernel 32×1, spatial/channel reduction)
  ├─ Sinusoidal Positional Embedding
  └─ Output: [batch, 128 timesteps, 16 features]
        │
  AttnRefine Stack 1
  ├─ Multi-Head Self-Attention (8 heads, d=16)
  ├─ Conv1D + BatchNorm + ELU + MaxPool1d (stride=2)
  ├─ Feature output: [batch, 64 timesteps, 16]
  └─ Intermediate classification head → [batch, 4]
        │
  AttnRefine Stack 2
  ├─ Multi-Head Self-Attention (8 heads, d=16)
  ├─ Conv1D + BatchNorm + ELU + MaxPool1d (stride=2)
  ├─ Feature output: [batch, 32 timesteps, 16]
  └─ Intermediate classification head → [batch, 4]
        │
  Concatenate: [batch, 4] + [batch, 4] → [batch, 8]
  Linear(8 → 2) → Left / Right
```

Total parameters: ~280K. Training time: ~30s/epoch on RTX 4090.

---

## Branch Summary

Two branches represent different experimental strategies for cross-subject generalization:

| Branch | Strategy | Val Acc | Test Acc |
|---|---|---|---|
| `main` | DARNet baseline | 53.1% | **49.63%** (baseline)  |
| `subject-adversarial-contrastive` | DARNet + Gradient Reversal + InfoNCE | 55.18% | **52.12%** (submitted) |

The `subject-adversarial-contrastive` branch was used for the final submission. The adversarial-contrastive branch achieved higher validation accuracy on cross-validation.

---

## Branch 1: `main` — Baseline DARNet

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 3e-4 |
| Batch size | 128 |
| Window length | 1 second (128 samples) |
| Overlap | 50% |
| Epochs | 100 |
| LR schedule | MultiStepLR — milestones [10, 35], γ=0.5 |
| Loss | CrossEntropyLoss |

### Training Dynamics

```
Epoch 1:   Train Loss 0.693 | Train Acc 50.2% | Val Acc 50.8%
Epoch 10:  LR → 2.5e-4     | Train Acc 73.4% | Val Acc 52.1%  ← checkpoint
Epoch 35:  LR → 1.25e-4    | Train Acc 78.9% | Val Acc 53.1%  ← best checkpoint
Epoch 100: Train Acc 85.2%  | Val Acc 52.8%   (overfitting onset)
Final test (best model loaded): 52.12%
```

---

## Branch 2: `subject-adversarial-contrastive` — Experimental

This branch extends DARNet with two objectives targeting cross-subject generalization.

### Gradient Reversal (Domain Adversarial Training)

A `GradientReversalLayer` is placed between the feature extractor and a `SubjectDiscriminator`. During backpropagation, gradients are negated — forcing the feature extractor to learn subject-invariant representations.

```
Features [batch, 8]
       │
  GradientReversalLayer  (grad × −λ during backward; λ anneals 0.0 → 0.1)
       │
  SubjectDiscriminator
  ├─ Linear(8 → 256) + ReLU + Dropout(0.5)
  ├─ Linear(256 → 128) + ReLU + Dropout(0.3)
  └─ Linear(128 → 30)   ← predicts subject ID 0–29
```

### InfoNCE Contrastive Loss

A `ContrastiveHead` projects features into 128-dim space. The InfoNCE loss pulls together samples sharing the same attention direction (positive pairs) and pushes apart opposite-direction samples (negative pairs):

```
L_contrastive = -log( exp(sim_pos / τ) / Σ exp(sim_neg / τ) )    τ = 0.07
```

### Combined Loss

```
L_total = L_attention  +  0.1 × L_subject  +  0.8 × L_contrastive
```

### Key Differences vs. `main`

| Parameter | `main` | `subject-adversarial-contrastive` |
|---|---|---|
| Window length | 1s (128 samples) | 2s (256 samples) |
| Batch size | 128 | 64 |
| Optimizer | AdamW | SGD + momentum 0.9 |
| Learning rate | 5e-4 | 1e-2 |
| λ_domain | — | 0.1 (annealed) |
| λ_contrastive | — | 0.8 |
| Temperature τ | — | 0.07 |
| **Val Acc** | 53.1% | **55.18%** |

---

## How to Run

### Requirements

```bash
Python 3.9
pip install -r requirements.txt
```

Tested on NVIDIA V100 Tesla. Training also run on SLURM-managed GPU clusters.

### Training

```bash
# Baseline (main branch)
python trainer.py

# Adversarial + Contrastive
git checkout subject-adversarial-contrastive
python trainer.py
```

Best checkpoint saved to `exps/cross-subject/DARNet/` by validation accuracy.

### Inference

Generates the submission CSV for Track 1:

```bash
python inference.py --model DARNet --resume exps/cross-subject/DARNet/baseline_2025-02-28-01-39-14
```

Output: `results_cross_subject_test_subject.csv`

Each CSV has two columns:
- `id` — sample ID
- `prediction` — predicted class (0 = Right, 1 = Left)

### SLURM Cluster

```bash
sbatch run.sh
```

Update the data paths in `run.sh` and `trainer.py` to match your cluster environment.

---

## Repository Structure

```
.
├── trainer.py          # Training loop (cross-subject)
├── model_module.py     # DARNet + adversarial/contrastive extensions
├── utils.py            # Dataset class, data loader, model save/load
├── inference.py        # Generates Track 1 submission CSV
├── requirements.txt    # Python dependencies
├── run.sh              # SLURM batch script
└── exps/
    └── cross-subject/
        └── DARNet/     # Checkpoints and training curves
```

---

## References

[1] Cunhang Fan et al. "Seeing helps hearing: A multi-modal dataset and a mamba-based dual branch parallel network for auditory attention decoding." *Information Fusion*, 2025.

[2] Sheng Yan et al. "DARNet: Dual Attention Refinement Network with Spatiotemporal Construction for Auditory Attention Detection." *NeurIPS*, 37:31688–31707, 2024.

---

## Acknowledgements

This work was carried out under the supervision of **Dr. Rajkumar Saini** at Luleå University of Technology, Sweden, as part of a research internship from IIT (BHU) Varanasi.
