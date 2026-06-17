<div align="center">

# EEG-AAD 2026: Auditory Attention Decoding Challenge
### Team Euler — IIT (BHU) Varanasi × Luleå University of Technology

[![Challenge](https://img.shields.io/badge/Challenge-ICASSP%202026%20EEG--AAD-blue)](https://fchest.github.io/icassp-aad/)
[![Track 1](https://img.shields.io/badge/Track%201%20(Cross--Subject)-Rank%209%20%7C%2052.12%25-green)]()
[![Track 2](https://img.shields.io/badge/Track%202%20(Cross--Session)-Rank%208%20%7C%2050.99%25-orange)]()

</div>

---

## Overview

This repository contains the implementation for the [IEEE ICASSP 2026 EEG-AAD Auditory Attention Decoding Challenge](https://fchest.github.io/icassp-aad/), developed by **Team Euler** under the supervision of **Dr. Rajkumar Saini** at Luleå University of Technology.

The challenge requires decoding the spatial orientation (left/right) of a subject's auditory attention from EEG signals, with two tasks:
- **Task 1 (Cross-Subject):** Generalize to unseen subjects in audio-only settings
- **Task 2 (Cross-Session):** Generalize to unseen audio-visual scenarios for subjects whose audio-only data was seen during training

**Team Euler final results:**

| Track | Task | Accuracy | Rank |
|-------|------|----------|------|
| Track 1 | Cross-Subject | 52.12% | **9** |
| Track 2 | Cross-Session | 50.99% | **8** |

---

## Dataset

The **MM-AAD (Multi-Modal Auditory Attention Decoding)** dataset [1] consists of EEG recordings from 40 subjects in two settings:
- **Audio-only** and **audio-visual** competitive listening scenarios
- Subjects attended to one of two competing voices from ±90° (left/right)
- ~55 minutes per subject → ~73.3 hours of total data

### Preprocessing Pipeline

- Bandpass filter: 0.1 Hz – 50 Hz
- 50 Hz power-line noise removal
- Ocular artifact removal via ICA
- Downsampling to 128 Hz
- Windowing: 1-second windows with 50% overlap → **6,580 decision windows per subject**
- Task 2 additionally: Common Spatial Pattern (CSP) filtering for cross-session generalization

---

## Architecture: DARNet

The baseline architecture is **DARNet** (Dual Attention Refinement Network) [2], a transformer-style model designed for auditory attention detection from EEG.

```
Input EEG (channels × time)
        │
  TokenEmbedding
  ├─ Conv2D spatial: (1, 8) kernel
  ├─ Conv2D channel reduction: (C_in, 1) kernel
  └─ Sinusoidal Positional Embedding
        │
  AttnRefine Stack 1
  ├─ Multi-Head Self-Attention (8 heads, d=16)
  ├─ Conv1D Refinement + MaxPool
  └─ Intermediate output (4-dim)
        │
  AttnRefine Stack 2
  ├─ Multi-Head Self-Attention (8 heads, d=16)
  ├─ Conv1D Refinement + MaxPool
  └─ Intermediate output (4-dim)
        │
  Concatenate stack outputs → Linear(8, 2)
        │
  Softmax → Left / Right
```

- **Task 1:** DARNet without CSP
- **Task 2:** DARNet with CSP preprocessing

---

## Training Details

### Task 1 — Cross-Subject

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 3e-4 |
| Batch size | 128 |
| Epochs | 100 |
| LR schedule | MultiStepLR (milestones: [10, 35], γ=0.5) |
| Loss | CrossEntropy |

**Data split:**
- Training subjects: 26 (from IDs 1–30, excluding val set)
- Validation subjects: 4 held-out (IDs: 1, 2, 3, 6)
- Test subjects: 10 unseen (IDs: 31–40)

### Task 2 — Cross-Session

- Training: audio-only scenario data for subjects 1–30
- Validation: audio-visual scenario data for the same 30 subjects
- Test: 10 unseen subjects (audio-only pre-training provided, tested on audio-visual)

---

## Team Contributions (beyond baseline)

The following improvements were explored beyond the provided baseline during the ICASSP 2026 competition:

- **Enhanced preprocessing:** Refined ICA ocular artifact removal and bandpass filter tuning
- **CSP integration for Task 2:** Applied Common Spatial Patterns filtering for cross-session domain adaptation
- **Validation strategy exploration:** Experimented with different held-out subject partitioning schemes as recommended by the challenge organizers
- **Submission pipeline:** End-to-end inference script generating the required per-subject CSV files for both tasks

---

## Repository Structure

```
.
├── trainer.py          # Main training loop (cross-subject)
├── model_module.py     # DARNet architecture (Attention, TokenEmbedding, AttnRefine)
├── utils.py            # Dataset class, data loader, model save/load utilities
├── inference.py        # Inference script generating submission CSVs
├── requirements.txt    # Python dependencies
├── run.sh              # SLURM batch script for cluster training
└── exps/
    └── cross-subject/
        └── DARNet/     # Saved model checkpoints and figures
```

---

## How to Run

### Requirements

```bash
Python 3.9
pip install -r requirements.txt
```

Tested on a single NVIDIA RTX 4090 GPU. Training was run on SLURM-managed GPU clusters.

### Training

```bash
python trainer.py
```

Best model checkpoint (by validation accuracy) is saved to `exps/cross-subject/DARNet/`.

### Inference

Generates the required CSV files for final submission:

```bash
python inference.py --model DARNet --resume exps/cross-subject/DARNet/baseline_2025-02-28-01-39-14
```

**Output files:**
- Task 1: `results_cross_subject_test_subject.csv`
- Task 2: `results_cross_session_test_subject.csv`

Each CSV contains:
- `id`: sample ID
- `prediction`: predicted class (0 = right, 1 = left)

### SLURM Cluster

```bash
sbatch run.sh
```

Update paths in `run.sh` and `trainer.py` to match your cluster environment.

---

## Baseline vs. Team Euler Results

| Model | Task | Val Acc (Baseline) | Val Acc (Team Euler) |
|---|---|---|---|
| DARNet (no CSP) | Cross-Subject | 53.1% | 52.12% (test) |
| DARNet (CSP) | Cross-Session | 57.33% | 50.99% (test) |

> Note: Baseline numbers are on the validation set; our results are final held-out test set scores reported by the challenge leaderboard.

---

## References

[1] Cunhang Fan et al. "Seeing helps hearing: A multi-modal dataset and a mamba-based dual branch parallel network for auditory attention decoding." *Information Fusion*, 2025.

[2] Sheng Yan et al. "DARNet: Dual Attention Refinement Network with Spatiotemporal Construction for Auditory Attention Detection." *NeurIPS*, 37:31688–31707, 2024.

---

## Citation

If you use this code, please cite the original DARNet paper and the MM-AAD dataset as above, and acknowledge the ICASSP 2026 EEG-AAD Challenge.

---

## Acknowledgements

This work was carried out under the supervision of **Dr. Rajkumar Saini** at Luleå University of Technology, Sweden, as part of a research internship from IIT (BHU) Varanasi.
