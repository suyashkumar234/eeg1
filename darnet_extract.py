#!/usr/bin/env python
"""
Extract DARNet 8 Flattened Features for Fold 1 (6-Fold CV)
===========================================================

Extract the 8 flattened features from DARNet (after stack2 concatenation and flatten)
before the final classification head.

Structure: subject_id | label | feature_0 | feature_1 | ... | feature_7

For Fold 1:
- Train: Subjects 6-30 (audio-visual data)
- Test: Subjects 1-5 (audio-visual data)

Maintains proper ordering for concatenation with LaBraM and DWT features.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CHECKPOINT_PATH = "/scratch/suyash.kumar.mec22.itbhu/EEG-AAD/eeg-aad-challenge2026-task1-baselines-master/exps/cross-subject/DARNet_Domain_Contra/baseline_task1/fold_1.pt"

# Data paths
DATA_PATHS = [
    "/Users/suyash/Desktop/EEG-AAD/EEG-AAD_audio_visual/preprocessed/data",
    "/scratch/suyash.kumar.mec22.itbhu/EEG-AAD_audio_visual/preprocessed/data",
]
LABEL_PATHS = [
    "/Users/suyash/Desktop/EEG-AAD/EEG-AAD_audio_visual/preprocessed/label",
    "/scratch/suyash.kumar.mec22.itbhu/EEG-AAD_audio_visual/preprocessed/label",
]

# Find correct path
DATA_PATH = None
LABEL_PATH = None
for data_p, label_p in zip(DATA_PATHS, LABEL_PATHS):
    if os.path.exists(data_p) and os.path.exists(label_p):
        DATA_PATH = data_p
        LABEL_PATH = label_p
        break

if DATA_PATH is None:
    raise FileNotFoundError(f"Could not find data paths")

OUTPUT_DIR = "darnet_features_fold1"
TRAIN_CSV = "darnet_features_train_fold1.csv"
TEST_CSV = "darnet_features_test_fold1.csv"

BATCH_SIZE = 64 if torch.cuda.is_available() else 16
RANDOM_SEED = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
else:
    print(f"⚠ GPU not available, using CPU")

print(f"Using device: {DEVICE}\n")

# ============================================================================
# DATASET
# ============================================================================

class MM_AAD_Dataset(Dataset):
    """MM-AAD EEG dataset for feature extraction."""

    def __init__(self, data_path, label_path, subject_ids):
        """Load data for multiple subjects in order."""
        self.data_path = data_path
        self.label_path = label_path
        self.subject_ids = subject_ids

        # Load data for all subjects in order
        self.eeg_data = []
        self.labels = []
        self.subject_indices = []

        for subject_id in subject_ids:
            data_file = f"{data_path}/S{subject_id}.npy"
            label_file = f"{label_path}/S{subject_id}.npy"

            if not os.path.exists(data_file) or not os.path.exists(label_file):
                print(f"⚠ Warning: Data not found for subject {subject_id}, skipping")
                continue

            eeg_data = np.load(data_file)  # (n_trials, 128, 32)
            labels = np.load(label_file)  # (n_trials,)

            # Flatten labels if needed
            if labels.ndim > 1:
                labels = labels.flatten()

            self.eeg_data.append(eeg_data)
            self.labels.append(labels)
            self.subject_indices.append(np.full(len(labels), subject_id))

        # Concatenate all data (maintains order)
        self.eeg_data = np.concatenate(self.eeg_data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.subject_indices = np.concatenate(self.subject_indices, axis=0)

        print(f"  Loaded {len(subject_ids)} subjects")
        print(f"  Total samples: {len(self.labels)}")
        print(f"  Sample ordering verified")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg_data = self.eeg_data[idx]  # (128, 32)
        label = self.labels[idx]
        subject_id = self.subject_indices[idx]

        eeg = torch.FloatTensor(eeg_data)
        label = torch.LongTensor([label])
        subject_id = torch.LongTensor([subject_id])

        return eeg, label, subject_id

# ============================================================================
# MODEL WRAPPER FOR FEATURE EXTRACTION
# ============================================================================

class DARNetFeatureExtractor(nn.Module):
    """Wrapper to extract 8 flattened features from DARNet."""

    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        # Copy the relevant layers from original model
        self.token_embedding = original_model.token_embedding
        self.stack1 = original_model.stack1
        self.stack2 = original_model.stack2
        self.flatten = original_model.flatten

    def forward(self, x):
        """
        Forward pass that returns 8 flattened features.

        x shape: (batch, 128, 32)
        """
        # Transpose to (batch, 32, 128) for model
        x = x.permute(0, 2, 1)  # (batch, 32, 128)

        # Token embedding
        x_src = self.token_embedding(x)

        # Stack 1
        x_src1, new_src1 = self.stack1(x_src)

        # Stack 2
        x_src2, new_src2 = self.stack2(x_src1)

        # Concatenate features from stack1 and stack2
        features = torch.cat([new_src1, new_src2], -1)  # (batch, seq, 8)

        # Flatten to get 8-dimensional feature vector
        features_flat = self.flatten(features)  # (batch, 8)

        return features_flat

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(model, data_loader, device):
    """
    Extract 8 flattened features from DARNet.

    Returns:
        features: (num_samples, 8)
        labels: (num_samples,)
        subject_ids: (num_samples,)
    """
    model.eval()

    all_features = []
    all_labels = []
    all_subject_ids = []

    pbar = tqdm(data_loader, desc="Extracting features")

    with torch.no_grad():
        for eeg, label, subject_id in pbar:
            eeg = eeg.to(device)  # (batch, 128, 32)
            label = label.to(device).squeeze()
            subject_id = subject_id.to(device).squeeze()

            # Forward pass to get 8-dim features
            features = model(eeg)  # (batch, 8)

            # Store
            all_features.append(features.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_subject_ids.append(subject_id.cpu().numpy())

    # Concatenate all batches
    features = np.concatenate(all_features, axis=0)  # (num_samples, 8)
    labels = np.concatenate(all_labels, axis=0)  # (num_samples,)
    subject_ids = np.concatenate(all_subject_ids, axis=0)  # (num_samples,)

    return features, labels, subject_ids

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("EXTRACT DARNET 8 FEATURES FOR FOLD 1 (6-FOLD CV)")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Verify checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"✗ ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        return 1

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Fold 1 configuration
    train_subject_ids = list(range(6, 31))  # [6, 7, ..., 30]
    test_subject_ids = list(range(1, 6))    # [1, 2, 3, 4, 5]

    print("=" * 80)
    print("FOLD 1 CONFIGURATION")
    print("=" * 80)
    print(f"Train subjects (audio-visual): {train_subject_ids}")
    print(f"Test subjects (audio-visual): {test_subject_ids}\n")

    # Load model
    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    try:
        print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        original_model = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        print(f"✓ Loaded checkpoint")

        # Wrap with feature extractor
        model = DARNetFeatureExtractor(original_model)
        model = model.to(DEVICE)
        print(f"✓ Created feature extractor wrapper\n")
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ========================================================================
    # TRAINING SET FEATURE EXTRACTION
    # ========================================================================
    print("=" * 80)
    print("EXTRACTING TRAINING SET FEATURES")
    print("=" * 80)
    print(f"\nLoading data for subjects {train_subject_ids}...")
    train_dataset = MM_AAD_Dataset(
        DATA_PATH, LABEL_PATH, train_subject_ids
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"\nExtracting features...")
    train_features, train_labels, train_subject_ids_array = extract_features(
        model, train_loader, DEVICE
    )

    print(f"\n✓ Training features extracted")
    print(f"  Shape: {train_features.shape}")
    print(f"  Labels shape: {train_labels.shape}")

    # Create DataFrame
    feature_columns = [f"feature_{i}" for i in range(train_features.shape[1])]
    train_df = pd.DataFrame(train_features, columns=feature_columns)
    train_df.insert(0, 'label', train_labels)
    train_df.insert(0, 'subject_id', train_subject_ids_array)

    # Save to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_csv_path = f"{OUTPUT_DIR}/{TRAIN_CSV}"
    train_df.to_csv(train_csv_path, index=False)

    print(f"\n✓ Training CSV saved: {train_csv_path}")
    print(f"  Shape: {train_df.shape}")
    print(f"  Columns: {list(train_df.columns)}")
    print(f"  First few rows:")
    print(train_df.head(3))

    # ========================================================================
    # TEST SET FEATURE EXTRACTION
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("EXTRACTING TEST SET FEATURES")
    print("=" * 80)
    print(f"\nLoading data for subjects {test_subject_ids}...")
    test_dataset = MM_AAD_Dataset(
        DATA_PATH, LABEL_PATH, test_subject_ids
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"\nExtracting features...")
    test_features, test_labels, test_subject_ids_array = extract_features(
        model, test_loader, DEVICE
    )

    print(f"\n✓ Test features extracted")
    print(f"  Shape: {test_features.shape}")
    print(f"  Labels shape: {test_labels.shape}")

    # Create DataFrame
    test_df = pd.DataFrame(test_features, columns=feature_columns)
    test_df.insert(0, 'label', test_labels)
    test_df.insert(0, 'subject_id', test_subject_ids_array)

    # Save to CSV
    test_csv_path = f"{OUTPUT_DIR}/{TEST_CSV}"
    test_df.to_csv(test_csv_path, index=False)

    print(f"\n✓ Test CSV saved: {test_csv_path}")
    print(f"  Shape: {test_df.shape}")
    print(f"  Columns: {list(test_df.columns)}")
    print(f"  First few rows:")
    print(test_df.head(3))

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nTrain CSV: {train_csv_path}")
    print(f"  - Subjects: {list(train_subject_ids)}")
    print(f"  - Samples: {len(train_df)}")
    print(f"  - Columns: subject_id | label | 8 features")

    print(f"\nTest CSV: {test_csv_path}")
    print(f"  - Subjects: {list(test_subject_ids)}")
    print(f"  - Samples: {len(test_df)}")
    print(f"  - Columns: subject_id | label | 8 features")

    print(f"\n✓ Ready for concatenation with LaBraM (200) and DWT (300) features!")
    print(f"  Total hybrid features: 8 + 200 + 300 = 508 dimensions")
    print(f"{'=' * 80}\n")

    return 0

if __name__ == '__main__':
    exit(main())
