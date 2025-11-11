import sys
import os

LABRAM_PATH = "/scratch/suyash.kumar.mec22.itbhu/eeg2/LaBraM"
if LABRAM_PATH not in sys.path:
    sys.path.insert(0, LABRAM_PATH)
    print(f"Using LaBraM from: {LABRAM_PATH}\n")


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


LABRAM_CHECKPOINT = "/scratch/suyash.kumar.mec22.itbhu/EEG-AAD/eeg-aad-challenge2026-task1-baselines-master/labram_crosssubject_6fold_cv_checkpoints/labram_fold_1.pth"

if LABRAM_CHECKPOINT is None:
    raise FileNotFoundError(f"Fine-tuned LaBraM checkpoint not found in: {LABRAM_CHECKPOINT_PATHS}")

TRAIN_DATA_PATH = "/scratch/suyash.kumar.mec22.itbhu/EEG-AAD_audio_visual/preprocessed/data"
TRAIN_LABEL_PATH = "/scratch/suyash.kumar.mec22.itbhu/EEG-AAD_audio_visual/preprocessed/label"


if TRAIN_DATA_PATH is None or TRAIN_LABEL_PATH is None:
    raise FileNotFoundError(f"Data paths not found in: {TRAIN_DATA_PATH_OPTIONS}")

OUTPUT_DIR = "labram_finetuned_features_csv_fold1"
TRAIN_CSV = "labram_features_train_fold1.csv"
TEST_CSV = "labram_features_test_fold1.csv"

BATCH_SIZE = 64 if torch.cuda.is_available() else 16
RANDOM_SEED = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
else:
    print(f"GPU not available, using CPU")

print(f"Using device: {DEVICE}\n")


class MM_AAD_Dataset(Dataset):
    def __init__(self, data_path, label_path, subject_ids):
        self.data_path = data_path
        self.label_path = label_path
        self.subject_ids = subject_ids

        self.eeg_data = []
        self.labels = []
        self.subject_indices = []

        for subject_id in subject_ids:
            data_file = f"{data_path}/S{subject_id}.npy"
            label_file = f"{label_path}/S{subject_id}.npy"

            if not os.path.exists(data_file) or not os.path.exists(label_file):
                print(f"Data not found for subject {subject_id}, skipping")
                continue

            eeg_data = np.load(data_file)  # (n_trials, 128, 32)
            labels = np.load(label_file)  # (n_trials,)


            if labels.ndim > 1:
                labels = labels.flatten()

            self.eeg_data.append(eeg_data)
            self.labels.append(labels)
            self.subject_indices.append(np.full(len(labels), subject_id))

        # Concatenate all data, maintaining order
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
# MODEL SETUP
# ============================================================================

def load_labram_finetuned():
    from modeling_finetune import labram_base_patch200_200

    print(f"Loading fine-tuned LaBraM checkpoint from: {LABRAM_CHECKPOINT}")

    # Create model
    model = labram_base_patch200_200(num_classes=2)

    # Load checkpoint - fine-tuned model saved with full state dict
    checkpoint = torch.load(LABRAM_CHECKPOINT, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        # checkpoint is the model itself
        return checkpoint

    # Remove student. prefix if exists (from SSL)
    state_dict = {
        k[8:] if k.startswith('student.') else k: v
        for k, v in state_dict.items()
    }

    # Remove classification head (we only want features)
    state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}

    # Load weights into model
    model.load_state_dict(state_dict, strict=False)
    print(f"Fine-tuned LaBraM checkpoint loaded successfully\n")

    return model

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(model, data_loader, device):

    model.eval()

    all_features = []
    all_labels = []
    all_subject_ids = []

    # Hook to capture features from LayerNorm
    captured_features = []

    def hook_fn(module, input, output):
        captured_features.append(output)

    # Register hook on the norm layer (last layer before classification head)
    hook = model.norm.register_forward_hook(hook_fn)

    pbar = tqdm(data_loader, desc="Extracting features")

    with torch.no_grad():
        for eeg, label, subject_id in pbar:
            eeg = eeg.to(device)  # (batch, 128, 32)
            label = label.to(device).squeeze()
            subject_id = subject_id.to(device).squeeze()

            # Pad EEG from 128 to 200 samples
            padding = (0, 0, 0, 200 - 128)
            eeg_padded = torch.nn.functional.pad(eeg, padding, mode='constant', value=0)

            # Prepare input: (batch, 32, 1, 200)
            x = eeg_padded.permute(0, 2, 1).unsqueeze(2)

            # Clear captured features
            captured_features.clear()

            # Forward pass through model
            _ = model(x)  # (batch, 2) - but we capture LayerNorm output via hook

            # Extract captured LayerNorm output
            features = captured_features[0]  # (batch, seq_len, 200)

            # Extract class token features (first token)
            class_features = features[:, 0, :]  # (batch, 200)

            # Store
            all_features.append(class_features.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_subject_ids.append(subject_id.cpu().numpy())

    # Remove hook
    hook.remove()

    # Concatenate all batches
    features = np.concatenate(all_features, axis=0)  # (num_samples, 200)
    labels = np.concatenate(all_labels, axis=0)  # (num_samples,)
    subject_ids = np.concatenate(all_subject_ids, axis=0)  # (num_samples,)

    return features, labels, subject_ids


def main():
    print("EXTRACT LABRAM FEATURES FOR FOLD 1 (6-FOLD CV)")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Fold 1 configuration
    # Train: Subjects 6-30 (from audio-visual)
    # Test: Subjects 1-5 (from audio-visual)
    train_subject_ids = list(range(6, 31))  # [6, 7, ..., 30]
    test_subject_ids = list(range(1, 6))    # [1, 2, 3, 4, 5]


    print("FOLD 1 CONFIGURATION")
    print(f"Train subjects (audio-only): {train_subject_ids}")
    print(f"Test subjects (audio-visual): {test_subject_ids}\n")

    # Load fine-tuned LaBraM model
    print("LOADING FINE-TUNED MODEL")
    model = load_labram_finetuned()
    model = model.to(DEVICE)

 
    print("EXTRACTING TRAINING SET FEATURES")
    print(f"\nLoading data for subjects {train_subject_ids}...")
    train_dataset = MM_AAD_Dataset(
        TRAIN_DATA_PATH, TRAIN_LABEL_PATH, train_subject_ids
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"\nExtracting features...")
    train_features, train_labels, train_subject_ids_array = extract_features(
        model, train_loader, DEVICE
    )

    print(f"\nTraining features extracted")
    print(f"  Shape: {train_features.shape}")
    print(f"  Labels shape: {train_labels.shape}")

    # Create DataFrame with proper column structure
    # Columns: subject_id | label | feature_0 | feature_1 | ... | feature_199
    feature_columns = [f"feature_{i}" for i in range(200)]
    train_df = pd.DataFrame(train_features, columns=feature_columns)
    train_df.insert(0, 'label', train_labels)
    train_df.insert(0, 'subject_id', train_subject_ids_array)

    # Save to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_csv_path = f"{OUTPUT_DIR}/{TRAIN_CSV}"
    train_df.to_csv(train_csv_path, index=False)

    print(f"\nTraining CSV saved: {train_csv_path}")
    print(f"  Shape: {train_df.shape}")
    print(f"  Columns: {list(train_df.columns[:5])} ... (200 features total)")
    print(f"  First few rows:")
    print(train_df.head(3))


    print("EXTRACTING TEST SET FEATURES")
    print(f"\nLoading data for subjects {test_subject_ids}...")
    test_dataset = MM_AAD_Dataset(
        TRAIN_DATA_PATH, TRAIN_LABEL_PATH, test_subject_ids
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"\nExtracting features...")
    test_features, test_labels, test_subject_ids_array = extract_features(
        model, test_loader, DEVICE
    )

    print(f"\nTest features extracted")
    print(f"  Shape: {test_features.shape}")
    print(f"  Labels shape: {test_labels.shape}")

    # Create DataFrame with proper column structure
    test_df = pd.DataFrame(test_features, columns=feature_columns)
    test_df.insert(0, 'label', test_labels)
    test_df.insert(0, 'subject_id', test_subject_ids_array)

    # Save to CSV
    test_csv_path = f"{OUTPUT_DIR}/{TEST_CSV}"
    test_df.to_csv(test_csv_path, index=False)

    print(f"\nTest CSV saved: {test_csv_path}")
    print(f"  Shape: {test_df.shape}")
    print(f"  Columns: {list(test_df.columns[:5])} ... (200 features total)")
    print(f"  First few rows:")
    print(test_df.head(3))


    print("EXTRACTION COMPLETE")
    print(f"\nTrain CSV: {train_csv_path}")
    print(f"  - Subjects: {list(train_subject_ids)}")
    print(f"  - Samples: {len(train_df)}")
    print(f"  - Columns: subject_id | label | 200 features")

    print(f"\nTest CSV: {test_csv_path}")
    print(f"  - Subjects: {list(test_subject_ids)}")
    print(f"  - Samples: {len(test_df)}")
    print(f"  - Columns: subject_id | label | 200 features")

    print(f"✓ Ready for concatenation with DWT features!")

if __name__ == '__main__':
    main()
