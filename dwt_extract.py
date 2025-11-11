from __future__ import print_function
import numpy as np
import pandas as pd
import pywt  # PyWavelets library
import os
from tqdm import tqdm
import warnings
from datetime import datetime
import time

warnings.filterwarnings('ignore')


# Both train and test use audio-visual data for Fold 1
DATA_PATH = "/scratch/suyash.kumar.mec22.itbhu/EEG-AAD_audio_visual/preprocessed/data/"
LABEL_PATH = "/scratch/suyash.kumar.mec22.itbhu/EEG-AAD_audio_visual/preprocessed/label/"

# Fold 1 configuration
TRAIN_SUBJECT_IDS = list(range(6, 31))    # Subjects 6-30 (25 subjects)
TEST_SUBJECT_IDS = list(range(1, 6))      # Subjects 1-5 (5 subjects)

# Output directory and filenames
OUTPUT_DIR = "dwt_features_csv_fold1"
TRAIN_CSV = "dwt_features_train_fold1.csv"
TEST_CSV = "dwt_features_test_fold1.csv"

# DWT parameters
WAVELET = 'db4'        # Daubechies-4 wavelet (good for EEG)
DECOMP_LEVEL = 4       # Number of decomposition levels
FS = 128               # Sampling frequency (Hz)

# Frequency bands for reference
FREQ_BANDS = {
    'delta': (0.5, 4),      # δ: Sleep, deep thinking
    'theta': (4, 8),        # θ: Drowsiness, meditation
    'alpha': (8, 12),       # α: Relaxation, eyes closed
    'beta': (12, 30),       # β: Active thinking, attention
    'gamma': (30, 100)      # γ: High cognition
}


STATS = ['energy', 'mean', 'std']
N_CHANNELS = 32


def extract_dwt_features(eeg_trial, wavelet='db4', decomp_level=4):
    """
    Extract time-frequency features using Discrete Wavelet Transform (DWT).

    This performs multi-level wavelet decomposition and extracts statistics
    from each coefficient level.

    Parameters:
    -----------
    eeg_trial : ndarray
        Shape (n_channels, n_timepoints) = (32, 128)
        EEG signal for one trial
    wavelet : str
        Wavelet type ('db4' = Daubechies-4 is recommended for EEG)
    decomp_level : int
        Number of decomposition levels (typically 3-4)

    Returns:
    --------
    features : ndarray
        Shape (480,) = 32 channels × 5 coeff_types × 3 statistics
        Concatenated features capturing time-frequency information
    """

    n_channels, n_timepoints = eeg_trial.shape
    features = []

    # Extract features for each channel
    for ch in range(n_channels):
        eeg_signal = eeg_trial[ch, :]

        # Perform multi-level DWT decomposition
        # Returns: list of [cA_n, cD_n, cD_n-1, ..., cD_1]
        # where cA = approximation, cD = detail at each level
        coefficients = pywt.wavedec(eeg_signal, wavelet, level=decomp_level)

        # Extract features from each coefficient set
        # coefficients[0] = approximation at decomp_level
        # coefficients[1:] = details from decomp_level down to 1
        for coeff_idx, coeff_set in enumerate(coefficients):
            abs_coeffs = np.abs(coeff_set)

            # Extract only most important statistics from coefficients
            energy = np.mean(abs_coeffs ** 2)           # Energy (power) - MOST IMPORTANT
            mean_val = np.mean(abs_coeffs)              # Mean amplitude - IMPORTANT
            std_val = np.std(abs_coeffs)                # Variability - IMPORTANT

            features.extend([energy, mean_val, std_val])

    return np.array(features, dtype=np.float32)

# ============================================================================
# CREATE COLUMN NAMES
# ============================================================================

def create_column_names_dwt(wavelet='db4', decomp_level=4):
    """
    Create column names for DWT features.

    Returns:
    --------
    columns : list
        List of column names
    """
    columns = []

    for ch in range(N_CHANNELS):
        # Approximation + detail coefficients (decomp_level + 1 total)
        n_coeff_sets = decomp_level + 1

        for coeff_idx in range(n_coeff_sets):
            # Name the coefficient set
            if coeff_idx == 0:
                coeff_name = f'cA{decomp_level}'  # Approximation at final level
            else:
                coeff_name = f'cD{decomp_level - coeff_idx + 1}'  # Detail levels

            # Add each statistic
            for stat in STATS:
                columns.append(f'dwt_{stat}_{coeff_name}_ch{ch}')

    return columns

# ============================================================================
# SUBJECT PROCESSING FUNCTION
# ============================================================================

def process_subject_wavelets(subject_id, wavelet, decomp_level):
    """
    Process a single subject's data and extract DWT features.

    Parameters:
    -----------
    subject_id : int
        Subject ID (1-30)
    wavelet : str
        Wavelet type
    decomp_level : int
        Decomposition level

    Returns:
    --------
    features : list
        List of feature arrays for each trial
    labels : list
        List of labels for each trial
    subject_ids : list
        List of subject IDs for each trial
    """
    features = []
    labels = []
    subject_ids = []

    # Load data
    data_file = f"{DATA_PATH}/S{subject_id}.npy"
    label_file = f"{LABEL_PATH}/S{subject_id}.npy"

    if os.path.exists(data_file) and os.path.exists(label_file):
        try:
            eeg = np.load(data_file)      # (n_trials, 128, 32)
            eeg_labels = np.load(label_file)  # (n_trials,)

            if eeg_labels.ndim > 1:
                eeg_labels = eeg_labels.flatten()

            # Extract DWT features for each trial
            for trial_idx in range(eeg.shape[0]):
                trial = eeg[trial_idx]           # (128, 32) = (time, channels)
                trial_transposed = trial.T       # (32, 128) = (channels, time)

                # Extract DWT features
                feat = extract_dwt_features(
                    trial_transposed,
                    wavelet=wavelet,
                    decomp_level=decomp_level
                )

                features.append(feat)
                labels.append(eeg_labels[trial_idx])
                subject_ids.append(subject_id)
        except Exception as e:
            print(f"  Error processing subject {subject_id}: {e}")
            return [], [], []
    else:
        print(f"  Warning: Data not found for subject {subject_id}")
        return [], [], []

    return features, labels, subject_ids

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    start_time = time.time()

    print("=" * 80)
    print("DWT WAVELET FEATURE EXTRACTION FOR FOLD 1 (6-FOLD CV)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    print("=" * 80)
    print("FOLD 1 CONFIGURATION")
    print("=" * 80)
    print(f"Train subjects (audio-visual): {TRAIN_SUBJECT_IDS}")
    print(f"Test subjects (audio-visual): {TEST_SUBJECT_IDS}\n")

    print("=" * 80)
    print("DWT CONFIGURATION")
    print("=" * 80)
    print(f"Wavelet: {WAVELET} (Daubechies-4)")
    print(f"Decomposition levels: {DECOMP_LEVEL}")
    print(f"Decomposition produces: {DECOMP_LEVEL + 1} coefficient sets")
    print("  (1 approximation + 4 detail coefficients)")
    print("\nFeatures per coefficient set (OPTIMIZED):")
    print("  - Energy (mean squared coefficients) - MOST IMPORTANT")
    print("  - Mean (average amplitude) - IMPORTANT")
    print("  - Std (variability) - IMPORTANT")
    print("  Note: Removed max, min to reduce noise and improve generalization")
    print(f"\nTotal: {len(STATS)} stats × {DECOMP_LEVEL + 1} coefficients × {N_CHANNELS} channels")
    print(f"     = {len(STATS) * (DECOMP_LEVEL + 1) * N_CHANNELS} features per trial\n")

    # Create column names
    print("Creating column names...")
    columns = create_column_names_dwt(WAVELET, DECOMP_LEVEL)
    print(f"Total features per trial: {len(columns)}\n")

    # ========================================================================
    # TRAINING SET FEATURE EXTRACTION
    # ========================================================================
    print("=" * 80)
    print("EXTRACTING TRAINING SET DWT FEATURES")
    print("=" * 80)

    all_train_features = []
    all_train_labels = []
    all_train_subjects = []

    pbar = tqdm(TRAIN_SUBJECT_IDS, desc="Training subjects")
    for subject_id in pbar:
        features, labels, subject_ids = process_subject_wavelets(
            subject_id, WAVELET, DECOMP_LEVEL
        )

        if features:
            all_train_features.extend(features)
            all_train_labels.extend(labels)
            all_train_subjects.extend(subject_ids)
            pbar.set_postfix({'trials': len(features)})

    train_features_array = np.array(all_train_features, dtype=np.float32)

    print(f"\n✓ Training features extracted")
    print(f"  Shape: {train_features_array.shape}")
    print(f"  Total trials: {len(all_train_labels)}")

    # Create training DataFrame
    print(f"\nCreating training DataFrame...")
    train_df = pd.DataFrame(train_features_array, columns=columns)
    train_df.insert(0, 'label', all_train_labels)
    train_df.insert(0, 'subject_id', all_train_subjects)

    # Save to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_csv_path = f"{OUTPUT_DIR}/{TRAIN_CSV}"
    train_df.to_csv(train_csv_path, index=False)

    print(f"✓ Training CSV saved: {train_csv_path}")
    print(f"  Shape: {train_df.shape}")
    print(f"  Columns: {list(train_df.columns[:5])} ... (300 features total)")
    print(f"  First few rows:")
    print(train_df.head(3))

    # ========================================================================
    # TEST SET FEATURE EXTRACTION
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("EXTRACTING TEST SET DWT FEATURES")
    print("=" * 80)

    all_test_features = []
    all_test_labels = []
    all_test_subjects = []

    pbar = tqdm(TEST_SUBJECT_IDS, desc="Test subjects")
    for subject_id in pbar:
        features, labels, subject_ids = process_subject_wavelets(
            subject_id, WAVELET, DECOMP_LEVEL
        )

        if features:
            all_test_features.extend(features)
            all_test_labels.extend(labels)
            all_test_subjects.extend(subject_ids)
            pbar.set_postfix({'trials': len(features)})

    test_features_array = np.array(all_test_features, dtype=np.float32)

    print(f"\n✓ Test features extracted")
    print(f"  Shape: {test_features_array.shape}")
    print(f"  Total trials: {len(all_test_labels)}")

    # Create test DataFrame
    print(f"\nCreating test DataFrame...")
    test_df = pd.DataFrame(test_features_array, columns=columns)
    test_df.insert(0, 'label', all_test_labels)
    test_df.insert(0, 'subject_id', all_test_subjects)

    # Save to CSV
    test_csv_path = f"{OUTPUT_DIR}/{TEST_CSV}"
    test_df.to_csv(test_csv_path, index=False)

    print(f"✓ Test CSV saved: {test_csv_path}")
    print(f"  Shape: {test_df.shape}")
    print(f"  Columns: {list(test_df.columns[:5])} ... (300 features total)")
    print(f"  First few rows:")
    print(test_df.head(3))

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nTrain CSV: {train_csv_path}")
    print(f"  - Subjects: {sorted(train_df['subject_id'].unique())}")
    print(f"  - Samples: {len(train_df)}")
    print(f"  - Columns: subject_id | label | 300 features")
    print(f"  - Class distribution:\n{train_df['label'].value_counts().sort_index()}")

    print(f"\nTest CSV: {test_csv_path}")
    print(f"  - Subjects: {sorted(test_df['subject_id'].unique())}")
    print(f"  - Samples: {len(test_df)}")
    print(f"  - Columns: subject_id | label | 300 features")
    print(f"  - Class distribution:\n{test_df['label'].value_counts().sort_index()}")

    print(f"\n✓ Ready for concatenation with LaBraM features!")
    print("=" * 80)

    # Print time taken
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"\n⏱️  Total time elapsed: {elapsed_minutes:.1f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()