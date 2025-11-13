import os
import numpy as np
import pandas as pd
from scipy import signal
import pywt
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

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

OUTPUT_DIR = 'eeg_relevant_bands_fold2'
TRAIN_CSV = 'eeg_bands_train_fold2.csv'
TEST_CSV = 'eeg_bands_test_fold2.csv'

# Sampling rate (typical for EEG)
SAMPLING_RATE = 128  # Hz (adjust if different)

# DWT Configuration
WAVELET = 'db4'  # Daubechies-4
DECOMPOSITION_LEVEL = 4

# RELEVANT BANDS FOR MM-AAD (mapped to DWT levels)
# With fs=128 Hz and 4-level decomposition:
# cD4: 4-8 Hz (Theta)
# cD3: 8-16 Hz (Alpha + lower Beta)
# cD2: 16-32 Hz (Beta + lower Gamma)
BANDS = {
    'theta': {'dwt_coeff': 'cD4', 'freq_range': '4-8 Hz', 'importance': 'Language processing'},
    'alpha': {'dwt_coeff': 'cD3', 'freq_range': '8-16 Hz', 'importance': 'Visual attention'},
    'beta': {'dwt_coeff': 'cD2', 'freq_range': '16-32 Hz', 'importance': 'Speech perception'},
}

RANDOM_SEED = 42

# ============================================================================
# DWT DECOMPOSITION AND FEATURE EXTRACTION
# ============================================================================

def perform_dwt_decomposition(signal_ch, wavelet=WAVELET, level=DECOMPOSITION_LEVEL):

    # Perform multi-level DWT decomposition
    coeffs = pywt.wavedec(signal_ch, wavelet, level=level)

    # coeffs = [cA4, cD4, cD3, cD2, cD1]
    # cA4: Approximation (0-4 Hz)
    # cD4: Detail level 4 (4-8 Hz) - THETA
    # cD3: Detail level 3 (8-16 Hz) - ALPHA
    # cD2: Detail level 2 (16-32 Hz) - BETA
    # cD1: Detail level 1 (32-64 Hz) - GAMMA (excluded)

    dwt_bands = {
        'cA4': coeffs[0],  # Approximation
        'cD4': coeffs[1],  # Theta (4-8 Hz)
        'cD3': coeffs[2],  # Alpha (8-16 Hz)
        'cD2': coeffs[3],  # Beta (16-32 Hz)
        'cD1': coeffs[4],  # Gamma (32-64 Hz) - not used
    }

    return dwt_bands

def compute_band_signal_features(eeg_data, sampling_rate=SAMPLING_RATE):


    n_channels = eeg_data.shape[0]
    num_features_per_channel = 18  # 6 features × 3 bands
    features = np.zeros((n_channels, num_features_per_channel))

    band_names = list(BANDS.keys())  # ['theta', 'alpha', 'beta']

    for ch in range(n_channels):
        signal_ch = eeg_data[ch, :]  # (128,)

        # ===== PERFORM DWT DECOMPOSITION =====
        dwt_bands = perform_dwt_decomposition(signal_ch, wavelet=WAVELET, level=DECOMPOSITION_LEVEL)

        feature_idx = 0

        # ===== EXTRACT FEATURES FROM EACH RELEVANT BAND =====
        for band_name in band_names:
            band_info = BANDS[band_name]
            dwt_coeff_name = band_info['dwt_coeff']

            # Get the DWT coefficients for this band
            band_coeffs = dwt_bands[dwt_coeff_name]

            # ===== FEATURE 1: ABSOLUTE ENERGY =====
            # Energy = sum of squares of DWT coefficients
            absolute_energy = np.sum(band_coeffs ** 2)

            # ===== FEATURE 2: RELATIVE ENERGY =====
            # Get energies from all three bands for normalization
            all_energies = []
            for bn in band_names:
                bi = BANDS[bn]
                coeff_name = bi['dwt_coeff']
                all_energies.append(np.sum(dwt_bands[coeff_name] ** 2))

            total_energy = np.sum(all_energies)
            relative_energy = absolute_energy / (total_energy + 1e-10)

            # ===== FEATURE 3: ENERGY ENTROPY =====
            # Shannon entropy of normalized power spectrum of DWT coefficients
            power = band_coeffs ** 2
            power_norm = power / (np.sum(power) + 1e-10)
            energy_entropy = -np.sum(power_norm * np.log2(power_norm + 1e-10))

            # ===== FEATURE 4: ZERO CROSSING RATE =====
            # How often DWT coefficients cross zero
            if len(band_coeffs) > 1:
                zero_crossings = np.sum(np.abs(np.diff(np.sign(band_coeffs)))) / 2.0
                zcr = zero_crossings / len(band_coeffs)
            else:
                zcr = 0

            # ===== FEATURE 5: PEAK COUNT =====
            # Number of local maxima in DWT coefficients with noise threshold
            # Ignore tiny peaks from noise - only count significant peaks
            if len(band_coeffs) > 3:
                coeff_abs = np.abs(band_coeffs)

                # Calculate dynamic threshold based on signal characteristics
                # Peaks must be at least 20% of the maximum amplitude to be counted
                coeff_max = np.max(coeff_abs)
                height_threshold = coeff_max * 0.2  # Only count peaks > 20% of max

                # Also require minimum distance between peaks (avoid counting closely-spaced noise)
                min_distance = len(band_coeffs) // 10  # At least 10% of signal length apart

                # Find peaks with thresholds
                peak_indices = signal.find_peaks(
                    coeff_abs,
                    height=height_threshold,
                    distance=max(1, min_distance)
                )[0]
                peak_count = len(peak_indices)
            else:
                peak_count = 0

            # ===== FEATURE 6: SIGNAL POWER =====
            # RMS (root mean square) of DWT coefficients - measure of signal strength/power
            signal_power = np.sqrt(np.mean(band_coeffs ** 2))

            # Store features for this band
            features[ch, feature_idx] = absolute_energy
            features[ch, feature_idx + 1] = relative_energy
            features[ch, feature_idx + 2] = energy_entropy
            features[ch, feature_idx + 3] = zcr
            features[ch, feature_idx + 4] = peak_count
            features[ch, feature_idx + 5] = signal_power

            feature_idx += 6

    # Flatten: (n_channels, num_features) -> (n_channels * num_features,)
    return features.flatten()

# ============================================================================
# DATASET LOADING AND PROCESSING
# ============================================================================

def extract_all_features(subject_ids, data_path, label_path):
    """Extract band-specific signal processing features for all subjects."""

    all_features = []
    all_labels = []
    all_subject_ids = []

    for subject_id in subject_ids:
        data_file = f"{data_path}/S{subject_id}.npy"
        label_file = f"{label_path}/S{subject_id}.npy"

        if not os.path.exists(data_file) or not os.path.exists(label_file):
            print(f"⚠ Warning: Data not found for subject {subject_id}, skipping")
            continue

        # Load data
        eeg_data = np.load(data_file)  # (n_trials, 128, 32)
        labels = np.load(label_file)  # (n_trials,)

        # Flatten labels if needed
        if labels.ndim > 1:
            labels = labels.flatten()

        print(f"  Subject {subject_id}: {len(labels)} trials")

        # Extract features for each trial
        for trial_idx in range(len(labels)):
            trial_data = eeg_data[trial_idx]  # (128, 32)

            # Transpose to (32 channels, 128 samples)
            trial_data_transposed = trial_data.T

            # Extract features from relevant bands
            features = compute_band_signal_features(trial_data_transposed)

            all_features.append(features)
            all_labels.append(labels[trial_idx])
            all_subject_ids.append(subject_id)

    return np.array(all_features), np.array(all_labels), np.array(all_subject_ids)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n")

    print("║" + "EXTRACT EEG FEATURES FROM RELEVANT BANDS - FOLD 1".center(80) + "║")
    print("║" + "MM-AAD Cocktail Party (Left vs Right Speaker Attention)".center(80) + "║")
    print()

    try:

        for band_name, band_info in BANDS.items():
            print(f"\n{band_name.upper()} ({band_info['freq_range']}) - {band_info['dwt_coeff']}")
            print(f"  Importance: {band_info['importance']}")

        print("\n" + "="*80)
        print("EXCLUDED BANDS (NOT USED)")
        print("="*80)
        print("\ncA4 (0-4 Hz, Delta): Too slow, associated with sleep/drowsiness")
        print("cD1 (32-64 Hz, Gamma): Too noisy, contains EMG/muscle artifacts")


        #fold2
        train_subject_ids = list(range(1, 6)) + list(range(11, 31))  # [1-5, 11-30]
        test_subject_ids = list(range(6, 11))  # [6, 7, 8, 9, 10]
        print("\n" + "="*80)
        print("FOLD 1 CONFIGURATION")
        print("="*80)
        print(f"Train subjects (audio-visual): {train_subject_ids}")
        print(f"Test subjects (audio-visual): {test_subject_ids}\n")

        # ===== TRAINING SET FEATURE EXTRACTION =====
        print("EXTRACTING TRAINING SET FEATURES")
        print(f"\nLoading data for subjects {train_subject_ids}...")
        train_features, train_labels, train_subject_ids_array = extract_all_features(
            train_subject_ids, DATA_PATH, LABEL_PATH
        )

        print(f"\n✓ Training features extracted")
        print(f"  Shape: {train_features.shape}")
        print(f"  Labels: {dict(pd.Series(train_labels).value_counts().sort_index())}")

        # ===== TEST SET FEATURE EXTRACTION =====
        print("\n" + "="*80)
        print("EXTRACTING TEST SET FEATURES")
        print("="*80)
        print(f"\nLoading data for subjects {test_subject_ids}...")
        test_features, test_labels, test_subject_ids_array = extract_all_features(
            test_subject_ids, DATA_PATH, LABEL_PATH
        )

        print(f"\n✓ Test features extracted")
        print(f"  Shape: {test_features.shape}")
        print(f"  Labels: {dict(pd.Series(test_labels).value_counts().sort_index())}")

        # ===== CREATE DATAFRAMES =====
        print("\n" + "="*80)
        print("CREATING OUTPUT CSVS")
        print("="*80)

        n_features = train_features.shape[1]
        feature_columns = [f"band_feature_{i}" for i in range(n_features)]

        train_df = pd.DataFrame(train_features, columns=feature_columns)
        train_df.insert(0, 'label', train_labels)
        train_df.insert(0, 'subject_id', train_subject_ids_array)

        test_df = pd.DataFrame(test_features, columns=feature_columns)
        test_df.insert(0, 'label', test_labels)
        test_df.insert(0, 'subject_id', test_subject_ids_array)

        # ===== SAVE TO CSV =====
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        train_csv_path = os.path.join(OUTPUT_DIR, TRAIN_CSV)
        test_csv_path = os.path.join(OUTPUT_DIR, TEST_CSV)

        train_df.to_csv(train_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)

        print(f"\n✓ Saved: {train_csv_path}")
        print(f"  Shape: {train_df.shape}")
        print(f"  Columns: subject_id | label | {n_features} features")

        print(f"\n✓ Saved: {test_csv_path}")
        print(f"  Shape: {test_df.shape}")
        print(f"  Columns: subject_id | label | {n_features} features")

        # ===== SUMMARY =====
        print("\n" + "="*80)
        print("✓ EXTRACTION COMPLETE")
        print("="*80)

        print(f"\nBand-Specific Signal Processing Features (DWT-based):")
        print(f"  - Wavelet: {WAVELET}")
        print(f"  - Decomposition level: {DECOMPOSITION_LEVEL}")
        print(f"  - Sampling rate: {SAMPLING_RATE} Hz")
        print(f"  - Total features: {n_features}")


        print(f"\nDataset Summary:")
        print(f"  Train: {len(train_df)} samples from {len(train_df['subject_id'].unique())} subjects")
        print(f"  Test: {len(test_df)} samples from {len(test_df['subject_id'].unique())} subjects")


        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
