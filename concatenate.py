# Total 500 features- 200 labram features and 300 dwt features

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

LABRAM_TRAIN_PATH = 'darnet_features_fold1/darnet_features_train_fold1.csv'
LABRAM_TEST_PATH = 'darnet_features_fold1/darnet_features_test_fold1.csv'

DWT_TRAIN_PATH = 'dwt_features_csv_fold1/dwt_features_train_fold1.csv'
DWT_TEST_PATH = 'dwt_features_csv_fold1/dwt_features_test_fold1.csv'

OUTPUT_DIR = 'darnet_features_fold1'
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, 'combined_train_features_fold1.csv')
OUTPUT_TEST = os.path.join(OUTPUT_DIR, 'combined_test_features_fold1.csv')


def verify_files_exist():
    print("\n" + "="*80)
    print("VERIFYING FEATURE FILES")

    files_to_check = [
        (LABRAM_TRAIN_PATH, "LaBraM Training Features"),
        (LABRAM_TEST_PATH, "LaBraM Test Features"),
        (DWT_TRAIN_PATH, "DWT Training Features"),
        (DWT_TEST_PATH, "DWT Test Features"),
    ]

    all_exist = True
    for filepath, description in files_to_check:
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"{description:30s} {filepath:50s} ({file_size/1024/1024:.2f} MB)")
        else:
            print(f"{description:30s} {filepath:50s} NOT FOUND")
            all_exist = False

    if not all_exist:
        print("\nMissing feature files!")
        print("\nMake sure to run these scripts first:")
        print("  1. python extract_labram_features_fold1.py  # Creates labram_features_*_fold1.csv")
        print("  2. python extract_dwt_features_fold1.py     # Creates dwt_features_*_fold1.csv")
        return False

    print("\n✓ All files found")
    return True

def load_features(labram_path, dwt_path, split_type):
    """Load and concatenate LaBraM and DWT features."""
    print(f"\nLoading {split_type} features...")

    # Load LaBraM features
    labram_df = pd.read_csv(labram_path)
    print(f"  LaBraM shape: {labram_df.shape}")
    print(f"  LaBraM columns: {list(labram_df.columns[:5])} ... (200 features total)")

    # Load DWT features
    dwt_df = pd.read_csv(dwt_path)
    print(f"  DWT shape: {dwt_df.shape}")
    print(f"  DWT columns: {list(dwt_df.columns[:5])} ... (300 features total)")

    # Verify same number of samples
    if len(labram_df) != len(dwt_df):
        print(f"\nDifferent number of samples!")
        print(f"    LaBraM: {len(labram_df)}, DWT: {len(dwt_df)}")
        min_len = min(len(labram_df), len(dwt_df))
        print(f"    Using min samples: {min_len}")
        labram_df = labram_df.iloc[:min_len]
        dwt_df = dwt_df.iloc[:min_len]

    # Verify matching subject and label
    if not np.array_equal(labram_df['subject_id'].values, dwt_df['subject_id'].values):
        print(f"\nSubject IDs don't match between LaBraM and DWT!")
        print(f"    This could indicate ordering issues")

    if not np.array_equal(labram_df['label'].values, dwt_df['label'].values):
        print(f"\nLabels don't match between LaBraM and DWT!")
        print(f"    This could indicate ordering issues")

    # Extract feature column names (skip subject_id and label)
    labram_cols = [col for col in labram_df.columns if col not in ['subject_id', 'label']]
    dwt_cols = [col for col in dwt_df.columns if col not in ['subject_id', 'label']]

    print(f"\n  LaBraM features: {len(labram_cols)}")
    print(f"  DWT features: {len(dwt_cols)}")

    # Concatenate horizontally (same rows, add columns)
    print(f"  Concatenating features...")
    combined_df = labram_df[['subject_id', 'label']].copy()

    # Add LaBraM features
    for col in labram_cols:
        combined_df[col] = labram_df[col].values

    # Add DWT features
    for col in dwt_cols:
        combined_df[col] = dwt_df[col].values

    print(f"  Combined shape: {combined_df.shape}")
    print(f"  Combined columns: {list(combined_df.columns[:5])} ... (500 features total)")

    return combined_df, labram_cols + dwt_cols

def normalize_features(train_df, test_df, feature_cols):

    print("NORMALIZING FEATURES")

    print("\nFitting StandardScaler on training data...")
    scaler = StandardScaler()

    train_features = train_df[feature_cols].values
    train_features_normalized = scaler.fit_transform(train_features)

    print(f"  Training features statistics:")
    print(f"    Mean: {train_features_normalized.mean(axis=0).mean():.6f}")
    print(f"    Std: {train_features_normalized.std(axis=0).mean():.6f}")

    test_features = test_df[feature_cols].values
    test_features_normalized = scaler.transform(test_features)

    print(f"    Mean: {test_features_normalized.mean(axis=0).mean():.6f}")
    print(f"    Std: {test_features_normalized.std(axis=0).mean():.6f}")

    # Create new dataframes with normalized features
    train_normalized = train_df[['subject_id', 'label']].copy()
    test_normalized = test_df[['subject_id', 'label']].copy()

    for i, col in enumerate(feature_cols):
        train_normalized[col] = train_features_normalized[:, i]
        test_normalized[col] = test_features_normalized[:, i]

    return train_normalized, test_normalized

def main():


    try:
        # Verify files exist
        if not verify_files_exist():
            return 1

        # Load features
        print("\n" + "="*80)
        print("LOADING FEATURES")
        print("="*80)

        train_df, train_cols = load_features(LABRAM_TRAIN_PATH, DWT_TRAIN_PATH, "Training")
        test_df, test_cols = load_features(LABRAM_TEST_PATH, DWT_TEST_PATH, "Test")

        # Verify feature columns match
        if train_cols != test_cols:
            print("\nERROR: Feature columns don't match between train and test!")
            print(f"  Train has {len(train_cols)} features")
            print(f"  Test has {len(test_cols)} features")
            return 1

        feature_cols = train_cols


        print("FEATURE STATISTICS BEFORE NORMALIZATION")

        print(f"\nTotal features: {len(feature_cols)}")
        print(f"  - LaBraM: 200 features")
        print(f"  - DWT: 300 features")

        print(f"\nTrain set: {len(train_df)} samples")
        print(f"  Subjects: {sorted(train_df['subject_id'].unique())}")
        print(f"  Labels: {sorted(train_df['label'].unique())}")
        print(f"  Label distribution:\n{train_df['label'].value_counts().sort_index()}")
        print(f"\n  Feature statistics:")
        print(f"    Min: {train_df[feature_cols].min().min():.6f}")
        print(f"    Max: {train_df[feature_cols].max().max():.6f}")
        print(f"    Mean: {train_df[feature_cols].mean().mean():.6f}")
        print(f"    Std: {train_df[feature_cols].std().mean():.6f}")

        print(f"\nTest set: {len(test_df)} samples")
        print(f"  Subjects: {sorted(test_df['subject_id'].unique())}")
        print(f"  Labels: {sorted(test_df['label'].unique())}")
        print(f"  Label distribution:\n{test_df['label'].value_counts().sort_index()}")
        print(f"\n  Feature statistics:")
        print(f"    Min: {test_df[feature_cols].min().min():.6f}")
        print(f"    Max: {test_df[feature_cols].max().max():.6f}")
        print(f"    Mean: {test_df[feature_cols].mean().mean():.6f}")
        print(f"    Std: {test_df[feature_cols].std().mean():.6f}")

        # Normalize
        train_normalized, test_normalized = normalize_features(train_df, test_df, feature_cols)

        # Save
 
        print("SAVING CONCATENATED FEATURES")
  

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        train_normalized.to_csv(OUTPUT_TRAIN, index=False)
        test_normalized.to_csv(OUTPUT_TEST, index=False)

        print(f"\nSaved: {OUTPUT_TRAIN}")
        print(f"  Shape: {train_normalized.shape}")

        print(f"\nSaved: {OUTPUT_TEST}")
        print(f"  Shape: {test_normalized.shape}")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())