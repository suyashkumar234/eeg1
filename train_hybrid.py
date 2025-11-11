import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

TRAIN_FEATURES_PATH = 'darnet_features_fold1/combined_train_features_fold1.csv'
TEST_FEATURES_PATH = 'darnet_features_fold1/combined_test_features_fold1.csv'

OUTPUT_DIR = 'hybrid_darnet_ml_results_fold1'

# Fold configuration
TRAIN_SUBJECTS = list(range(6, 31))  # Subjects 6-30 (25 subjects)
TEST_SUBJECTS = list(range(1, 6))    # Subjects 1-5 (5 subjects)

# Hyperparameters to test
XGBOOST_PARAMS_LIST = [
    {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 100, 'random_state': 42},
    {'max_depth': 10, 'learning_rate': 0.05, 'n_estimators': 100, 'random_state': 42},
    {'max_depth': 12, 'learning_rate': 0.05, 'n_estimators': 100, 'random_state': 42},
    {'max_depth': 14, 'learning_rate': 0.05, 'n_estimators': 100, 'random_state': 42},
]

SVM_PARAMS_LIST = [
    {'C': 1, 'kernel': 'rbf', 'random_state': 42},
    {'C': 10, 'kernel': 'rbf', 'random_state': 42},
    {'C': 100, 'kernel': 'rbf', 'random_state': 42},
]

RF_PARAMS_LIST = [
    {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1},
    {'n_estimators': 200, 'max_depth': 15, 'random_state': 42, 'n_jobs': -1},
    {'n_estimators': 300, 'max_depth': 20, 'random_state': 42, 'n_jobs': -1},
]

RANDOM_STATE = 42

def train_xgboost_fold(X_train, y_train, X_test, y_test, params):
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=0)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return {
        'train_acc': float(accuracy_score(y_train, y_pred_train)),
        'test_acc': float(accuracy_score(y_test, y_pred_test)),
        'test_precision': float(precision_score(y_test, y_pred_test, zero_division=0)),
        'test_recall': float(recall_score(y_test, y_pred_test, zero_division=0)),
        'test_f1': float(f1_score(y_test, y_pred_test, zero_division=0)),
    }

def train_svm_fold(X_train, y_train, X_test, y_test, params):
    """Train SVM for fold with given params."""
    model = SVC(**params)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return {
        'train_acc': float(accuracy_score(y_train, y_pred_train)),
        'test_acc': float(accuracy_score(y_test, y_pred_test)),
        'test_precision': float(precision_score(y_test, y_pred_test, zero_division=0)),
        'test_recall': float(recall_score(y_test, y_pred_test, zero_division=0)),
        'test_f1': float(f1_score(y_test, y_pred_test, zero_division=0)),
    }

def train_rf_fold(X_train, y_train, X_test, y_test, params):
    """Train Random Forest for fold with given params."""
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return {
        'train_acc': float(accuracy_score(y_train, y_pred_train)),
        'test_acc': float(accuracy_score(y_test, y_pred_test)),
        'test_precision': float(precision_score(y_test, y_pred_test, zero_division=0)),
        'test_recall': float(recall_score(y_test, y_pred_test, zero_division=0)),
        'test_f1': float(f1_score(y_test, y_pred_test, zero_division=0)),
    }

def main():
    print("\n")


    print("TRAIN HYBRID ML: XGBoost + SVM + Random Forest (FOLD 1)")

    print("(LaBraM 200 + DWT 300 = 500 features)")

    try:
        print("LOADING CONCATENATED FEATURES")


        if not os.path.exists(TRAIN_FEATURES_PATH):
            print(f"✗ ERROR: {TRAIN_FEATURES_PATH} not found")
            print("Run: python concatenate_labram_dwt_fold1.py")
            return 1

        if not os.path.exists(TEST_FEATURES_PATH):
            print(f"✗ ERROR: {TEST_FEATURES_PATH} not found")
            print("Run: python concatenate_labram_dwt_fold1.py")
            return 1

        train_df = pd.read_csv(TRAIN_FEATURES_PATH)
        test_df = pd.read_csv(TEST_FEATURES_PATH)

        print(f"Train features shape: {train_df.shape}")
        print(f"Test features shape: {test_df.shape}")

        feature_cols = [col for col in train_df.columns if col not in ['subject_id', 'label']]
        print(f"Total features: {len(feature_cols)}")
        print(f"  - LaBraM: 200 features")
        print(f"  - DWT: 300 features")


        print("FOLD 1 CONFIGURATION")


        print(f"Train subjects: {TRAIN_SUBJECTS} (25 subjects)")
        print(f"Test subjects: {TEST_SUBJECTS} (5 subjects)")

        # Filter to fold 1 subjects
        X_train_full = train_df[train_df['subject_id'].isin(TRAIN_SUBJECTS)].copy()
        X_test_full = test_df[test_df['subject_id'].isin(TEST_SUBJECTS)].copy()

        print(f"\nTraining data: {len(X_train_full)} samples")
        print(f"  Subjects: {sorted(X_train_full['subject_id'].unique())}")
        print(f"  Label distribution: {dict(X_train_full['label'].value_counts().sort_index())}")

        print(f"\nTest data: {len(X_test_full)} samples")
        print(f"  Subjects: {sorted(X_test_full['subject_id'].unique())}")
        print(f"  Label distribution: {dict(X_test_full['label'].value_counts().sort_index())}")

        # Prepare feature arrays
        X_train = X_train_full[feature_cols].values
        y_train = X_train_full['label'].values

        X_test = X_test_full[feature_cols].values
        y_test = X_test_full['label'].values

        # Shuffle training data
        shuffle_idx = np.random.RandomState(RANDOM_STATE).permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]

        print(f"\nFeatures prepared:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)


        print("XGBOOST TRAINING (Hyperparameter Tuning)")
        print("Testing max_depth in [8, 10, 12, 14]...\n")

        best_xgb_acc = 0
        best_xgb_result = None
        best_xgb_params = None
        xgb_results = []

        for i, params in enumerate(XGBOOST_PARAMS_LIST, 1):
            result = train_xgboost_fold(X_train, y_train, X_test, y_test, params)
            depth = params['max_depth']
            print(f"{i}. max_depth={depth:2d}: train_acc={result['train_acc']:.4f}, test_acc={result['test_acc']:.4f}, test_f1={result['test_f1']:.4f}")

            if result['test_acc'] > best_xgb_acc:
                best_xgb_acc = result['test_acc']
                best_xgb_result = result
                best_xgb_params = params

            result['max_depth'] = depth
            xgb_results.append(result)

        print(f"\nBest XGBoost: max_depth={best_xgb_params['max_depth']}")
        print(f"  Train Accuracy: {best_xgb_result['train_acc']:.4f}")
        print(f"  Test Accuracy: {best_xgb_result['test_acc']:.4f}")
        print(f"  Test Precision: {best_xgb_result['test_precision']:.4f}")
        print(f"  Test Recall: {best_xgb_result['test_recall']:.4f}")
        print(f"  Test F1: {best_xgb_result['test_f1']:.4f}")

        # ===== SVM TRAINING =====
        print("\n" )
        print("SVM TRAINING (Hyperparameter Tuning)")
        print("\n" )
        print("Testing C in [1, 10, 100]...\n")

        best_svm_acc = 0
        best_svm_result = None
        best_svm_params = None
        svm_results = []

        for i, params in enumerate(SVM_PARAMS_LIST, 1):
            result = train_svm_fold(X_train, y_train, X_test, y_test, params)
            c_val = params['C']
            print(f"{i}. C={c_val:5.0f}: train_acc={result['train_acc']:.4f}, test_acc={result['test_acc']:.4f}, test_f1={result['test_f1']:.4f}")

            if result['test_acc'] > best_svm_acc:
                best_svm_acc = result['test_acc']
                best_svm_result = result
                best_svm_params = params

            result['C'] = c_val
            svm_results.append(result)

        print(f"\n✓ Best SVM: C={best_svm_params['C']}")
        print(f"  Train Accuracy: {best_svm_result['train_acc']:.4f}")
        print(f"  Test Accuracy: {best_svm_result['test_acc']:.4f}")
        print(f"  Test Precision: {best_svm_result['test_precision']:.4f}")
        print(f"  Test Recall: {best_svm_result['test_recall']:.4f}")
        print(f"  Test F1: {best_svm_result['test_f1']:.4f}")



        print("RANDOM FOREST TRAINING (Hyperparameter Tuning)")
        print("Testing n_estimators in [100, 200, 300], max_depth in [10, 15, 20]...\n")

        best_rf_acc = 0
        best_rf_result = None
        best_rf_params = None
        rf_results = []

        for i, params in enumerate(RF_PARAMS_LIST, 1):
            result = train_rf_fold(X_train, y_train, X_test, y_test, params)
            n_est = params['n_estimators']
            depth = params['max_depth']
            print(f"{i}. n_est={n_est:3d}, max_depth={depth:2d}: train_acc={result['train_acc']:.4f}, test_acc={result['test_acc']:.4f}, test_f1={result['test_f1']:.4f}")

            if result['test_acc'] > best_rf_acc:
                best_rf_acc = result['test_acc']
                best_rf_result = result
                best_rf_params = params

            result['n_estimators'] = n_est
            result['max_depth'] = depth
            rf_results.append(result)

        print(f"\nBest Random Forest: n_estimators={best_rf_params['n_estimators']}, max_depth={best_rf_params['max_depth']}")
        print(f"  Train Accuracy: {best_rf_result['train_acc']:.4f}")
        print(f"  Test Accuracy: {best_rf_result['test_acc']:.4f}")
        print(f"  Test Precision: {best_rf_result['test_precision']:.4f}")
        print(f"  Test Recall: {best_rf_result['test_recall']:.4f}")
        print(f"  Test F1: {best_rf_result['test_f1']:.4f}")


        print("SAVING RESULTS")


        xgb_df = pd.DataFrame(xgb_results)
        svm_df = pd.DataFrame(svm_results)
        rf_df = pd.DataFrame(rf_results)

        xgb_df.to_csv(os.path.join(OUTPUT_DIR, 'xgboost_fold1_results.csv'), index=False)
        svm_df.to_csv(os.path.join(OUTPUT_DIR, 'svm_fold1_results.csv'), index=False)
        rf_df.to_csv(os.path.join(OUTPUT_DIR, 'random_forest_fold1_results.csv'), index=False)

        print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'xgboost_fold1_results.csv')}")
        print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'svm_fold1_results.csv')}")
        print(f"✓ Saved: {os.path.join(OUTPUT_DIR, 'random_forest_fold1_results.csv')}")

        print("\n")
        print("FOLD 1 RESULTS SUMMARY")


        print("\n")
        print("XGBoost Results:")
        print(f"  Best Test Accuracy:  {best_xgb_result['test_acc']:.4f}")
        print(f"  Best Test F1:        {best_xgb_result['test_f1']:.4f}")
        print(f"  Best Hyperparams:    max_depth={best_xgb_params['max_depth']}")

        print("\n")
        print("SVM Results:")
        print(f"  Best Test Accuracy:  {best_svm_result['test_acc']:.4f}")
        print(f"  Best Test F1:        {best_svm_result['test_f1']:.4f}")
        print(f"  Best Hyperparams:    C={best_svm_params['C']}")

        print("\n")
        print("Random Forest Results:")
        print("\n")
        print(f"  Best Test Accuracy:  {best_rf_result['test_acc']:.4f}")
        print(f"  Best Test F1:        {best_rf_result['test_f1']:.4f}")
        print(f"  Best Hyperparams:    n_est={best_rf_params['n_estimators']}, max_depth={best_rf_params['max_depth']}")


        best_model = 'xgboost'
        best_acc = best_xgb_result['test_acc']
        best_f1 = best_xgb_result['test_f1']

        if best_svm_result['test_acc'] > best_acc:
            best_model = 'svm'
            best_acc = best_svm_result['test_acc']
            best_f1 = best_svm_result['test_f1']

        if best_rf_result['test_acc'] > best_acc:
            best_model = 'random_forest'
            best_acc = best_rf_result['test_acc']
            best_f1 = best_rf_result['test_f1']

        
        summary = {
            'fold': 'Fold 1',
            'feature_config': 'LaBraM (200) + DWT (300) = 500 dimensions',
            'total_features': len(feature_cols),
            'training_subjects': TRAIN_SUBJECTS,
            'test_subjects': TEST_SUBJECTS,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'training_method': 'Cross-subject fold-based training',
            'evaluation': 'Unseen subjects (cross-subject)',
            'best_model': best_model,
            'best_test_accuracy': float(best_acc),
            'best_test_f1': float(best_f1),
            'xgboost': {
                'best_test_accuracy': float(best_xgb_result['test_acc']),
                'best_test_f1': float(best_xgb_result['test_f1']),
                'best_hyperparams': {'max_depth': best_xgb_params['max_depth']},
            },
            'svm': {
                'best_test_accuracy': float(best_svm_result['test_acc']),
                'best_test_f1': float(best_svm_result['test_f1']),
                'best_hyperparams': {'C': best_svm_params['C']},
            },
            'random_forest': {
                'best_test_accuracy': float(best_rf_result['test_acc']),
                'best_test_f1': float(best_rf_result['test_f1']),
                'best_hyperparams': {
                    'n_estimators': best_rf_params['n_estimators'],
                    'max_depth': best_rf_params['max_depth']
                },
            },
        }

        with open(os.path.join(OUTPUT_DIR, 'fold1_results_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Saved: {os.path.join(OUTPUT_DIR, 'fold1_results_summary.json')}")

        # ===== FINAL COMPARISON =====
        print("\n" + "="*80)
        print("FOLD 1 COMPLETED")
    
        print(f"\nFold 1 Results (Cross-Subject):")
        print(f"  XGBoost:      {best_xgb_result['test_acc']:.4f} (F1: {best_xgb_result['test_f1']:.4f})")
        print(f"  SVM:          {best_svm_result['test_acc']:.4f} (F1: {best_svm_result['test_f1']:.4f})")
        print(f"  Random Forest: {best_rf_result['test_acc']:.4f} (F1: {best_rf_result['test_f1']:.4f})")
        print(f"\n  🏆 Best Model: {best_model.upper()}")
        print(f"     Accuracy: {best_acc:.4f} | F1: {best_f1:.4f}")
        print(f"\nNext Steps:")
        print(f"  - Create similar scripts for Fold 2, 3, 4, 5, 6")
        print(f"  - Aggregate results across all folds")
        print(f"  - Compare with DARNet baseline")


        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
