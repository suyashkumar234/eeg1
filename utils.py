import os
import numpy as np
from torch.utils.data import Dataset
import torch
import scipy.io as scio
import math
from scipy.io import loadmat
from scipy.linalg import sqrtm
from mne.decoding import CSP

def makePath(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


class CustomDatasets(Dataset):
    def __init__(self, data, event_data, subject_ids=None):
        self.data = data
        self.label = event_data
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = torch.Tensor(self.data[index])
        label = torch.LongTensor(self.label[index])

        if self.subject_ids is not None:
            subject_id = self.subject_ids[index]  # Return scalar, not tensor
            return data, label, subject_id

        return data, label

class EEGDataLoader(Dataset):
    def __init__(self, x, y):
        self.data = torch.from_numpy(x)
        self.labels = torch.from_numpy(y)  # label without one-hot coding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor = self.data[idx]
        label_tensor = self.labels[idx]
        return data_tensor, label_tensor


def getData(args,  sub_ids, apply_csp=False, apply_ea=False, apply_normalize=False):
    """
    Load MM-AAD dataset with optional preprocessing

    Args:
        args: arguments
        sub_ids: subject IDs to load
        apply_csp: Apply CSP transformation (default: False)
        apply_ea: Apply Euclidean Alignment (default: False)
        apply_normalize: Apply normalization (default: False)

    Returns:
        alldata: list of data arrays per subject
        alllabel: list of label arrays per subject
    """
    alldata = []
    alllabel = []

    for id in sub_ids:
        onedata = np.load(args.data_path +  f'S{id}.npy')
        onelabel = np.load(args.label_path +  f'S{id}.npy')
        onedata = onedata.transpose(0,2,1)  # (N, channels, samples)

        # Apply preprocessing if requested
        if apply_normalize:
            onedata = data_norm_aved(onedata)

        if apply_ea:
            onedata = euclidean_alignment_aved(onedata)

        if apply_csp:
            onedata = apply_csp_transform(onedata, onelabel, n_components=args.eeg_channel)

        alldata.append(onedata)
        alllabel.append(onelabel)

    return alldata,  alllabel


def getData_DTU(args, sub_ids, apply_csp=False, apply_ea=False, apply_normalize=False):
    """
    Load DTU dataset from .mat files for spatial attention detection

    DTU Dataset Specs:
    - 18 subjects, 64 EEG channels, 60 trials per subject
    - Sampling rate: 512 Hz (downsampled from original 512 Hz recording)
    - Each trial: 3200 samples × 66 channels (we use first 64 EEG channels)
    - Trial duration: 3200 / 512 = 6.25 seconds per trial
    - Labels: attend_lr (spatial attention)
      - 1 = Left speaker attended
      - 2 = Right speaker attended
      - Converted to 0/1 for binary classification

    Args:
        args: Arguments containing data_path, window_length, overlap
        sub_ids: List of subject IDs to load (1-18)
        apply_csp: Apply CSP transformation (default: False)
        apply_ea: Apply Euclidean Alignment (default: False)
        apply_normalize: Apply normalization (default: False)

    Returns:
        alldata: List of windowed EEG data per subject (num_windows, 64, window_size)
        alllabel: List of labels per subject (num_windows, 1)
    """
    alldata = []
    alllabel = []

    for id in sub_ids:
        mat_path = args.data_path + f'S{id}_data_preproc.mat'

        # Load .mat file
        mat_eeg_data = []
        mat_event_data = []
        matstruct_contents = loadmat(mat_path)
        matstruct_contents = matstruct_contents['data']

        # Extract event and EEG data structure
        # From inspection: data[0,0]['event'] has shape (1, 1) with field 'eeg'
        # data[0,0]['event'] -> (1, 1) structured array
        # data[0,0]['event'][0, 0] -> struct with 'eeg' field
        # data[0,0]['event'][0, 0]['eeg'] -> (1, 60) array of trial events
        mat_event_eeg = matstruct_contents[0, 0]['event'][0, 0]['eeg']  # (1, 60)
        mat_eeg = matstruct_contents[0, 0]['eeg']  # (1, 60) trials, each trial is (3200, 66)

        # Extract EEG data and labels for each trial
        for trial_idx in range(mat_eeg.shape[1]):
            # Get EEG data for this trial
            trial_eeg = mat_eeg[0, trial_idx]  # (3200, 66)
            mat_eeg_data.append(trial_eeg)

            # Get label (attend_lr: 1=left, 2=right)
            # mat_event_eeg is (1, 60), each element has 'value' field
            trial_event = mat_event_eeg[0, trial_idx]
            label_value = trial_event['value']  # This is [[2]] or [[1]]

            # Extract scalar value from nested array
            if isinstance(label_value, np.ndarray):
                label_value = int(label_value.item())  # Use .item() to avoid deprecation warning

            mat_event_data.append(label_value)

        # Process windowing
        eeg_data = np.array(mat_eeg_data)  # (60, 3200, 66)
        eeg_data = eeg_data[:, :, 0:64]  # Take first 64 EEG channels → (60, 3200, 64)
        event_data = np.array(mat_event_data)  # (60,)

        # Apply sliding window segmentation
        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))

        windowed_eeg = []
        windowed_labels = []

        for trial_idx in range(len(event_data)):
            eeg = eeg_data[trial_idx]  # (3200, 64)
            label = event_data[trial_idx] - 1  # Convert 1/2 to 0/1 (left=0, right=1)

            # Sliding window
            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i + window_size, :]  # (window_size, 64)
                windowed_eeg.append(window)
                windowed_labels.append(label)

        # Convert to numpy arrays and transpose to (windows, channels, time)
        windowed_eeg = np.array(windowed_eeg)  # (num_windows, window_size, 64)
        windowed_eeg = windowed_eeg.transpose(0, 2, 1)  # (num_windows, 64, window_size)
        windowed_labels = np.array(windowed_labels).reshape(-1, 1)  # (num_windows, 1)

        # Apply preprocessing if requested
        if apply_normalize:
            windowed_eeg = data_norm_aved(windowed_eeg)

        if apply_ea:
            windowed_eeg = euclidean_alignment_aved(windowed_eeg)

        if apply_csp:
            windowed_eeg = apply_csp_transform(windowed_eeg, windowed_labels, n_components=args.eeg_channel)

        alldata.append(windowed_eeg)
        alllabel.append(windowed_labels)

    return alldata, alllabel


# ========================= AVED Preprocessing Functions =====================================
def data_norm_aved(data):
    """
    Per-sample max-absolute normalization for AVED dataset
    Normalizes each sample to [-1, 1] range by dividing by max absolute value

    Args:
        data: ndarray, shape [N, channels, samples]

    Returns:
        normalized data in [-1, 1] range
    """
    data_copy = np.copy(data)
    for i in range(len(data)):
        max_val = np.max(np.abs(data[i]))
        if max_val > 0:  # Avoid division by zero
            data_copy[i] = data_copy[i] / max_val
    return data_copy


def euclidean_alignment_aved(data):
    """
    Euclidean Alignment (EA) preprocessing for AVED dataset
    Applies whitening transformation to decorrelate EEG channels using covariance matrix

    This is a domain adaptation technique that:
    1. Computes average covariance matrix across all samples
    2. Applies inverse square root transformation to align data

    Args:
        data: ndarray, shape [N, channels, samples]

    Returns:
        aligned data with same shape [N, channels, samples]
    """
    # Compute covariance matrix: R_bar = sum(data_i @ data_i.T) for all i
    # Using Einstein summation for efficiency
    R_bar = np.einsum('ijk,ilk->jl', data, data)  # [channels, channels]
    R_bar_mean = R_bar / len(data)

    # Compute inverse square root of covariance matrix
    # This whitens the data by decorrelating channels
    inv_sqrt_R_bar_mean = np.linalg.inv(sqrtm(R_bar_mean))

    # Ensure result is real (sqrtm can introduce small imaginary components due to numerical errors)
    if np.iscomplexobj(inv_sqrt_R_bar_mean):
        inv_sqrt_R_bar_mean = np.real(inv_sqrt_R_bar_mean)

    # Apply transformation: data_aligned = inv_sqrt(R) @ data
    data_aligned = np.einsum('ij,kjm->kim', inv_sqrt_R_bar_mean, data)

    return data_aligned


def preprocess_aved(data):
    """
    Complete preprocessing pipeline for AVED dataset
    Applies both normalization and Euclidean Alignment

    Args:
        data: ndarray, shape [N, channels, samples]

    Returns:
        preprocessed data, shape [N, channels, samples]
    """
    # Step 1: Normalize to [-1, 1]
    data_normalized = data_norm_aved(data)

    # Step 2: Apply Euclidean Alignment
    data_ea = euclidean_alignment_aved(data_normalized)

    return data_ea


def apply_csp_transform(data, labels, n_components=64):
    """
    Apply CSP (Common Spatial Patterns) transformation
    Used by DARNet for spatial filtering

    Args:
        data: ndarray, shape [N, channels, samples]
        labels: ndarray, shape [N, 1] or [N,]
        n_components: number of CSP components to extract (default: 64)

    Returns:
        transformed data, shape [N, n_components, samples]
    """
    # CSP expects shape (trials, channels, samples)
    # Our data is already in this format

    # Flatten labels if needed
    if labels.ndim > 1:
        labels_flat = labels.flatten()
    else:
        labels_flat = labels

    # Initialize CSP
    csp = CSP(n_components=n_components, reg=None, log=None,
              cov_est='concat', transform_into='csp_space', norm_trace=True)

    # Fit and transform
    data_csp = csp.fit_transform(data, labels_flat)

    return data_csp


# ========================= AVED dataset =====================================
def getData_AVED(args, sub_ids, apply_csp=False, apply_ea=True, apply_normalize=True):
    """
    Load AVED (AHU-20) dataset with cross-subject setup

    Parameters:
    - args: arguments containing data_path, window_length, etc.
    - sub_ids: list of subject IDs to load (1-10)
    - apply_csp: Apply CSP transformation (default: False)
    - apply_ea: Apply Euclidean Alignment (default: True - matches ListenNet)
    - apply_normalize: Apply normalization (default: True - matches ListenNet)

    Returns:
    - seq_alldata: list of numpy arrays, one per subject (windows, channels, samples)
    - alllabel: list of numpy arrays, one per subject (windows, 1)

    Dataset info:
    - 10 subjects
    - 32 EEG channels
    - 16 trials per subject
    - 128 Hz sampling rate
    - Labels: alternating pattern [1,2,1,2,...] → converted to [0,1,0,1,...]
    """
    import pandas as pd

    seq_alldata = []
    alllabel = []

    # AVED parameters
    trail_number = 16
    eeg_channel = 32
    fs = 128

    # Calculate window parameters
    window_length = args.window_length  # Should be 128 for 1-second windows @ 128 Hz
    stride = window_length // 2  # 50% overlap

    print(f"\n{'='*80}")
    print(f"Loading AVED Dataset (AHU-20)")
    print(f"{'='*80}")
    print(f"Data path: {args.data_path}")
    print(f"Subjects to load: {sub_ids}")
    print(f"Channels: {eeg_channel}, Trials: {trail_number}, Sampling rate: {fs} Hz")
    print(f"Window: {window_length} samples ({args.win_time}s), Overlap: 50%")
    print(f"{'='*80}\n")

    for sub_id in sub_ids:
        print(f"Loading Subject {sub_id}...")

        # Load subject data from CSV
        filename = args.data_path + f"sub{sub_id}.csv"

        if not os.path.exists(filename):
            raise FileNotFoundError(f"AVED data file not found: {filename}")

        data_pf = pd.read_csv(filename, header=None)
        eeg_data = data_pf.values  # All data for this subject

        # Hardcoded labels (alternating pattern as in ListenNet)
        # Convert from [1,2,1,2,...] to [0,1,0,1,...]
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        # Reshape to trials: (16 trials, time_samples, 32 channels)
        samples_per_trial = eeg_data.shape[0] // trail_number
        eeg_data = eeg_data.reshape([trail_number, samples_per_trial, eeg_channel])

        print(f"  Raw data shape: ({trail_number}, {samples_per_trial}, {eeg_channel})")

        # Sliding window segmentation
        windows = []
        window_labels = []

        for trial_idx in range(trail_number):
            trial_data = eeg_data[trial_idx]  # (time_samples, 32)
            trial_label = labels[trial_idx]

            # Extract windows with overlap
            for i in range(0, trial_data.shape[0] - window_length + 1, stride):
                window = trial_data[i:i + window_length, :]  # (128, 32)
                windows.append(window)
                window_labels.append(trial_label)

        windows = np.array(windows)  # (num_windows, 128, 32)
        window_labels = np.array(window_labels).reshape(-1, 1)  # (num_windows, 1)

        # Transpose to (num_windows, 32, 128) to match model input format
        windows = windows.transpose(0, 2, 1)

        print(f"  After windowing: {windows.shape}")
        print(f"  Labels: {window_labels.shape}, Unique: {np.unique(window_labels)}")
        print(f"  Label distribution: 0={np.sum(window_labels==0)}, 1={np.sum(window_labels==1)}")

        # Apply preprocessing based on flags
        preprocessing_applied = []
        if apply_normalize:
            windows = data_norm_aved(windows)
            preprocessing_applied.append("Normalization")

        if apply_ea:
            windows = euclidean_alignment_aved(windows)
            preprocessing_applied.append("EA")

        if apply_csp:
            windows = apply_csp_transform(windows, window_labels, n_components=args.eeg_channel)
            preprocessing_applied.append("CSP")

        if preprocessing_applied:
            print(f"  Applied preprocessing: {', '.join(preprocessing_applied)}")
            print(f"  After preprocessing: {windows.shape}")

        seq_alldata.append(windows)
        alllabel.append(window_labels)

    print(f"\n{'='*80}")
    print(f"AVED Dataset Loaded Successfully!")
    print(f"Total subjects: {len(seq_alldata)}")
    print(f"{'='*80}\n")

    return seq_alldata, alllabel


# ========================= KUL dataset =====================================
def getData_KUL(args, sub_ids, apply_csp=False, apply_ea=True, apply_normalize=True):
    """
    Load KUL dataset from .mat files for auditory attention detection

    KUL Dataset Specs:
    - 16 subjects, 64 EEG channels
    - 20 trials per subject total (but we use only first 8 trials from experiments 1 and 2)
    - Sampling rate: 128 Hz (already downsampled and preprocessed)
    - Each trial: variable length (approximately 6 minutes each)
    - Experiments:
      - Experiment 1: 4 trials (attention to track 1)
      - Experiment 2: 4 trials (attention to track 2)
      - Experiment 3: 12 trials (repetitions - NOT USED)
    - Labels: attended_ear ('L' or 'R') → converted to 0/1
    - Data is already preprocessed: high-pass filtered (0.5 Hz), artifact-removed (MWF)

    Args:
        args: Arguments containing data_path, window_length, overlap
        sub_ids: List of subject IDs to load (1-16)
        apply_csp: Apply CSP transformation (default: False)
        apply_ea: Apply Euclidean Alignment (default: True - matches ListenNet)
        apply_normalize: Apply normalization (default: True - matches ListenNet)

    Returns:
        alldata: List of windowed EEG data per subject (num_windows, 64, window_size)
        alllabel: List of labels per subject (num_windows, 1)
    """
    alldata = []
    alllabel = []

    print(f"\n{'='*80}")
    print(f"Loading KUL Dataset")
    print(f"{'='*80}")
    print(f"Data path: {args.data_path}")
    print(f"Subjects to load: {sub_ids}")
    print(f"Channels: 64, Sampling rate: 128 Hz")
    print(f"Window: {args.window_length} samples ({args.win_time}s), Overlap: {int(args.overlap*100)}%")
    print(f"Using only Experiments 1 and 2 (8 trials per subject)")
    print(f"{'='*80}\n")

    for sub_id in sub_ids:
        print(f"Loading Subject S{sub_id}...")

        # Load subject .mat file
        mat_path = args.data_path + f'S{sub_id}.mat'

        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"KUL data file not found: {mat_path}")

        mat_data = loadmat(mat_path)

        # Extract trial data - filter for experiments 1 and 2 only (first 8 trials)
        # KUL structure: mat_data['trials'] is (1, 20) array of trial structs
        trial_eeg_list = []
        trial_labels = []

        trials_array = mat_data['trials']  # (1, 20)

        trial_count = 0
        for trial_idx in range(trials_array.shape[1]):  # Iterate over 20 trials
            trial = trials_array[0, trial_idx]

            # Extract experiment number
            experiment = int(trial['experiment'][0, 0].item())

            # Only use experiments 1 and 2 (skip experiment 3 repetitions)
            if experiment in [1, 2]:
                # Extract EEG data: RawData is (1,1), then [0,0] has fields, then EegData is (1,1)
                rawdata = trial['RawData'][0, 0]
                eeg_data = rawdata['EegData'][0, 0]  # (samples, 64)

                # Extract attended ear label
                attended_ear = trial['attended_ear'][0, 0][0]  # 'L' or 'R'
                label = 0 if attended_ear == 'L' else 1  # L=0, R=1

                trial_eeg_list.append(eeg_data)
                trial_labels.append(label)
                trial_count += 1

        if trial_count != 8:
            print(f"  WARNING: Expected 8 trials from experiments 1&2, found {trial_count}")

        print(f"  Loaded {trial_count} trials from experiments 1 and 2")

        # Apply sliding window segmentation across all trials
        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))

        windowed_eeg = []
        windowed_labels = []

        for trial_idx, (eeg, label) in enumerate(zip(trial_eeg_list, trial_labels)):
            # eeg shape: (time_samples, 64)

            # Sliding window
            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i + window_size, :]  # (window_size, 64)
                windowed_eeg.append(window)
                windowed_labels.append(label)

        # Convert to numpy arrays and transpose to (windows, channels, time)
        windowed_eeg = np.array(windowed_eeg)  # (num_windows, window_size, 64)
        windowed_eeg = windowed_eeg.transpose(0, 2, 1)  # (num_windows, 64, window_size)
        windowed_labels = np.array(windowed_labels).reshape(-1, 1)  # (num_windows, 1)

        print(f"  After windowing: {windowed_eeg.shape}")
        print(f"  Labels: {windowed_labels.shape}, Unique: {np.unique(windowed_labels)}")
        print(f"  Label distribution: L(0)={np.sum(windowed_labels==0)}, R(1)={np.sum(windowed_labels==1)}")

        # Apply preprocessing based on flags
        preprocessing_applied = []
        if apply_normalize:
            windowed_eeg = data_norm_aved(windowed_eeg)
            preprocessing_applied.append("Normalization")

        if apply_ea:
            windowed_eeg = euclidean_alignment_aved(windowed_eeg)
            preprocessing_applied.append("EA")

        if apply_csp:
            windowed_eeg = apply_csp_transform(windowed_eeg, windowed_labels, n_components=args.eeg_channel)
            preprocessing_applied.append("CSP")

        if preprocessing_applied:
            print(f"  Applied preprocessing: {', '.join(preprocessing_applied)}")
            print(f"  After preprocessing: {windowed_eeg.shape}")

        alldata.append(windowed_eeg)
        alllabel.append(windowed_labels)

    print(f"\n{'='*80}")
    print(f"KUL Dataset Loaded Successfully!")
    print(f"Total subjects: {len(alldata)}")
    print(f"{'='*80}\n")

    return alldata, alllabel


# ========================= model =====================================
def save_model(args, subject_name, best_acc, val_acc, model, epoch, model_name = None):
    print(f'Validation acc increase ({best_acc:.6f} --> {val_acc:.6f}) in epoch ({epoch}).  Saving model ...')
    # Save
    if model_name is None:
        model_save_path = args.model_save_path + subject_name + ".pt"
    else:
        model_save_path = args.model_save_path + model_name + ".pt"
    makePath(args.model_save_path)
    torch.save(model, model_save_path)
    

def load_model(path, subject_name, model_name = None):
    # Load
    if model_name is None:
        model_save_path = path + subject_name + ".pt"
    else:
        model_save_path = path + model_name + ".pt"
    model = torch.load(model_save_path, weights_only=False)
    return model
# import os
# import numpy as np
# from torch.utils.data import Dataset
# import torch
# import scipy.io as scio

# def makePath(path):
#     if not os.path.isdir(path):
#         os.makedirs(path)
#     return path


# class CustomDatasets(Dataset):
#     def __init__(self, data, event_data, subject_ids=None):
#         self.data = data
#         self.label = event_data
#         self.subject_ids = subject_ids

#     def __len__(self):
#         return len(self.label)

#     def __getitem__(self, index):
#         data = torch.Tensor(self.data[index]) 
#         label = torch.LongTensor(self.label[index])
        
#         if self.subject_ids is not None:
#             subject_id = torch.LongTensor([self.subject_ids[index]])
#             return data, label, subject_id
        
#         return data, label

# class EEGDataLoader(Dataset):
#     def __init__(self, x, y):
#         self.data = torch.from_numpy(x)
#         self.labels = torch.from_numpy(y)  # label without one-hot coding

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         data_tensor = self.data[idx]
#         label_tensor = self.labels[idx]
#         return data_tensor, label_tensor


# def getData(args,  sub_ids):
#     alldata = []
#     alllabel = [] 
#     for id in sub_ids:    
#         onedata = np.load(args.data_path +  f'S{id}.npy')
#         onelabel = np.load(args.label_path +  f'S{id}.npy')
#         onedata = onedata.transpose(0,2,1)
#         alldata.append(onedata)
#         alllabel.append(onelabel)
#     return alldata,  alllabel

# # ========================= model =====================================
# def save_model(args, subject_name, best_acc, val_acc, model, epoch, model_name = None):
#     print(f'Validation acc increase ({best_acc:.6f} --> {val_acc:.6f}) in epoch ({epoch}).  Saving model ...')
#     # Save
#     if model_name is None:
#         model_save_path = args.model_save_path + subject_name + ".pt"
#     else:
#         model_save_path = args.model_save_path + model_name + ".pt"
#     makePath(args.model_save_path)
#     torch.save(model, model_save_path)
    

# def load_model(path, subject_name, model_name = None):
#     # Load
#     if model_name is None:
#         model_save_path = path + subject_name + ".pt"
#     else:
#         model_save_path = path + model_name + ".pt"
#     model = torch.load(model_save_path)
#     return model 

