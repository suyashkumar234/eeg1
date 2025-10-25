import os
import numpy as np
from torch.utils.data import Dataset
import torch
import scipy.io as scio

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


def getData(args,  sub_ids):
    alldata = []
    alllabel = [] 
    for id in sub_ids:    
        onedata = np.load(args.data_path +  f'S{id}.npy')
        onelabel = np.load(args.label_path +  f'S{id}.npy')
        onedata = onedata.transpose(0,2,1)
        alldata.append(onedata)
        alllabel.append(onelabel)
    return alldata,  alllabel

# ========================= Augmentation utilities =====================================
def get_augmentation_statistics():
    """Get statistics about applied augmentations for monitoring"""
    stats = {
        'total_batches': 0,
        'augmented_batches': 0,
        'time_shift_count': 0,
        'noise_count': 0,
        'time_mask_count': 0,
        'channel_dropout_count': 0,
        'amplitude_scale_count': 0
    }
    return stats

def log_augmentation_stats(stats, epoch):
    """Log augmentation statistics for monitoring"""
    if stats['total_batches'] > 0:
        aug_rate = stats['augmented_batches'] / stats['total_batches'] * 100
        print(f"Epoch {epoch} Augmentation Stats:")
        print(f"  - Augmentation rate: {aug_rate:.1f}%")
        print(f"  - Time shifts: {stats['time_shift_count']}")
        print(f"  - Noise additions: {stats['noise_count']}")
        print(f"  - Time masks: {stats['time_mask_count']}")
        print(f"  - Channel dropouts: {stats['channel_dropout_count']}")
        print(f"  - Amplitude scales: {stats['amplitude_scale_count']}")

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
    model = torch.load(model_save_path)
    return model