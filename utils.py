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
    def __init__(self, data, event_data):
        self.data = data
        self.label = event_data

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = torch.Tensor(self.data[index])
        label = torch.LongTensor(self.label[index])

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

def getData(args,  sub_ids, analyze_variation=False, normalize=True):
    alldata = []
    alllabel = [] 
    
    if analyze_variation:
        print("=" * 80)
        print("Subject-wise EEG Data Analysis" + (" (Before Normalization)" if normalize else ""))
        print("=" * 80)
        print(f"{'Subject':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Shape'}")
        print("-" * 80)
    
    for id in sub_ids:    
        onedata = np.load(args.data_path +  f'S{id}.npy')
        onelabel = np.load(args.label_path +  f'S{id}.npy')
        onedata = onedata.transpose(0,2,1)
        
        if analyze_variation:
            # Show stats before normalization
            data_mean = np.mean(onedata)
            data_std = np.std(onedata)
            data_min = np.min(onedata)
            data_max = np.max(onedata)
            print(f"S{id:<7} {data_mean:<12.6f} {data_std:<12.6f} {data_min:<12.6f} {data_max:<12.6f} {onedata.shape}")
        
        # Apply subject-wise z-score normalization
        if normalize:
            subject_mean = np.mean(onedata, axis=(0, 2), keepdims=True)  # Mean across trials and time, keep channel dim
            subject_std = np.std(onedata, axis=(0, 2), keepdims=True) + 1e-8  # Avoid division by zero
            onedata = (onedata - subject_mean) / subject_std
        
        alldata.append(onedata)
        alllabel.append(onelabel)
    
    if analyze_variation:
        print("-" * 80)
        # Overall statistics
        all_data_concat = np.concatenate(alldata, axis=0)
        overall_mean = np.mean(all_data_concat)
        overall_std = np.std(all_data_concat)
        overall_min = np.min(all_data_concat)
        overall_max = np.max(all_data_concat)
        print(f"Overall: Mean={overall_mean:.6f}, Std={overall_std:.6f}")
        print(f"Overall: Min={overall_min:.6f}, Max={overall_max:.6f}")
        
        if normalize:
            print("\nAfter Normalization - Subject-wise Statistics:")
            print("-" * 80)
            print(f"{'Subject':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
            print("-" * 80)
            for i, (id, data) in enumerate(zip(sub_ids, alldata)):
                data_mean = np.mean(data)
                data_std = np.std(data)
                data_min = np.min(data)
                data_max = np.max(data)
                print(f"S{id:<7} {data_mean:<12.6f} {data_std:<12.6f} {data_min:<12.6f} {data_max:<12.6f}")
        
        print("=" * 80)
        
    return alldata,  alllabel
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

