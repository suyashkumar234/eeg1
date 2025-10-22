import os
import numpy as np
from torch.utils.data import Dataset
import torch
import scipy.io as scio


def makePath(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
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


class SSLCustomDatasets(Dataset):
    """SSL Dataset without labels"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.Tensor(self.data[index])
        return data


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


def getData(args, sub_ids):
    alldata = []
    alllabel = []
    for id in sub_ids:
        onedata = np.load(args.data_path + f'S{id}.npy')
        onelabel = np.load(args.label_path + f'S{id}.npy')
        onedata = onedata.transpose(0, 2, 1)
        alldata.append(onedata)
        alllabel.append(onelabel)
    return alldata, alllabel


def getSSLData(args, sub_ids, mask_ratio=0.15):
    """Load EEG data for SSL pre-training (no labels needed)"""
    alldata = []
    
    print(f"Loading SSL data for subjects: {sub_ids}")
    
    for id in sub_ids:    
        onedata = np.load(args.data_path + f'S{id}.npy')
        onedata = onedata.transpose(0, 2, 1)  # [trials, channels, time]
        alldata.append(onedata)
        print(f"Loaded subject S{id}: {onedata.shape}")
    
    # Concatenate all subjects data
    all_ssl_data = np.concatenate(alldata, axis=0)
    print(f"Total SSL data shape: {all_ssl_data.shape}")
    
    return all_ssl_data


def create_masked_batch(batch_data, mask_ratio=0.15):
    """Create masked batch for SSL training"""
    batch_size, channels, time_steps = batch_data.shape
    
    masked_batch = batch_data.clone()
    mask_indices_batch = []
    
    for i in range(batch_size):
        # Random temporal masking for each sample
        num_masked = int(time_steps * mask_ratio)
        mask_indices = torch.randperm(time_steps)[:num_masked]
        mask_indices_batch.append(mask_indices)
        
        # Zero out masked regions
        masked_batch[i, :, mask_indices] = 0
    
    return masked_batch, mask_indices_batch


def ssl_reconstruction_loss(reconstructed, original, mask_indices_batch, device, use_perceptual_loss=False):
    """Calculate reconstruction loss only on masked regions with optional perceptual loss"""
    batch_size = reconstructed.shape[0]
    total_loss = 0
    
    for i in range(batch_size):
        mask_indices = mask_indices_batch[i].to(device)
        
        # Primary MSE loss on masked regions
        mse_loss = torch.nn.functional.mse_loss(
            reconstructed[i, :, mask_indices], 
            original[i, :, mask_indices]
        )
        
        sample_loss = mse_loss
        
        # Optional: Add L1 loss for sharper reconstruction
        l1_loss = torch.nn.functional.l1_loss(
            reconstructed[i, :, mask_indices], 
            original[i, :, mask_indices]
        )
        
        # Combine losses (MSE for smoothness + L1 for sharpness)
        sample_loss = 0.8 * mse_loss + 0.2 * l1_loss
        
        total_loss += sample_loss
    
    return total_loss / batch_size


def full_darnet_reconstruction_loss(reconstructed, original, mask_indices_batch, device):
    """Reconstruction loss for full DARNet SSL approach"""
    batch_size = reconstructed.shape[0]
    total_loss = 0
    
    for i in range(batch_size):
        mask_indices = mask_indices_batch[i].to(device)
        
        # Primary MSE loss on masked regions
        mse_masked = torch.nn.functional.mse_loss(
            reconstructed[i, :, mask_indices], 
            original[i, :, mask_indices]
        )
        
        # L1 loss on masked regions (for sharper reconstruction)
        l1_masked = torch.nn.functional.l1_loss(
            reconstructed[i, :, mask_indices], 
            original[i, :, mask_indices]
        )
        
        # Global consistency loss (since we're reconstructing from highly compressed features)
        global_loss = 0.2 * torch.nn.functional.mse_loss(
            reconstructed[i], original[i]
        )
        
        # Combined loss - focus more on global consistency for full DARNet approach
        sample_loss = 0.5 * mse_masked + 0.3 * l1_masked + global_loss
        total_loss += sample_loss
    
    return total_loss / batch_size


def unet_reconstruction_loss(reconstructed, original, mask_indices_batch, device):
    """Enhanced reconstruction loss for U-Net decoder"""
    batch_size = reconstructed.shape[0]
    total_loss = 0
    
    for i in range(batch_size):
        mask_indices = mask_indices_batch[i].to(device)
        
        # MSE loss on masked regions (primary objective)
        mse_masked = torch.nn.functional.mse_loss(
            reconstructed[i, :, mask_indices], 
            original[i, :, mask_indices]
        )
        
        # L1 loss on masked regions (for sharper details)
        l1_masked = torch.nn.functional.l1_loss(
            reconstructed[i, :, mask_indices], 
            original[i, :, mask_indices]
        )
        
        # Small global consistency loss (helps with skip connections)
        global_loss = 0.1 * torch.nn.functional.mse_loss(
            reconstructed[i], original[i]
        )
        
        # Combined loss
        sample_loss = 0.7 * mse_masked + 0.2 * l1_masked + global_loss
        total_loss += sample_loss
    
    return total_loss / batch_size


# ========================= model =====================================
def save_model(args, subject_name, best_acc, val_acc, model, epoch, model_name=None):
    print(f'Validation acc increase ({best_acc:.6f} --> {val_acc:.6f}) in epoch ({epoch}).  Saving model ...')
    # Save
    if model_name is None:
        model_save_path = args.model_save_path + subject_name + ".pt"
    else:
        model_save_path = args.model_save_path + model_name + ".pt"
    makePath(args.model_save_path)
    torch.save(model, model_save_path)


def load_model(path, subject_name, model_name=None):
    # Load
    if model_name is None:
        model_save_path = path + subject_name + ".pt"
    else:
        model_save_path = path + model_name + ".pt"
    model = torch.load(model_save_path)
    return model


def save_ssl_model(model, save_path, epoch, loss):
    """Save SSL pre-trained model"""
    makePath(os.path.dirname(save_path))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, save_path)
    print(f'SSL model saved at epoch {epoch} with loss {loss:.6f} to {save_path}')