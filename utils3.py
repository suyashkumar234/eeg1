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


def getSSLData(args, sub_ids, mask_ratio=0.75, use_patch_masking=True):
    """Load EEG data for SSL pre-training (no labels needed)"""
    alldata = []
    
    print(f"Loading SSL data for subjects: {sub_ids}")
    print(f"SSL Config: mask_ratio={mask_ratio}, patch_masking={use_patch_masking}")
    
    for id in sub_ids:    
        onedata = np.load(args.data_path + f'S{id}.npy')
        onedata = onedata.transpose(0, 2, 1)  # [trials, channels, time]
        alldata.append(onedata)
        print(f"Loaded subject S{id}: {onedata.shape}")
    
    # Concatenate all subjects data
    all_ssl_data = np.concatenate(alldata, axis=0)
    print(f"Total SSL data shape: {all_ssl_data.shape}")
    
    if use_patch_masking:
        # Calculate patch statistics
        channels, time_steps = all_ssl_data.shape[1], all_ssl_data.shape[2]
        patch_h, patch_w = 4, 16  # Default patch size
        num_patches = (channels // patch_h) * (time_steps // patch_w)
        num_masked = int(num_patches * mask_ratio)
        print(f"Patch info: {channels}×{time_steps} → {num_patches} patches, masking {num_masked} ({mask_ratio*100:.1f}%)")
    
    return all_ssl_data


def create_patch_masked_batch(batch_data, patch_size=(4, 16), mask_ratio=0.75):
    """Create patch-based masked batch for SSL training (MAE-EEG style)"""
    batch_size, channels, time_steps = batch_data.shape  # [batch, 32, 128]
    
    # Calculate patch dimensions
    patch_h, patch_w = patch_size  # (channels, time)
    num_patches_h = channels // patch_h  # 32 // 4 = 8 spatial patches
    num_patches_w = time_steps // patch_w  # 128 // 16 = 8 temporal patches
    total_patches = num_patches_h * num_patches_w  # 8 × 8 = 64 total patches
    
    masked_batch = batch_data.clone()
    mask_info_batch = []
    
    for i in range(batch_size):
        # Random patch masking
        num_masked = int(total_patches * mask_ratio)
        mask_patch_indices = torch.randperm(total_patches)[:num_masked]
        
        # Store mask information for loss calculation
        masked_patches = []
        
        for patch_idx in mask_patch_indices:
            # Convert linear patch index to 2D coordinates
            patch_row = patch_idx // num_patches_w  # Spatial patch index
            patch_col = patch_idx % num_patches_w   # Temporal patch index
            
            # Calculate actual coordinates in EEG data
            ch_start = patch_row * patch_h
            ch_end = ch_start + patch_h
            t_start = patch_col * patch_w
            t_end = t_start + patch_w
            
            # Zero out this patch
            masked_batch[i, ch_start:ch_end, t_start:t_end] = 0
            
            # Store patch coordinates for loss calculation
            masked_patches.append({
                'ch_start': ch_start, 'ch_end': ch_end,
                't_start': t_start, 't_end': t_end,
                'patch_idx': patch_idx
            })
        
        mask_info_batch.append(masked_patches)
    
    return masked_batch, mask_info_batch


def create_masked_batch(batch_data, mask_ratio=0.75, use_patch_masking=True, patch_size=(4, 16)):
    """Unified masking function - supports both temporal and patch-based masking"""
    if use_patch_masking:
        return create_patch_masked_batch(batch_data, patch_size, mask_ratio)
    else:
        # Original temporal masking (backward compatibility)
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


def transformer_reconstruction_loss(reconstructed, original, mask_info_batch, device):
    """Transformer decoder reconstruction loss with enhanced objectives"""
    batch_size = reconstructed.shape[0]
    total_loss = 0
    
    for i in range(batch_size):
        masked_patches = mask_info_batch[i]
        sample_loss = 0
        
        # Patch-based reconstruction loss
        for patch_info in masked_patches:
            ch_start, ch_end = patch_info['ch_start'], patch_info['ch_end']
            t_start, t_end = patch_info['t_start'], patch_info['t_end']
            
            # Extract patch regions
            recon_patch = reconstructed[i, ch_start:ch_end, t_start:t_end]
            orig_patch = original[i, ch_start:ch_end, t_start:t_end]
            
            # MSE loss on masked patch
            patch_mse = torch.nn.functional.mse_loss(recon_patch, orig_patch)
            
            # L1 loss on masked patch (for sharper details)
            patch_l1 = torch.nn.functional.l1_loss(recon_patch, orig_patch)
            
            # Combined patch loss
            patch_loss = 0.6 * patch_mse + 0.4 * patch_l1
            sample_loss += patch_loss
        
        # Normalize by number of masked patches
        if len(masked_patches) > 0:
            sample_loss = sample_loss / len(masked_patches)
        
        # Global consistency loss (lighter weight for transformer)
        global_loss = 0.05 * torch.nn.functional.mse_loss(reconstructed[i], original[i])
        sample_loss += global_loss
        
        # Spectral loss (for EEG frequency preservation)
        recon_fft = torch.fft.fft(reconstructed[i], dim=-1)
        orig_fft = torch.fft.fft(original[i], dim=-1)
        spectral_loss = 0.1 * torch.nn.functional.mse_loss(
            torch.abs(recon_fft), torch.abs(orig_fft)
        )
        sample_loss += spectral_loss
        
        total_loss += sample_loss
    
    return total_loss / batch_size


def patch_reconstruction_loss(reconstructed, original, mask_info_batch, device):
    """Patch-based reconstruction loss for MAE-EEG style training"""
    batch_size = reconstructed.shape[0]
    total_loss = 0
    
    for i in range(batch_size):
        masked_patches = mask_info_batch[i]
        sample_loss = 0
        
        for patch_info in masked_patches:
            ch_start, ch_end = patch_info['ch_start'], patch_info['ch_end']
            t_start, t_end = patch_info['t_start'], patch_info['t_end']
            
            # Extract patch regions
            recon_patch = reconstructed[i, ch_start:ch_end, t_start:t_end]
            orig_patch = original[i, ch_start:ch_end, t_start:t_end]
            
            # MSE loss on masked patch
            patch_mse = torch.nn.functional.mse_loss(recon_patch, orig_patch)
            
            # L1 loss on masked patch (for sharper details)
            patch_l1 = torch.nn.functional.l1_loss(recon_patch, orig_patch)
            
            # Combined patch loss
            patch_loss = 0.7 * patch_mse + 0.3 * patch_l1
            sample_loss += patch_loss
        
        # Normalize by number of masked patches
        if len(masked_patches) > 0:
            sample_loss = sample_loss / len(masked_patches)
        
        # Small global consistency loss (optional)
        global_loss = 0.1 * torch.nn.functional.mse_loss(reconstructed[i], original[i])
        sample_loss += global_loss
        
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