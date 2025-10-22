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


class SSLCustomDatasets(Dataset):
    """SSL Dataset without labels but with subject IDs for domain adversarial training"""
    def __init__(self, data, subject_ids=None):
        self.data = data
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.Tensor(self.data[index])
        
        if self.subject_ids is not None:
            subject_id = self.subject_ids[index]  # Return scalar, not tensor
            return data, subject_id
        
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
    
    print(f"Loading fine-tuning data with target window length: {args.win_len} time steps")
    
    for id in sub_ids:
        onedata = np.load(args.data_path + f'S{id}.npy')
        onelabel = np.load(args.label_path + f'S{id}.npy')
        onedata = onedata.transpose(0, 2, 1)
        
        # Handle different window sizes for fine-tuning data too
        if args.win_len > onedata.shape[2]:
            # Need to create longer windows by concatenating overlapping segments
            onedata = create_longer_windows(onedata, target_len=args.win_len)
        elif args.win_len < onedata.shape[2]:
            # Truncate to desired length
            onedata = onedata[:, :, :args.win_len]
        
        alldata.append(onedata)
        alllabel.append(onelabel)
        print(f"Loaded fine-tuning subject S{id}: {onedata.shape}")
    
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


def getSSLDataWithSubjects(args, sub_ids, mask_ratio=0.15):
    """Load EEG data for SSL pre-training with subject IDs for domain adversarial training"""
    alldata = []
    all_subject_ids = []
    
    print(f"Loading SSL data with subject IDs for subjects: {sub_ids}")
    print(f"Target window length: {args.win_len} time steps")
    
    for id in sub_ids:    
        onedata = np.load(args.data_path + f'S{id}.npy')
        onedata = onedata.transpose(0, 2, 1)  # [trials, channels, time]
        
        # Handle different window sizes
        if args.win_len > onedata.shape[2]:
            # Need to create longer windows by concatenating overlapping segments
            print(f"Creating {args.win_len}-step windows from {onedata.shape[2]}-step data")
            onedata = create_longer_windows(onedata, target_len=args.win_len)
        elif args.win_len < onedata.shape[2]:
            # Truncate to desired length
            onedata = onedata[:, :, :args.win_len]
        
        alldata.append(onedata)
        
        # Create subject ID array for each trial (0-based indexing for discriminator)
        subject_id = id - 1  # Convert to 0-based
        subject_ids_for_this_subject = [subject_id] * len(onedata)
        all_subject_ids.extend(subject_ids_for_this_subject)
        
        print(f"Loaded subject S{id}: {onedata.shape}, Subject ID: {subject_id}")
    
    # Concatenate all subjects data
    all_ssl_data = np.concatenate(alldata, axis=0)
    print(f"Total SSL data shape: {all_ssl_data.shape}")
    print(f"Total subject IDs: {len(all_subject_ids)}")
    
    return all_ssl_data, all_subject_ids


def create_longer_windows(data, target_len):
    """Create longer windows by concatenating overlapping segments"""
    trials, channels, time_steps = data.shape
    
    if target_len <= time_steps:
        return data[:, :, :target_len]
    
    # Calculate how many segments we need
    segments_needed = target_len // time_steps
    remainder = target_len % time_steps
    
    new_data = []
    for trial_idx in range(trials):
        trial_data = data[trial_idx]  # [channels, time_steps]
        
        # Concatenate multiple segments
        segments = []
        for seg in range(segments_needed):
            segments.append(trial_data)
        
        # Add partial segment if needed
        if remainder > 0:
            segments.append(trial_data[:, :remainder])
        
        # Concatenate along time dimension
        new_trial = np.concatenate(segments, axis=1)  # [channels, target_len]
        new_data.append(new_trial)
    
    return np.array(new_data)  # [trials, channels, target_len]


def create_masked_batch(batch_data, mask_ratio=0.15):
    """Create masked batch for SSL training (temporal masking)"""
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


def create_patch_masked_batch(batch_data, patch_size=(4, 16), mask_ratio=0.15):
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


def create_masked_batch_unified(batch_data, mask_ratio=0.15, use_patch_masking=False, patch_size=(4, 16)):
    """Unified masking function - supports both temporal and patch-based masking"""
    if use_patch_masking:
        return create_patch_masked_batch(batch_data, patch_size, mask_ratio)
    else:
        return create_masked_batch(batch_data, mask_ratio)


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


def full_darnet_reconstruction_loss(reconstructed, original, mask_indices_batch, device, use_patch_masking=False):
    """Unified reconstruction loss for full DARNet SSL approach"""
    if use_patch_masking:
        return patch_reconstruction_loss(reconstructed, original, mask_indices_batch, device)
    else:
        # Original temporal masking loss
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


def domain_adversarial_loss(subject_logits, subject_labels, device):
    """Domain adversarial loss for subject invariant features"""
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(subject_logits, subject_labels.to(device))


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