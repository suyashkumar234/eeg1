from __future__ import division
from __future__ import print_function
import os
import math

import numpy as np
import pandas as pd

from datetime import datetime
import argparse
import copy

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from torch.optim.optimizer import Optimizer
from typing import Optional
import torchinfo
import matplotlib.pyplot as plt

from utils3 import *
from collections import OrderedDict
from model_module3 import DARNet_ViT_Decoder


def set_all_seeds(seed):
    """Set all possible seeds for reproducibility"""
    import random
    import os
    
    # Python random seed
    random.seed(seed)
    
    # Numpy seed
    np.random.seed(seed)
    
    # PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # PyTorch deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set worker seed for DataLoader
    def worker_init_fn(worker_id):
        # Each worker gets a different but deterministic seed
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    return worker_init_fn


def get_device():
    """Get the best available device: CUDA, MPS (Apple Silicon), or CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")


if "SLURM_JOB_GPUS" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_JOB_GPUS"]

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current device is", device)


class StepwiseLR_GRL:
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.001,
                 gamma: Optional[float] = 0.01, decay_rate: Optional[float] = 0.1, max_iter: Optional[float] = 100):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter = max_iter

    def get_lr(self) -> float:
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num / self.max_iter)) ** (self.decay_rate)
        if lr <= 1e-8:
            lr = 1e-8
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']
        self.iter_num += 1


class SSLTrainer():
    """SSL Pre-training trainer with Transformer decoder"""
    def __init__(self, model, ssl_loader, batch_size, lr, weight_decay, worker_init_fn):
        self.model = model
        self.ssl_loader = ssl_loader
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.worker_init_fn = worker_init_fn
        
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.MSELoss()
        
        # Training logs
        self.ssl_df = pd.DataFrame()
        
    def ssl_step(self):
        """SSL pre-training step with Transformer decoder"""
        self.model.set_training_mode('ssl')
        self.model.train()
        
        ssl_losses = []
        
        for i_batch, batch_data in enumerate(self.ssl_loader):
            eeg_data = batch_data.to(device).float()  # [batch, 32, 128]
            
            # Create patch-based masked version (MAE-EEG style)
            masked_eeg, mask_info_batch = create_masked_batch(
                eeg_data, 
                mask_ratio=0.25,  # Higher masking ratio like MAE-EEG
                use_patch_masking=True, 
                patch_size=(4, 16)  # 4 channels × 16 time steps per patch
            )
            
            # Forward pass: reconstruct original from masked
            reconstructed = self.model(masked_eeg)  # [batch, 32, 128]
            
            # Transformer-specific reconstruction loss
            ssl_loss = transformer_reconstruction_loss(
                reconstructed, eeg_data, mask_info_batch, device
            )
            
            ssl_losses.append(ssl_loss.cpu().detach().numpy())
            
            # Backward and optimize
            self.optimizer.zero_grad()
            ssl_loss.backward()
            self.optimizer.step()
        
        epoch_ssl_loss = sum(ssl_losses) / len(ssl_losses)
        
        # Log SSL training
        ssl_dict = {'ssl_loss': [epoch_ssl_loss]}
        self.ssl_df = pd.concat([self.ssl_df, pd.DataFrame(ssl_dict)], ignore_index=True)
        
        return epoch_ssl_loss
    
    def ssl_train(self, args, ssl_epochs=30):
        """SSL pre-training loop"""
        print("=" * 50)
        print("Starting SSL Pre-training with Transformer Decoder")
        print("=" * 50)
        
        torchinfo.summary(self.model)
        
        best_ssl_loss = float('inf')
        
        for epoch in range(1, ssl_epochs + 1):
            ssl_loss = self.ssl_step()
            
            print(f'SSL Epoch {epoch:2d} | SSL Loss: {ssl_loss:.6f}')
            
            # Save best SSL model
            if ssl_loss < best_ssl_loss:
                best_ssl_loss = ssl_loss
                ssl_model_path = args.ssl_model_save_path + "ssl_best_model.pt"
                save_ssl_model(self.model, ssl_model_path, epoch, ssl_loss)
                print(f'SSL model saved with loss {ssl_loss:.6f}')
        
        print("=" * 50)
        print(f"SSL Pre-training completed. Best loss: {best_ssl_loss:.6f}")
        print("=" * 50)
        
        return ssl_model_path


class Trynetwork():
    """Fine-tuning trainer (based on original trainer)"""
    def __init__(self, model, train_loader, valid_loader, test_loader, batch_size, lr, weight_decay, worker_init_fn):
        self.model = model
        self.datasets = OrderedDict((("train", train_loader), ("valid", valid_loader), ("test", test_loader)))
        if valid_loader is None:
            self.datasets.pop("valid")
        if test_loader is None:
            self.datasets.pop("test")
        self.best_test = 0
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.worker_init_fn = worker_init_fn
        
        # Create optimizer based on args
        if args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr, 
                                           momentum=getattr(args, 'momentum', 0.9), 
                                           weight_decay=self.weight_decay)
        elif args.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(params=self.model.parameters(), lr=self.lr,
                                               alpha=getattr(args, 'alpha', 0.99),
                                               weight_decay=self.weight_decay)
        elif args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr,
                                            betas=(getattr(args, 'beta1', 0.9), getattr(args, 'beta2', 0.999)),
                                            weight_decay=self.weight_decay)
        else:  # Default to AdamW
            self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr,
                                             betas=(getattr(args, 'beta1', 0.9), getattr(args, 'beta2', 0.999)),
                                             weight_decay=self.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 35], gamma=0.5)
        self.scheduler_down = StepwiseLR_GRL(self.optimizer, init_lr=args.lr, gamma=10, decay_rate=args.lr_decayrate, max_iter=args.max_epoch)
        self.criterion = nn.CrossEntropyLoss()
        
        # initialize epoch dataFrame instead of loss and acc for train and test
        self.val_df = pd.DataFrame()
        self.train_df = pd.DataFrame()
        self.epoch_df = pd.DataFrame()

    def __getModel__(self):
        return self.model

    def save_acc_loss_fig(self, args, sub_id):
        valid_acc = self.epoch_df['valid_acc'].values.tolist()
        valid_loss = self.epoch_df['valid_loss'].values.tolist()
        train_acc = self.epoch_df['train_acc'].values.tolist()
        train_loss = self.epoch_df['train_loss'].values.tolist()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # First subgraph: Accuracy and loss of training data
        ax1.plot(range(len(train_acc)), train_acc, label='Train Accuracy', color='blue', linewidth=0.7)
        ax1.plot(range(len(valid_acc)), valid_acc, label='Valid Accuracy', color='red', linewidth=0.7)
        ax1.set_title('Acc Performance')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc='upper right')

        # Second subgraph: Accuracy and loss of test data
        ax2.plot(range(len(train_loss)), train_loss, label='Train Loss', color='green', linewidth=0.7)
        ax2.plot(range(len(valid_loss)), valid_loss, label='Valid Loss', color='purple', linewidth=0.7)
        ax2.set_title('Loss Performance')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(args.fig_path, 'Loss_Acc.png'))

    def train_step(self):
        """Fine-tuning training step (same as original trainer)"""
        self.model.set_training_mode('aad')  # Set to AAD mode
        self.model.train()
        train_dicts_per_epoch = OrderedDict()
        Batch_size, Cls_loss, Train_acc = [], [], []
        
        for i_batch, batch_data in enumerate(self.datasets['train']):
            train_data, train_label = batch_data
            train_label = train_label.squeeze(-1)
            train_data, train_label = train_data.to(device).float(), train_label.to(device).long()

            source_softmax = self.model(train_data)
            nll_loss = self.criterion(source_softmax, train_label)

            Batch_size.append(len(train_label))
            _, predicted = torch.max(source_softmax.data, 1)
            batch_acc = np.equal(predicted.cpu().detach().numpy(), train_label.cpu().detach().numpy()).sum() / len(train_label)
            # Forward pass
            Train_acc.append(batch_acc)
            cls_loss_np = nll_loss.cpu().detach().numpy()
            Cls_loss.append(cls_loss_np)

            # Backward and optimize
            self.optimizer.zero_grad()
            nll_loss.backward()
            self.optimizer.step()

        epoch_acc = sum(Train_acc) / len(Train_acc) * 100
        epoch_loss = sum(Cls_loss) / len(Cls_loss)

        cls_loss = {'train_loss': epoch_loss}
        train_acc = {'train_acc': epoch_acc}
        train_dicts_per_epoch.update(cls_loss)
        train_dicts_per_epoch.update(train_acc)
        train_dicts_per_epoch = {k: [v] for k, v in train_dicts_per_epoch.items()}
        self.train_df = pd.concat([self.train_df, pd.DataFrame(train_dicts_per_epoch)], ignore_index=True)
        self.train_df = self.train_df[list(train_dicts_per_epoch.keys())]
        return epoch_loss, epoch_acc

    def test_batch(self, input, label):
        self.model.set_training_mode('aad')  # Set to AAD mode
        self.model.eval()
        with torch.no_grad():
            val_input = input.to(device).float()
            val_label = label.to(device).long()
            val_fc1 = self.model(val_input)
            loss = self.criterion(val_fc1, val_label)
            _, preds = torch.max(val_fc1.data, 1)
            preds = preds.cpu().detach().numpy()
            loss = loss.cpu().detach().numpy()
        return preds, loss

    def evaluate_step(self, Flag_test):
        if Flag_test and "test" in self.datasets:
            setname = 'test'
        else:
            setname = 'valid'
        result_dicts_per_monitor = OrderedDict()
        with torch.no_grad():
            Batch_size, Epochs_loss, Epochs_acc = [], [], []
            for i_batch, batch_data in enumerate(self.datasets[setname]):
                seq_input, target = batch_data
                target = target.squeeze(-1)
                pred, loss = self.test_batch(seq_input, target)
                Epochs_loss.append(loss)
                Batch_size.append(len(target))
                Epochs_acc.append(np.equal(pred, target.numpy()).sum())
        epoch_acc = sum(Epochs_acc) / sum(Batch_size) * 100
        epoch_loss = sum(Epochs_loss) / len(Epochs_loss)
        key_loss = setname + '_loss'
        key_acc = setname + '_acc'
        loss = {key_loss: epoch_loss}
        acc = {key_acc: epoch_acc}
        result_dicts_per_monitor.update(loss)
        result_dicts_per_monitor.update(acc)
        result_dicts_per_monitor = {k: [v] for k, v in result_dicts_per_monitor.items()}
        self.val_df = pd.concat([self.val_df, pd.DataFrame(result_dicts_per_monitor)], ignore_index=True)
        self.val_df = self.val_df[list(result_dicts_per_monitor.keys())]
        return epoch_loss, epoch_acc

    def train(self, args, testsub_id, ssl_model_path=None):
        """Fine-tuning training loop"""
        # Load SSL pre-trained weights if provided
        if ssl_model_path:
            print(f"🔄 Loading SSL pre-trained weights from: {ssl_model_path}")
            checkpoint = torch.load(ssl_model_path, map_location=device)
            
            # Debug: Show all available keys in checkpoint
            all_keys = list(checkpoint['model_state_dict'].keys())
            print(f"📋 Available keys in SSL checkpoint: {len(all_keys)} total")
            for key in all_keys:
                print(f"  - {key}: {checkpoint['model_state_dict'][key].shape}")
            
            # Load encoder weights (token_embedding, stack1, stack2)
            encoder_state_dict = {}
            
            for key, value in checkpoint['model_state_dict'].items():
                # Load all encoder components for ViT decoder approach
                if key.startswith(('token_embedding', 'stack1', 'stack2')):
                    encoder_state_dict[key] = value
                    print(f"  ✅ Loading: {key} -> {value.shape}")
                else:
                    print(f"  ❌ Skipping: {key} (decoder component)")
            
            print(f"🔧 Loading {len(encoder_state_dict)} encoder components into fine-tuning model...")
            missing_keys, unexpected_keys = self.model.load_state_dict(encoder_state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️  Missing keys (newly initialized): {missing_keys}")
            if unexpected_keys:
                print(f"❗ Unexpected keys: {unexpected_keys}")
                
            print(f"✅ Successfully loaded SSL pre-trained weights!")
            print(f"🎯 Transformer decoder removed, feature classifier ready for fine-tuning")
        
        print("=" * 50)
        print("Starting Fine-tuning")
        print("=" * 50)
        
        torchinfo.summary(self.model)
        testsub_name = 'S' + str(testsub_id)
        
        best_epoch = 0
        best_acc = 0
        early_stopping_patience = 30
        epochs_without_improvement = 0
        
        for epoch in range(1, args.max_epoch + 1):
            train_loss, train_acc = self.train_step()
            val_loss, val_acc = self.evaluate_step(False)
            
            print('TestSub:', testsub_name,
                  'Epoch {:2d} Finsh | Now_lr {:2.4f}/{:2.4f}|Train Loss {:2.4f} | Valid Loss {:2.4f} | Train Acc {:5.4f}| Valid Acc {:5.4f}'.format(epoch,
                                                                                                                                                self.optimizer.param_groups[0]["lr"], args.lr,
                                                                                                                                                train_loss,
                                                                                                                                                val_loss,
                                                                                                                                                train_acc,
                                                                                                                                                val_acc))
            if val_acc > best_acc:
                save_model(args, testsub_name, best_acc, val_acc, self.model, epoch, args.model)
                best_acc = val_acc
                best_epoch = epoch
                epochs_without_improvement = 0  # Reset counter
            else:
                epochs_without_improvement += 1
                
            # Early stopping check
            if epochs_without_improvement >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch} epochs ({early_stopping_patience} epochs without improvement)')
                break

            self.epoch_df = pd.concat([self.train_df, self.val_df], axis=1)
        model = load_model(args.model_save_path, testsub_name, args.model)
        self.model = model
        test_loss, model_test_acc = self.evaluate_step(True)
        self.save_acc_loss_fig(args, testsub_id)
        print("-" * 50)
        print('Test_Subject :{:s} |Best epoch:{:d} | Test Loss:{:2.4f} | Best Acc {:2.4f} | Savemodel Acc {:2.4f}'.format(testsub_name,
                                                                                                    best_epoch,
                                                                                                    test_loss,
                                                                                                    best_acc,
                                                                                                    model_test_acc))
        print("-" * 50)
        return model_test_acc


def ssl_pretrain(args, train_ids, worker_init_fn):
    """SSL Pre-training phase with Transformer decoder"""
    print("=" * 80)
    print("SSL PRE-TRAINING PHASE - TRANSFORMER DECODER")
    print("=" * 80)
    
    # Load SSL data (training subjects only, no labels) with patch-based masking
    ssl_data = getSSLData(args, train_ids, mask_ratio=0.75, use_patch_masking=True)
    ssl_dataset = SSLCustomDatasets(ssl_data)
    ssl_loader = DataLoader(dataset=ssl_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                           worker_init_fn=worker_init_fn, num_workers=4)  # Deterministic multi-worker
    
    # Create SSL model with Transformer decoder
    ssl_model = DARNet_ViT_Decoder(args).to(device)
    ssl_model.set_training_mode('ssl')
    print(f"🏗️  SSL Model Architecture:")
    print(f"   - Model Type: DARNet with Transformer Decoder")
    print(f"   - Training Mode: {getattr(ssl_model, 'training_mode', 'unknown')}")
    print(f"   - Decoder Type: {type(ssl_model.ssl_decoder).__name__}")
    
    # Count parameters in different components
    encoder_params = sum(p.numel() for name, p in ssl_model.named_parameters() 
                       if name.startswith(('token_embedding', 'stack1', 'stack2')))
    decoder_params = sum(p.numel() for name, p in ssl_model.named_parameters() 
                       if name.startswith('ssl_decoder'))
    total_params = sum(p.numel() for p in ssl_model.parameters())
    
    print(f"   - Encoder parameters: {encoder_params:,}")
    print(f"   - Transformer Decoder parameters: {decoder_params:,}")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Decoder/Total ratio: {decoder_params/total_params:.1%}")
    
    # SSL trainer
    ssl_trainer = SSLTrainer(ssl_model, ssl_loader, args.batch_size, args.ssl_lr, args.weight_decay, worker_init_fn)
    
    # Run SSL pre-training
    ssl_model_path = ssl_trainer.ssl_train(args, ssl_epochs=args.ssl_epochs)
    
    return ssl_model_path


def cross_subject(args, sub_ids, train_ids, val_ids, seq_alldata, alllabel, worker_init_fn, ssl_model_path=None):
    """Fine-tuning phase (same as original trainer)"""
    tempt_data, tempt_label = copy.deepcopy(seq_alldata), copy.deepcopy(alllabel)

    # get val data
    val_data, val_label = [], []
    val_index = [sub_ids.index(val_id) for val_id in val_ids]
    for idx in sorted(val_index, reverse=True):
        val_data.append(tempt_data[idx])
        val_label.append(tempt_label[idx])

    # get train data
    train_data = [seq for idx, seq in enumerate(tempt_data) if sub_ids[idx] in train_ids]
    train_label = [lbl for idx, lbl in enumerate(tempt_label) if sub_ids[idx] in train_ids]
    
    train_data = np.concatenate(train_data, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    train_label = train_label.flatten()
    val_data = np.concatenate(val_data, axis=0)
    val_label = np.concatenate(val_label, axis=0)
    val_label = val_label.flatten()

    train_data = np.squeeze(train_data)
    val_data = np.squeeze(val_data)

    train_label = train_label.reshape(-1, 1)
    val_label = val_label.reshape(-1, 1)
    
    print(f"train_data_shape{train_data.shape},val_data_shape{val_data.shape}")

    train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
                              batch_size=args.batch_size, drop_last=True, shuffle=True,
                              worker_init_fn=worker_init_fn, num_workers=4)
    valid_loader = DataLoader(dataset=CustomDatasets(val_data, val_label),
                              batch_size=args.batch_size, drop_last=True,
                              worker_init_fn=worker_init_fn, num_workers=4)
    
    #####################################################################################
    #2.define model
    #####################################################################################
    BaselineNet = Trynetwork(
        model=DARNet_ViT_Decoder(args).to(device),
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=None,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        worker_init_fn=worker_init_fn)
    model_val_acc = BaselineNet.train(args, val_ids, ssl_model_path)
    return model_val_acc


if __name__ == '__main__':
    # Training settings
    args = argparse.ArgumentParser()
    args.seed = 42
    
    # Set all seeds for reproducibility
    print(f"Setting all seeds to {args.seed} for reproducibility...")
    worker_init_fn = set_all_seeds(args.seed)
    print("All seeds set successfully!")

    # data
    args.dataset = 'MM-AAD'
    args.start_time = datetime.now().strftime(f"task1_AAD_{args.dataset}_vit_decoder_%Y-%m-%d-%H-%M-%S")
    print('start time:', args.start_time)
    options = {'MM-AAD': [40, 32, 20, 128, "/home/suyash.kumar.mec22.itbhu/EEG-AAD_audio_only/preprocessed/data/", "/home/suyash.kumar.mec22.itbhu/EEG-AAD_audio_only/preprocessed/label/"]}
    args.subject_number = options[args.dataset][0]
    args.eeg_channel = options[args.dataset][1]
    args.trail_number = options[args.dataset][2]
    args.fs = options[args.dataset][3]
    args.data_path = options[args.dataset][4]
    args.label_path = options[args.dataset][5]

    args.win_time = 1
    args.win_len = math.ceil(args.fs * args.win_time)
    args.overlap = 0.5
    args.window_lap = args.win_len * (1 - args.overlap)

    # basic info of the model
    args.model = "DARNet_SSL_ViT_Decoder"
    args.batch_size = 128
    args.lr = 5e-4
    args.ssl_lr = 1e-3  # Higher learning rate for SSL pre-training
    args.ssl_epochs = 30  # SSL pre-training epochs
    args.lam = 0.2
    args.lr_decayrate = 0.5
    args.weight_decay = 3e-4
    args.max_epoch = 100
    args.patience = 10
    args.log_interval = 10
    
    # Optimizer parameters
    args.optimizer = 'AdamW'  # Can be 'SGD', 'RMSprop', 'Adam', 'AdamW'
    args.momentum = 0.9  # For SGD
    args.alpha = 0.99  # For RMSprop
    args.beta1 = 0.9  # For Adam/AdamW
    args.beta2 = 0.999  # For Adam/AdamW
    
    # save to
    filename = "./exps/cross-subject/%s/" % args.model
    args.model_save_path = f'{filename}baseline_%s/' % args.start_time
    makePath(args.model_save_path)
    args.ssl_model_save_path = f'{filename}ssl_models_%s/' % args.start_time
    makePath(args.ssl_model_save_path)
    args.fig_path = f'{filename}figures/'
    makePath(args.fig_path)

    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

    sub_ids = list(range(1, args.subject_number + 1))
    del_ids = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    sub_ids = [sub_id for sub_id in sub_ids if sub_id not in del_ids]

    # Define sequential folds: [1–5], [6–10], [11–15], [16–20], [21–25], [26–30]
    folds = [
        list(range(1, 6)),
        list(range(6, 11)),
        list(range(11, 16)),
        list(range(16, 21)),
        list(range(21, 26)),
        list(range(26, 31))
    ]

    # Load all subject data once
    seq_alldata, alllabel = getData(args, sub_ids)

    all_fold_acc = []

    for fold, val_ids in enumerate(folds, start=1):
        train_ids = [s for s in sub_ids if s not in val_ids]
        print(f"\n========== Fold {fold} ==========")
        print(f"Train IDs: {train_ids}")
        print(f"Val IDs:   {val_ids}")

        # SSL Pre-training on training subjects only with Transformer decoder
        ssl_model_path = ssl_pretrain(args, train_ids, worker_init_fn)
        
        # Fine-tuning with SSL pre-trained weights
        fold_acc = cross_subject(args, sub_ids, train_ids, val_ids, seq_alldata, alllabel, worker_init_fn, ssl_model_path)
        all_fold_acc.append(fold_acc)

    mean_acc = np.mean(all_fold_acc)

    print(f"lr:{args.lr } -> bs:{args.batch_size}")
    print(f"6-Fold Cross Subject Accuracies: {all_fold_acc}")
    print(f"Mean Accuracy over 6 folds: {mean_acc:.4f}")
    print('=' * 108)
    now1 = datetime.now().strftime("%y-%m-%d-%H:%M:%S")
    print('end time:', now1)