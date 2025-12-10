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

from utils import *
from collections import OrderedDict
from model_module import DARNet, contrastive_loss

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

    # Set CUBLAS environment variable for deterministic CUDA operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Enable deterministic algorithms (may impact performance)
    try:
        torch.use_deterministic_algorithms(True)
    except:
        # Fallback for older PyTorch versions
        try:
            torch.set_deterministic(True)
        except:
            pass  # If neither works, continue without strict determinism

    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set worker seed for DataLoader
    def worker_init_fn(worker_id):
        import random
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)

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

GPU_ID = 0  # Change to 0 or 1 to force specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

device = get_device()
print("Current device is", device)
print(f"Using GPU: {GPU_ID}")

def makePath(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

class StepwiseLR_GRL: 
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.001,
                 gamma: Optional[float] = 0.01, decay_rate: Optional[float] = 0.1,max_iter: Optional[float] = 100):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter=max_iter

    def get_lr(self) -> float:
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num/self.max_iter)) ** (self.decay_rate)
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

class Trynetwork():
    def __init__(self, model, train_loader, valid_loader, test_loader, batch_size, lr, weight_decay, worker_init_fn, args):
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
        self.args = args  # Store args for accessing hyperparameters
        
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

        # Learning rate scheduler: decay every 10 epochs by factor of 0.8
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.8)
        self.scheduler_down = StepwiseLR_GRL(self.optimizer, init_lr= args.lr, gamma= 10, decay_rate=args.lr_decayrate,max_iter=args.max_epoch)
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
        self.model.train()
        train_dicts_per_epoch = OrderedDict()
        Batch_size, Cls_loss, Train_acc, Domain_loss, Contrastive_loss = [], [], [], [], []
        
        for i_batch, batch_data in enumerate(self.datasets['train']):
            if len(batch_data) == 3:  # Has subject IDs
                train_data, train_label, subject_ids = batch_data
                subject_ids = subject_ids.clone().detach().to(device).long()
                has_subject_ids = True
            else:  # Regular training
                train_data, train_label = batch_data
                has_subject_ids = False
                
            train_label = train_label.squeeze(-1)
            train_data, train_label = train_data.to(device).float(), train_label.to(device).long()

            # Forward pass - handle different output combinations
            model_outputs = self.model(train_data)
            
            # Parse outputs based on model configuration
            if not self.model.use_domain_adversarial and not self.model.use_contrastive:
                # Case 1: Original DARNet only
                attention_logits = model_outputs
                total_loss = self.criterion(attention_logits, train_label)
                attention_loss = total_loss
                
            elif self.model.use_domain_adversarial and not self.model.use_contrastive:
                # Case 2: Domain adversarial only
                if isinstance(model_outputs, tuple):
                    attention_logits, subject_logits = model_outputs
                    attention_loss = self.criterion(attention_logits, train_label)
                    if has_subject_ids:
                        subject_loss = self.criterion(subject_logits, subject_ids)
                        total_loss = attention_loss + subject_loss
                        Domain_loss.append(subject_loss.cpu().detach().numpy())
                    else:
                        total_loss = attention_loss
                else:
                    attention_logits = model_outputs
                    total_loss = self.criterion(attention_logits, train_label)
                    attention_loss = total_loss
                    
            elif not self.model.use_domain_adversarial and self.model.use_contrastive:
                # Case 3: Contrastive only
                if isinstance(model_outputs, tuple):
                    attention_logits, contrastive_features = model_outputs
                    attention_loss = self.criterion(attention_logits, train_label)
                    contra_loss = contrastive_loss(contrastive_features, train_label, temperature=self.args.temperature)
                    total_loss = attention_loss + self.args.lambda_contrastive * contra_loss
                    Contrastive_loss.append(contra_loss.cpu().detach().numpy())
                else:
                    attention_logits = model_outputs
                    total_loss = self.criterion(attention_logits, train_label)
                    attention_loss = total_loss
                    
            else:
                # Case 4: Both domain adversarial and contrastive
                if isinstance(model_outputs, tuple) and len(model_outputs) == 3:
                    attention_logits, subject_logits, contrastive_features = model_outputs
                    attention_loss = self.criterion(attention_logits, train_label)
                    
                    # Domain adversarial loss
                    if has_subject_ids:
                        subject_loss = self.criterion(subject_logits, subject_ids)
                        Domain_loss.append(subject_loss.cpu().detach().numpy())
                    else:
                        subject_loss = 0
                    
                    # Contrastive loss
                    contra_loss = contrastive_loss(contrastive_features, train_label, temperature=self.args.temperature)
                    Contrastive_loss.append(contra_loss.cpu().detach().numpy())
                    
                    total_loss = attention_loss + subject_loss + self.args.lambda_contrastive * contra_loss
                else:
                    attention_logits = model_outputs
                    total_loss = self.criterion(attention_logits, train_label)
                    attention_loss = total_loss

            Batch_size.append(len(train_label))
            _, predicted = torch.max(attention_logits.data, 1)
            batch_acc = np.equal(predicted.cpu().detach().numpy(), train_label.cpu().detach().numpy()).sum() / len(train_label)
            
            Train_acc.append(batch_acc)
            Cls_loss.append(attention_loss.cpu().detach().numpy())

            # Backward and optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        epoch_acc = sum(Train_acc) / len(Train_acc) * 100
        epoch_loss = sum(Cls_loss) / len(Cls_loss)

        cls_loss = {'train_loss': epoch_loss}
        train_acc = {'train_acc': epoch_acc}
        train_dicts_per_epoch.update(cls_loss)
        train_dicts_per_epoch.update(train_acc)
        
        if Domain_loss:
            domain_loss = {'domain_loss': sum(Domain_loss) / len(Domain_loss)}
            train_dicts_per_epoch.update(domain_loss)
            
        if Contrastive_loss:
            contrastive_loss_avg = {'contrastive_loss': sum(Contrastive_loss) / len(Contrastive_loss)}
            train_dicts_per_epoch.update(contrastive_loss_avg)
        
        train_dicts_per_epoch = {k: [v] for k, v in train_dicts_per_epoch.items()}
        self.train_df = pd.concat([self.train_df, pd.DataFrame(train_dicts_per_epoch)], ignore_index=True)
        self.train_df = self.train_df[list(train_dicts_per_epoch.keys())]  
        return epoch_loss, epoch_acc
    
    def test_batch(self, input, label):
        self.model.eval()
        with torch.no_grad():
            val_input = input.to(device).float()
            val_label = label.to(device).long()
            
            # Model returns only attention logits during evaluation
            model_output = self.model(val_input)
            if isinstance(model_output, tuple):
                val_fc1 = model_output[0]  # Get attention logits
            else:
                val_fc1 = model_output
                
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
            all_preds = []  # Store all predictions for confusion matrix
            all_targets = []  # Store all targets for confusion matrix
            for i_batch, batch_data in enumerate(self.datasets[setname]):
                if len(batch_data) == 3:  # Has subject IDs
                    seq_input, target, subject_ids = batch_data
                else:  # Regular training
                    seq_input, target = batch_data

                target = target.squeeze(-1)
                pred, loss = self.test_batch(seq_input,  target)
                Epochs_loss.append(loss)
                Batch_size.append(len(target))
                Epochs_acc.append(np.equal(pred, target.numpy()).sum())
                all_preds.extend(pred.tolist())
                all_targets.extend(target.numpy().tolist())

        epoch_acc = sum(Epochs_acc) / sum(Batch_size) * 100
        epoch_loss = sum(Epochs_loss) / len(Epochs_loss)

        # Compute and store confusion matrix for validation
        if setname == 'valid':
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(all_targets, all_preds)
            # Store confusion matrix for potential display
            if not hasattr(self, 'last_confusion_matrix'):
                self.last_confusion_matrix = cm
            else:
                self.last_confusion_matrix = cm

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


    def train(self, args, testsub_id):
        torchinfo.summary(self.model)
        testsub_name = 'S' + str(testsub_id)
        
        best_epoch = 0
        best_acc = 0
        early_stopping_patience = 20
        epochs_without_improvement = 0
        for epoch in range(1, args.max_epoch + 1):
            # Update lambda for domain adversarial training (gradual increase)
            if hasattr(self.model, 'use_domain_adversarial') and self.model.use_domain_adversarial:
                progress = epoch / args.max_epoch
                lambda_p = 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0
                lambda_p = max(0.0, lambda_p) * args.lambda_domain
                self.model.update_lambda(lambda_p)
                
            train_loss, train_acc = self.train_step()
            val_loss, val_acc = self.evaluate_step(False)
            
            # Print training progress based on enabled components
            print_str = f'TestSub: {testsub_name} Epoch {epoch:2d} Finsh | Now_lr {self.optimizer.param_groups[0]["lr"]:.4f}/{args.lr:.4f} | Train Loss {train_loss:.4f} | Valid Loss {val_loss:.4f} | Train Acc {train_acc:.4f}| Valid Acc {val_acc:.4f}'
            
            if hasattr(self.model, 'use_domain_adversarial') and self.model.use_domain_adversarial:
                domain_loss_val = self.train_df['domain_loss'].iloc[-1] if 'domain_loss' in self.train_df.columns else 0.0
                print_str += f' | Domain Loss {domain_loss_val:.4f} | Lambda {lambda_p:.4f}'
                
            if hasattr(self.model, 'use_contrastive') and self.model.use_contrastive:
                contrastive_loss_val = self.train_df['contrastive_loss'].iloc[-1] if 'contrastive_loss' in self.train_df.columns else 0.0
                print_str += f' | Contrastive Loss {contrastive_loss_val:.4f}'

            print(print_str)

            # Print confusion matrix every 5 epochs to check for class collapse
            if epoch % 5 == 0 and hasattr(self, 'last_confusion_matrix'):
                print(f'    Confusion Matrix (Epoch {epoch}):')
                print(f'    [[TN={self.last_confusion_matrix[0,0]}, FP={self.last_confusion_matrix[0,1]}],')
                print(f'     [FN={self.last_confusion_matrix[1,0]}, TP={self.last_confusion_matrix[1,1]}]]')
                # Check if model is predicting only one class
                if self.last_confusion_matrix[0,1] == 0 and self.last_confusion_matrix[1,1] == 0:
                    print(f'    ⚠️  WARNING: Model predicting only class 0! (Class collapse detected)')
                elif self.last_confusion_matrix[0,0] == 0 and self.last_confusion_matrix[1,0] == 0:
                    print(f'    ⚠️  WARNING: Model predicting only class 1! (Class collapse detected)')

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
    


def cross_subject(args, sub_ids, train_ids, val_ids, seq_alldata, alllabel, worker_init_fn):
    tempt_data, tempt_label = copy.deepcopy(seq_alldata), copy.deepcopy(alllabel)

    # get val data
    val_data, val_label = [], []
    val_index = [sub_ids.index(val_id) for val_id in val_ids]
    for idx in sorted(val_index, reverse=True):
        val_data.append(tempt_data[idx])
        val_label.append(tempt_label[idx])

    # get train data and create subject ID arrays
    train_data = []
    train_label = []
    train_subject_ids = []
    
    for idx, seq in enumerate(tempt_data):
        if sub_ids[idx] in train_ids:
            train_data.append(seq)
            train_label.append(tempt_label[idx])
            # Create subject ID array for each trial (0-based indexing for discriminator)
            subject_id = sub_ids[idx] - 1  # Convert to 0-based
            train_subject_ids.extend([subject_id] * len(seq))
    
    val_subject_ids = []
    for idx in val_index:
        subject_id = sub_ids[idx] - 1  # Convert to 0-based
        val_subject_ids.extend([subject_id] * len(tempt_data[idx]))
    
    train_data = np.concatenate(train_data, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    train_label = train_label.flatten()
    val_data = np.concatenate(val_data, axis=0)
    val_label = np.concatenate(val_label, axis=0)
    val_label = val_label.flatten()

    train_data = np.squeeze(train_data)
    val_data = np.squeeze(val_data)

    train_label = train_label.reshape(-1,1)
    val_label = val_label.reshape(-1,1)
    
    print(f"train_data_shape{train_data.shape},val_data_shape{val_data.shape}")

    # Create datasets with subject IDs if using domain adversarial training
    use_domain_adversarial = getattr(args, 'use_domain_adversarial', False)

    # Create generator for reproducible DataLoader
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Balanced sampling setup
    train_sampler = None
    if args.balanced_sampling:
        # Calculate class weights for balanced sampling
        class_counts = np.bincount(train_label.flatten().astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_label.flatten().astype(int)]

        from torch.utils.data import WeightedRandomSampler
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=g  # Add generator for reproducibility
        )

        print(f"\n  ⚖️ Balanced Sampling Enabled:")
        print(f"     Class 0 (left): {class_counts[0]} samples, weight: {class_weights[0]:.4f}")
        print(f"     Class 1 (right): {class_counts[1]} samples, weight: {class_weights[1]:.4f}")

    if use_domain_adversarial:
        train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label, train_subject_ids),
                                      batch_size=args.batch_size, drop_last=True,
                                      shuffle=(train_sampler is None),  # Don't shuffle if using sampler
                                      sampler=train_sampler,
                                      worker_init_fn=worker_init_fn, num_workers=0,
                                      generator=g)  # Add generator for reproducibility
        valid_loader = DataLoader(dataset=CustomDatasets(val_data, val_label, val_subject_ids),
                                      batch_size=args.batch_size, drop_last=True,
                                      worker_init_fn=worker_init_fn, num_workers=0,
                                      generator=g)  # Add generator for reproducibility
    else:
        train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
                                      batch_size=args.batch_size, drop_last=True,
                                      shuffle=(train_sampler is None),  # Don't shuffle if using sampler
                                      sampler=train_sampler,
                                      worker_init_fn=worker_init_fn, num_workers=0,
                                      generator=g)  # Add generator for reproducibility
        valid_loader = DataLoader(dataset=CustomDatasets(val_data, val_label),
                                      batch_size=args.batch_size, drop_last=True,
                                      worker_init_fn=worker_init_fn, num_workers=0,
                                      generator=g)  # Add generator for reproducibility
    
    #####################################################################################
    #2.define model
    #####################################################################################
    BaselineNet = Trynetwork(
        model = DARNet(args).to(device),
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        test_loader=None,
        batch_size = args.batch_size, 
        lr = args.lr,
        weight_decay = args.weight_decay,
        worker_init_fn = worker_init_fn,
        args = args)
    model_val_acc = BaselineNet.train(args, val_ids)
    return model_val_acc

    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='EEG Auditory Attention Detection Training')
    parser.add_argument('--dataset', type=str, default='MM-AAD',
                        choices=['MM-AAD', 'KUL', 'DTU', 'AVED-audio', 'AVED-audiovisual'],
                        help='Dataset to use: MM-AAD, KUL, DTU, AVED-audio, or AVED-audiovisual (default: MM-AAD)')
    parser.add_argument('--win_time', type=int, default=None,
                        help='Window time in seconds (default: auto - 1s for AVED/DTU, 2s for others)')

    # Preprocessing arguments
    parser.add_argument('--apply_csp', action='store_true',
                        help='Apply CSP (Common Spatial Patterns) transformation')
    parser.add_argument('--apply_ea', action='store_true',
                        help='Apply Euclidean Alignment preprocessing')
    parser.add_argument('--apply_normalize', action='store_true',
                        help='Apply normalization preprocessing')

    parsed_args = parser.parse_args()

    # Create args namespace
    args = argparse.Namespace()
    args.seed = 42  # Hardcoded seed for reproducibility
    args.dataset = parsed_args.dataset
    # Auto-enable balanced sampling ONLY for DTU (due to severe class imbalance)
    args.balanced_sampling = (args.dataset == 'DTU')
    args.win_time_override = parsed_args.win_time  # Store user override

    # Set all seeds for reproducibility
    print(f"Setting all seeds to {args.seed} for reproducibility...")
    worker_init_fn = set_all_seeds(args.seed)
    print("All seeds set successfully!")

    # Print balanced sampling status
    if args.balanced_sampling:
        print(f"\n⚖️  Balanced Sampling: ENABLED (auto-enabled for {args.dataset} due to class imbalance)")
    else:
        print(f"\n⚖️  Balanced Sampling: DISABLED ({args.dataset} has balanced classes)")

    # Dataset configuration
    args.start_time = datetime.now().strftime(f"task1_AAD_{args.dataset}_%Y-%m-%d-%H-%M-%S")
    print('start time:',args.start_time)

    options = {
        'MM-AAD': [40, 32, 20, 128,
                   "/home/suyash.kumar.mec22.itbhu/EEG-AAD_audio_visual/preprocessed/data/",
                   "/home/suyash.kumar.mec22.itbhu/EEG-AAD_audio_visual/preprocessed/label/"],
        'KUL': [16, 64, 8, 128,
                "/home/suyash.kumar.mec22.itbhu/KUL/",
                "/home/suyash.kumar.mec22.itbhu/KUL"],
        'DTU': [18, 64, 60, 512,
                "/home/suyash.kumar.mec22.itbhu/DTU/",
                "/home/suyash.kumar.mec22.itbhu/DTU/"],
        'AVED-audio': [10, 32, 16, 128,
                       "/scratch/suyash.kumar.mec22.itbhu/Aved/audio/",
                       "/scratch/suyash.kumar.mec22.itbhu/Aved/audio/"],
        'AVED-audiovisual': [10, 32, 16, 128,
                             "/scratch/suyash.kumar.mec22.itbhu/Aved/audio-visual/",
                             "/scratch/suyash.kumar.mec22.itbhu/Aved/audio-visual/"]
    }

    args.subject_number = options[args.dataset][0]
    args.eeg_channel = options[args.dataset][1]
    args.trail_number = options[args.dataset][2]
    args.fs  = options[args.dataset][3]
    args.data_path = options[args.dataset][4]
    args.label_path = options[args.dataset][5]

    # Window configuration - allow user override or use dataset default
    if args.win_time_override is not None:
        args.win_time = args.win_time_override  # User specified window time
    elif args.dataset.startswith('AVED'):
        args.win_time = 1  # AVED default: 1-second windows
    else:
        args.win_time = 2  # Default: 2-second windows

    args.win_len = math.ceil(args.fs * args.win_time)
    args.overlap = 0.5
    args.window_lap = args.win_len * (1 - args.overlap)

    # ====================================================================
    # OPTIMIZED HYPERPARAMETERS FOR CROSS-SUBJECT AAD
    # Based on: DARNet, ListenNet best practices + domain adversarial + contrastive learning
    # ====================================================================

    # Loss Weights (tuned for cross-subject AAD)
    args.lambda_domain = 0.3            # INCREASED from 0.15 (stronger domain adaptation)
    args.lambda_contrastive = 0.2       # REDUCED from 0.5 (was too dominant)
    args.temperature = 0.15             # INCREASED from 0.1 (softer contrastive learning)

    # Optimizer Configuration
    args.optimizer = 'Adam'             # Adam generally better than SGD for attention models
    args.lr = 1e-4                      # REDUCED from 5e-4 (slower, more stable learning)
    args.weight_decay = 5e-4            # INCREASED from 1e-4 (stronger regularization)
    args.momentum = 0.9                 # For SGD (not used with Adam)

    # Training Configuration
    args.batch_size = 32                # Increased from 16 (better gradient estimates)
    args.max_epoch = 100                # Standard training epochs (early stopping will catch convergence)
    args.patience = 20                  # Early stopping patience (reasonable for 100 epochs)
    args.log_interval = 10

    # Learning Rate Scheduler
    args.lam = 0.2                      # Not used currently
    args.lr_decayrate = 0.5             # LR decay rate
    
    # Model Configuration - Set these flags to control what gets used
    args.use_domain_adversarial = True   # Set to False to disable domain adversarial
    args.use_contrastive = True          # Set to False to disable contrastive learning

    # Preprocessing Configuration - Set these flags to control preprocessing
    # You can change these defaults here, OR use command-line args to override:
    #   --apply_csp          : Enable CSP transformation
    #   --apply_ea           : Enable Euclidean Alignment
    #   --apply_normalize    : Enable normalization

    # Hardcoded defaults (change these to True/False as needed):
    DEFAULT_CSP = False          # CSP: Not recommended for cross-subject (no dimension reduction with 64→64)
    DEFAULT_EA = True            # EA: Recommended (aligns spatial covariances across subjects)
    DEFAULT_NORMALIZE = True     # Normalization: Recommended (scales to [-1,1])

    # Apply defaults, but allow command-line to override if flags are provided
    args.apply_csp = parsed_args.apply_csp if parsed_args.apply_csp else DEFAULT_CSP
    args.apply_ea = parsed_args.apply_ea if parsed_args.apply_ea else DEFAULT_EA
    args.apply_normalize = parsed_args.apply_normalize if parsed_args.apply_normalize else DEFAULT_NORMALIZE

    # Print preprocessing configuration
    preprocessing_enabled = []
    if args.apply_csp:
        preprocessing_enabled.append("CSP")
    if args.apply_ea:
        preprocessing_enabled.append("EA")
    if args.apply_normalize:
        preprocessing_enabled.append("Normalization")

    if preprocessing_enabled:
        print(f"\n🔧 Preprocessing: {' + '.join(preprocessing_enabled)}")
    else:
        print(f"\n🔧 Preprocessing: NONE (raw data)")

    # Loss weights
    # args.lambda_domain = 0.15       # Domain adversarial weight
    # args.lambda_contrastive = 0.5   # Contrastive learning weight
    # args.temperature = 0.1          # InfoNCE temperature parameter
    
    # Dynamic model naming based on enabled components
    model_components = ["DARNet"]
    if args.use_domain_adversarial:
        model_components.append("Domain")
    if args.use_contrastive:
        model_components.append("Contra")
    args.model = "_".join(model_components)
    
    # save to 
    filename = "./exps/cross-subject/%s/" % args.model
    args.model_save_path = f'{filename}baseline_%s/' % args.start_time
    makePath(args.model_save_path)
    args.fig_path = f'{filename}figures/' 
    makePath(args.fig_path)

    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)
   
    # Subject IDs and fold configuration
    # ALL DATASETS NOW USE LOSO (Leave-One-Subject-Out)
    if args.dataset == 'MM-AAD':
        sub_ids = list(range(1, args.subject_number+1))
        del_ids = [31,32,33,34,35,36,37,38,39,40]
        sub_ids = [sub_id for sub_id in sub_ids if sub_id not in del_ids]

        # LOSO: Each subject is validation once (30 subjects)
        folds = [[i] for i in sub_ids]  # [[1], [2], ..., [30]]
        num_folds = len(sub_ids)  # 30 folds

    elif args.dataset == 'KUL':
        sub_ids = list(range(1, args.subject_number+1))  # 1-16

        # LOSO: Each subject is validation once (16 subjects)
        folds = [[i] for i in sub_ids]  # [[1], [2], ..., [16]]
        num_folds = args.subject_number  # 16 folds

    elif args.dataset == 'DTU':
        sub_ids = list(range(1, args.subject_number+1))  # 1-18

        # LOSO: Each subject is validation once (18 subjects)
        folds = [[i] for i in sub_ids]  # [[1], [2], [3], ..., [18]]
        num_folds = args.subject_number  # 18 folds

    elif args.dataset.startswith('AVED'):
        sub_ids = list(range(1, args.subject_number+1))  # 1-10

        # LOSO: Each subject is validation once (10 subjects)
        folds = [[i] for i in sub_ids]  # [[1], [2], [3], ..., [10]]
        num_folds = args.subject_number  # 10 folds

    # Print preprocessing configuration
    print(f"\n{'='*80}")
    print(f"Preprocessing Configuration:")
    print(f"  CSP:           {args.apply_csp}")
    print(f"  EA:            {args.apply_ea}")
    print(f"  Normalization: {args.apply_normalize}")
    print(f"{'='*80}\n")

    # Load all subject data once
    if args.dataset == 'DTU':
        # For DTU, we need to add window_length for getData_DTU
        args.window_length = math.ceil(args.fs * args.win_time)
        seq_alldata, alllabel = getData_DTU(args, sub_ids,
                                             apply_csp=args.apply_csp,
                                             apply_ea=args.apply_ea,
                                             apply_normalize=args.apply_normalize)
    elif args.dataset == 'KUL':
        # For KUL, we need to add window_length for getData_KUL
        args.window_length = math.ceil(args.fs * args.win_time)
        seq_alldata, alllabel = getData_KUL(args, sub_ids,
                                             apply_csp=args.apply_csp,
                                             apply_ea=args.apply_ea,
                                             apply_normalize=args.apply_normalize)
    elif args.dataset.startswith('AVED'):
        # For AVED, we need to add window_length for getData_AVED
        args.window_length = math.ceil(args.fs * args.win_time)
        seq_alldata, alllabel = getData_AVED(args, sub_ids,
                                              apply_csp=args.apply_csp,
                                              apply_ea=args.apply_ea,
                                              apply_normalize=args.apply_normalize)
    else:
        seq_alldata, alllabel = getData(args, sub_ids,
                                         apply_csp=args.apply_csp,
                                         apply_ea=args.apply_ea,
                                         apply_normalize=args.apply_normalize)

    all_fold_acc = []

    for fold, val_ids in enumerate(folds, start=1):
        train_ids = [s for s in sub_ids if s not in val_ids]
        print(f"\n========== Fold {fold}/{num_folds} ==========")
        print(f"Train IDs: {train_ids}")
        print(f"Val IDs:   {val_ids}")

        fold_acc = cross_subject(args, sub_ids, train_ids, val_ids, seq_alldata, alllabel, worker_init_fn)
        all_fold_acc.append(fold_acc)

    mean_acc = np.mean(all_fold_acc)

    print(f"\n{'='*108}")
    print(f"FINAL RESULTS - {args.dataset} Dataset (LOSO)")
    print(f"{'='*108}")
    print(f"lr: {args.lr} | batch_size: {args.batch_size}")
    print(f"LOSO Accuracies ({num_folds} subjects): {all_fold_acc}")
    print(f"Mean Accuracy over {num_folds} subjects: {mean_acc:.4f}")
    print('=' * 108)
    now1 = datetime.now().strftime("%y-%m-%d-%H:%M:%S")
    print('end time:', now1)
# from __future__ import division
# from __future__ import print_function
# import os
# import math

# import numpy as np
# import pandas as pd

# from datetime import datetime
# import argparse
# import copy

# import torch
# from torch.utils.data import DataLoader
# import torch.nn as nn

# from torch.optim.optimizer import Optimizer
# from typing import Optional
# import torchinfo
# import matplotlib.pyplot as plt

# from utils import *
# from collections import OrderedDict
# from model_module import DARNet

# def set_all_seeds(seed):
#     """Set all possible seeds for reproducibility"""
#     import random
#     import os
    
#     # Python random seed
#     random.seed(seed)
    
#     # Numpy seed
#     np.random.seed(seed)
    
#     # PyTorch seeds
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
#     # PyTorch deterministic behavior
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
#     # Python hash seed
#     os.environ['PYTHONHASHSEED'] = str(seed)
    
#     # Set worker seed for DataLoader
#     def worker_init_fn(worker_id):
#         np.random.seed(seed + worker_id)
    
#     return worker_init_fn

# def get_device():
#     """Get the best available device: CUDA, MPS (Apple Silicon), or CPU"""
#     if torch.cuda.is_available():
#         return torch.device("cuda")
#     elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
#         return torch.device("mps")
#     else:
#         return torch.device("cpu")

# if "SLURM_JOB_GPUS" in os.environ:
#     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_JOB_GPUS"]

# device = get_device()
# print("Current device is", device)

# def makePath(path):
#     if not os.path.isdir(path):
#         os.makedirs(path)
#     return path

# class StepwiseLR_GRL: 
#     def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.001,
#                  gamma: Optional[float] = 0.01, decay_rate: Optional[float] = 0.1,max_iter: Optional[float] = 100):
#         self.init_lr = init_lr
#         self.gamma = gamma
#         self.decay_rate = decay_rate
#         self.optimizer = optimizer
#         self.iter_num = 0
#         self.max_iter=max_iter

#     def get_lr(self) -> float:
#         lr = self.init_lr / (1.0 + self.gamma * (self.iter_num/self.max_iter)) ** (self.decay_rate)
#         if lr <= 1e-8:
#             lr = 1e-8
#         return lr

#     def step(self):
#         """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
#         lr = self.get_lr()
#         for param_group in self.optimizer.param_groups:
#             if 'lr_mult' not in param_group:
#                 param_group['lr_mult'] = 1.
#             param_group['lr'] = lr * param_group['lr_mult']
#         self.iter_num += 1

# class Trynetwork():
#     def __init__(self, model, train_loader, valid_loader, test_loader, batch_size, lr, weight_decay, worker_init_fn):
#         self.model = model
#         self.datasets = OrderedDict((("train", train_loader), ("valid", valid_loader), ("test", test_loader)))
#         if valid_loader is None:
#             self.datasets.pop("valid")
#         if test_loader is None:
#             self.datasets.pop("test")
#         self.best_test = 0
#         self.batch_size = batch_size
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.worker_init_fn = worker_init_fn
        
#         # Create optimizer based on args
#         if args.optimizer == 'SGD':
#             self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr, 
#                                            momentum=getattr(args, 'momentum', 0.9), 
#                                            weight_decay=self.weight_decay)
#         elif args.optimizer == 'RMSprop':
#             self.optimizer = torch.optim.RMSprop(params=self.model.parameters(), lr=self.lr,
#                                                alpha=getattr(args, 'alpha', 0.99),
#                                                weight_decay=self.weight_decay)
#         elif args.optimizer == 'Adam':
#             self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr,
#                                             betas=(getattr(args, 'beta1', 0.9), getattr(args, 'beta2', 0.999)),
#                                             weight_decay=self.weight_decay)
#         else:  # Default to AdamW
#             self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr,
#                                              betas=(getattr(args, 'beta1', 0.9), getattr(args, 'beta2', 0.999)),
#                                              weight_decay=self.weight_decay)

#         self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 35], gamma=0.5)
#         self.scheduler_down = StepwiseLR_GRL(self.optimizer, init_lr= args.lr, gamma= 10, decay_rate=args.lr_decayrate,max_iter=args.max_epoch)
#         self.criterion = nn.CrossEntropyLoss()
        
#         # initialize epoch dataFrame instead of loss and acc for train and test
#         self.val_df = pd.DataFrame()  
#         self.train_df = pd.DataFrame()  
#         self.epoch_df = pd.DataFrame()  
        

#     def __getModel__(self):
#         return self.model

#     def save_acc_loss_fig(self, args, sub_id):

#         valid_acc = self.epoch_df['valid_acc'].values.tolist()
#         valid_loss = self.epoch_df['valid_loss'].values.tolist()
#         train_acc = self.epoch_df['train_acc'].values.tolist()
#         train_loss = self.epoch_df['train_loss'].values.tolist()

#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

#         # First subgraph: Accuracy and loss of training data
#         ax1.plot(range(len(train_acc)), train_acc, label='Train Accuracy', color='blue', linewidth=0.7)
#         ax1.plot(range(len(valid_acc)), valid_acc, label='Valid Accuracy', color='red', linewidth=0.7)
#         ax1.set_title('Acc Performance')
#         ax1.set_xlabel('Epoch')
#         ax1.set_ylabel('Accuracy')
#         ax1.legend(loc='upper right')

#         # Second subgraph: Accuracy and loss of test data
#         ax2.plot(range(len(train_loss)), train_loss, label='Train Loss', color='green', linewidth=0.7)
#         ax2.plot(range(len(valid_loss)), valid_loss, label='Valid Loss', color='purple', linewidth=0.7)
#         ax2.set_title('Loss Performance')
#         ax2.set_xlabel('Epoch')
#         ax2.set_ylabel('Loss')
#         ax2.legend(loc='upper right')

#         plt.tight_layout()  
#         plt.savefig(os.path.join(args.fig_path, 'Loss_Acc.png'))

#     def train_step(self):
#         self.model.train()
#         train_dicts_per_epoch = OrderedDict()
#         Batch_size, Cls_loss, Train_acc, Domain_loss = [], [], [], []
        
#         for i_batch, batch_data in enumerate(self.datasets['train']):
#             if len(batch_data) == 3:  # Domain adversarial training
#                 train_data, train_label, subject_ids = batch_data
#                 subject_ids = subject_ids.squeeze(-1).to(device).long()
#                 use_domain_adversarial = True
#             else:  # Regular training
#                 train_data, train_label = batch_data
#                 use_domain_adversarial = False
                
#             train_label = train_label.squeeze(-1)
#             train_data, train_label = train_data.to(device).float(), train_label.to(device).long()

#             # Forward pass
#             if use_domain_adversarial and self.model.use_domain_adversarial:
#                 attention_logits, subject_logits = self.model(train_data)
                
#                 # Attention loss (main task)
#                 attention_loss = self.criterion(attention_logits, train_label)
                
#                 # Subject loss (domain adversarial)
#                 subject_loss = self.criterion(subject_logits, subject_ids)
                
#                 # Combined loss
#                 total_loss = attention_loss + subject_loss  # Lambda is handled in gradient reversal
                
#                 Domain_loss.append(subject_loss.cpu().detach().numpy())
#             else:
#                 attention_logits = self.model(train_data)
#                 total_loss = self.criterion(attention_logits, train_label)
#                 attention_loss = total_loss

#             Batch_size.append(len(train_label))
#             _, predicted = torch.max(attention_logits.data, 1)
#             batch_acc = np.equal(predicted.cpu().detach().numpy(), train_label.cpu().detach().numpy()).sum() / len(train_label)
            
#             Train_acc.append(batch_acc)
#             Cls_loss.append(attention_loss.cpu().detach().numpy())

#             # Backward and optimize
#             self.optimizer.zero_grad()
#             total_loss.backward()
#             self.optimizer.step()

#         epoch_acc = sum(Train_acc) / len(Train_acc) * 100
#         epoch_loss = sum(Cls_loss) / len(Cls_loss)

#         cls_loss = {'train_loss': epoch_loss}
#         train_acc = {'train_acc': epoch_acc}
#         train_dicts_per_epoch.update(cls_loss)
#         train_dicts_per_epoch.update(train_acc)
        
#         if Domain_loss:
#             domain_loss = {'domain_loss': sum(Domain_loss) / len(Domain_loss)}
#             train_dicts_per_epoch.update(domain_loss)
        
#         train_dicts_per_epoch = {k: [v] for k, v in train_dicts_per_epoch.items()}
#         self.train_df = pd.concat([self.train_df, pd.DataFrame(train_dicts_per_epoch)], ignore_index=True)
#         self.train_df = self.train_df[list(train_dicts_per_epoch.keys())]  
#         return epoch_loss, epoch_acc
    
#     def test_batch(self, input, label):
#         self.model.eval()
#         with torch.no_grad():
#             val_input = input.to(device).float()
#             val_label = label.to(device).long()
            
#             # Model returns only attention logits during evaluation
#             model_output = self.model(val_input)
#             if isinstance(model_output, tuple):
#                 val_fc1 = model_output[0]  # Get attention logits
#             else:
#                 val_fc1 = model_output
                
#             loss = self.criterion(val_fc1, val_label)
#             _, preds = torch.max(val_fc1.data, 1)  
#             preds = preds.cpu().detach().numpy()
#             loss = loss.cpu().detach().numpy()
#         return preds, loss

#     def evaluate_step(self, Flag_test):
#         if Flag_test and "test" in self.datasets:
#             setname = 'test'
#         else:
#             setname = 'valid'
#         result_dicts_per_monitor = OrderedDict()  
#         with torch.no_grad():
#             Batch_size, Epochs_loss, Epochs_acc = [], [], []
#             for i_batch, batch_data in enumerate(self.datasets[setname]):
#                 if len(batch_data) == 3:  # Domain adversarial training
#                     seq_input, target, subject_ids = batch_data
#                 else:  # Regular training
#                     seq_input, target = batch_data
                    
#                 target = target.squeeze(-1)
#                 pred, loss = self.test_batch(seq_input,  target)  
#                 Epochs_loss.append(loss)
#                 Batch_size.append(len(target))
#                 Epochs_acc.append(np.equal(pred, target.numpy()).sum())  
#         epoch_acc = sum(Epochs_acc) / sum(Batch_size) * 100
#         epoch_loss = sum(Epochs_loss) / len(Epochs_loss)
#         key_loss = setname + '_loss'
#         key_acc = setname + '_acc'
#         loss = {key_loss: epoch_loss}
#         acc = {key_acc: epoch_acc}
#         result_dicts_per_monitor.update(loss)
#         result_dicts_per_monitor.update(acc)
#         result_dicts_per_monitor = {k: [v] for k, v in result_dicts_per_monitor.items()}
#         self.val_df = pd.concat([self.val_df, pd.DataFrame(result_dicts_per_monitor)], ignore_index=True)
#         self.val_df = self.val_df[list(result_dicts_per_monitor.keys())]  
#         return epoch_loss, epoch_acc


#     def train(self, args, testsub_id):
#         torchinfo.summary(self.model)
#         testsub_name = 'S' + str(testsub_id)
        
#         best_epoch = 0
#         best_acc = 0
#         early_stopping_patience = 30
#         epochs_without_improvement = 0
#         for epoch in range(1, args.max_epoch + 1):
#             # Update lambda for domain adversarial training (gradual increase)
#             if hasattr(self.model, 'use_domain_adversarial') and self.model.use_domain_adversarial:
#                 progress = epoch / args.max_epoch
#                 lambda_p = 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0
#                 lambda_p = max(0.0, lambda_p) * args.lambda_domain
#                 self.model.update_lambda(lambda_p)
                
#             train_loss, train_acc = self.train_step()
#             val_loss, val_acc = self.evaluate_step(False)
            
#             if hasattr(self.model, 'use_domain_adversarial') and self.model.use_domain_adversarial:
#                 # Get domain loss from the latest training step
#                 domain_loss_val = self.train_df['domain_loss'].iloc[-1] if 'domain_loss' in self.train_df.columns else 0.0

#                 print('TestSub:', testsub_name,
#                 'Epoch {:2d} Finsh | Now_lr {:2.4f}/{:2.4f} | Lambda {:2.4f} | Train Loss {:2.4f} | Domain Loss {:2.4f} | Valid Loss {:2.4f} | Train Acc {:5.4f}| Valid Acc {:5.4f}'.format(epoch,
#                                                    self.optimizer.param_groups[0]["lr"], args.lr,
#                                                    lambda_p,
#                                                    train_loss,
#                                                    domain_loss_val,
#                                                    val_loss,
#                                                    train_acc,
#                                                    val_acc))
#             else:
#                 print('TestSub:', testsub_name,
#                       'Epoch {:2d} Finsh | Now_lr {:2.4f}/{:2.4f}|Train Loss {:2.4f} | Valid Loss {:2.4f} | Train Acc {:5.4f}| Valid Acc {:5.4f}'.format(epoch,
#                                                                                                                                                     self.optimizer.param_groups[0]["lr"], args.lr,
#                                                                                                                                                     train_loss,
#                                                                                                                                                     val_loss,
#                                                                                                                                                     train_acc,
#                                                                                                                                                     val_acc))
#             if val_acc > best_acc:
#                 save_model(args, testsub_name, best_acc, val_acc, self.model, epoch, args.model)
#                 best_acc = val_acc
#                 best_epoch = epoch
#                 epochs_without_improvement = 0  # Reset counter
#             else:
#                 epochs_without_improvement += 1
                
#             # Early stopping check
#             if epochs_without_improvement >= early_stopping_patience:
#                 print(f'Early stopping triggered after {epoch} epochs ({early_stopping_patience} epochs without improvement)')
#                 break
    
#             self.epoch_df = pd.concat([self.train_df, self.val_df], axis=1)
#         model = load_model(args.model_save_path, testsub_name, args.model)
#         self.model = model
#         test_loss, model_test_acc = self.evaluate_step(True)
#         self.save_acc_loss_fig(args, testsub_id)
#         print("-" * 50)
#         print('Test_Subject :{:s} |Best epoch:{:d} | Test Loss:{:2.4f} | Best Acc {:2.4f} | Savemodel Acc {:2.4f}'.format(testsub_name,
#                                                                                                     best_epoch,
#                                                                                                     test_loss,
#                                                                                                     best_acc,
#                                                                                                     model_test_acc))
#         print("-" * 50)
#         return model_test_acc
    


# def cross_subject(args, sub_ids, train_ids, val_ids, seq_alldata, alllabel, worker_init_fn):
#     tempt_data, tempt_label = copy.deepcopy(seq_alldata), copy.deepcopy(alllabel)

#     # get val data
#     val_data, val_label = [], []
#     val_index = [sub_ids.index(val_id) for val_id in val_ids]
#     for idx in sorted(val_index, reverse=True):
#         val_data.append(tempt_data[idx])
#         val_label.append(tempt_label[idx])

#     # get train data and create subject ID arrays
#     train_data = []
#     train_label = []
#     train_subject_ids = []
    
#     for idx, seq in enumerate(tempt_data):
#         if sub_ids[idx] in train_ids:
#             train_data.append(seq)
#             train_label.append(tempt_label[idx])
#             # Create subject ID array for each trial (0-based indexing for discriminator)
#             subject_id = sub_ids[idx] - 1  # Convert to 0-based
#             train_subject_ids.extend([subject_id] * len(seq))
    
#     val_subject_ids = []
#     for idx in val_index:
#         subject_id = sub_ids[idx] - 1  # Convert to 0-based
#         val_subject_ids.extend([subject_id] * len(tempt_data[idx]))
    
#     train_data = np.concatenate(train_data, axis=0)
#     train_label = np.concatenate(train_label, axis=0)
#     train_label = train_label.flatten()
#     val_data = np.concatenate(val_data, axis=0)
#     val_label = np.concatenate(val_label, axis=0)
#     val_label = val_label.flatten()

#     train_data = np.squeeze(train_data)
#     val_data = np.squeeze(val_data)

#     train_label = train_label.reshape(-1,1)
#     val_label = val_label.reshape(-1,1)
    
#     print(f"train_data_shape{train_data.shape},val_data_shape{val_data.shape}")

#     # Create datasets with subject IDs if using domain adversarial training
#     use_domain_adversarial = getattr(args, 'use_domain_adversarial', False)
    
#     if use_domain_adversarial:
#         train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label, train_subject_ids),
#                                       batch_size=args.batch_size, drop_last=True, shuffle=True,
#                                       worker_init_fn=worker_init_fn, num_workers=0)  # num_workers=0 for determinism
#         valid_loader = DataLoader(dataset=CustomDatasets(val_data, val_label, val_subject_ids),
#                                       batch_size=args.batch_size, drop_last=True,
#                                       worker_init_fn=worker_init_fn, num_workers=0)
#     else:
#         train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
#                                       batch_size=args.batch_size, drop_last=True, shuffle=True,
#                                       worker_init_fn=worker_init_fn, num_workers=0)
#         valid_loader = DataLoader(dataset=CustomDatasets(val_data, val_label),
#                                       batch_size=args.batch_size, drop_last=True,
#                                       worker_init_fn=worker_init_fn, num_workers=0)
    
#     #####################################################################################
#     #2.define model
#     #####################################################################################
#     BaselineNet = Trynetwork(
#         model = DARNet(args).to(device),
#         train_loader=train_loader, 
#         valid_loader=valid_loader, 
#         test_loader=None,
#         batch_size = args.batch_size, 
#         lr = args.lr,
#         weight_decay = args.weight_decay,
#         worker_init_fn = worker_init_fn)
#     model_val_acc = BaselineNet.train(args, val_ids)
#     return model_val_acc

    
# if __name__ == '__main__':
#     # Training settings
#     args = argparse.ArgumentParser()
#     args.seed = 42
    
#     # Set all seeds for reproducibility
#     print(f"Setting all seeds to {args.seed} for reproducibility...")
#     worker_init_fn = set_all_seeds(args.seed)
#     print("All seeds set successfully!")

#     # data
#     args.dataset = 'MM-AAD'
#     args.start_time = datetime.now().strftime(f"task1_AAD_{args.dataset}_trainer2_%Y-%m-%d-%H-%M-%S")
#     print('start time:',args.start_time)
#     options = {'MM-AAD':[40 ,32, 20, 128, "/home/suyash.kumar.mec22.itbhu/EEG-AAD_audio_visual/preprocessed/data/", "/home/suyash.kumar.mec22.itbhu/EEG-AAD_audio_visual/preprocessed/label/"]}
#     args.subject_number = options[args.dataset][0]
#     args.eeg_channel = options[args.dataset][1]
#     args.trail_number = options[args.dataset][2]
#     args.fs  = options[args.dataset][3]
#     args.data_path = options[args.dataset][4]
#     args.label_path = options[args.dataset][5]

#     args.win_time = 2
#     args.win_len = math.ceil(args.fs * args.win_time)
#     args.overlap = 0.5
#     args.window_lap = args.win_len * (1 - args.overlap)

#     # basic info of the model
#     args.optimizer='AdamW'
#     args.model = "DARNet"
#     args.batch_size = 160
#     args.lr = 7e-4
#     args.lam = 0.2
#     args.lr_decayrate = 0.45
#     args.beta1=0.92
#     args.beta2=0.998
#     args.weight_decay = 8e-4
#     args.max_epoch = 100
#     args.patience = 10
#     args.log_interval = 10
    
#     # Domain Adversarial Training parameters
#     args.use_domain_adversarial = True  # Enable domain adversarial training
#     args.lambda_domain = 0.08  # Domain adversarial weight (start small)
    
#     # save to 
#     filename = "./exps/cross-subject/%s/" % args.model
#     args.model_save_path = f'{filename}baseline_%s/' % args.start_time
#     makePath(args.model_save_path)
#     args.fig_path = f'{filename}figures/' 
#     makePath(args.fig_path)

#     print('=' * 108)
#     print('Arguments =')
#     for arg in np.sort(list(vars(args).keys())):
#         print('\t' + arg + ':', getattr(args, arg))
#     print('=' * 108)
   
#     sub_ids =  list(range(1, args.subject_number+1)) 
#     del_ids = [31,32,33,34,35,36,37,38,39,40]
#     sub_ids = [sub_id for sub_id in sub_ids if sub_id not in del_ids]

#     # Load all subject data once
#     seq_alldata, alllabel = getData(args, sub_ids)

#     # Define sequential folds: [1–5], [6–10], [11–15], [16–20], [21–25], [26–30]
#     folds = [
#         list(range(1, 6)),
#         list(range(6, 11)),
#         list(range(11, 16)),
#         list(range(16, 21)),
#         list(range(21, 26)),
#         list(range(26, 31))
#     ]

#     all_fold_acc = []

#     for fold, val_ids in enumerate(folds, start=1):
#         train_ids = [s for s in sub_ids if s not in val_ids]
#         print(f"\n========== Fold {fold} ==========")
#         print(f"Train IDs: {train_ids}")
#         print(f"Val IDs:   {val_ids}")

#         fold_acc = cross_subject(args, sub_ids, train_ids, val_ids, seq_alldata, alllabel, worker_init_fn)
#         all_fold_acc.append(fold_acc)

#     mean_acc = np.mean(all_fold_acc)

#     print(f"lr:{args.lr } -> bs:{args.batch_size}")
#     print(f"6-Fold Cross Subject Accuracies: {all_fold_acc}")
#     print(f"Mean Accuracy over 6 folds: {mean_acc:.4f}")
#     print('=' * 108)
#     now1 = datetime.now().strftime("%y-%m-%d-%H:%M:%S")
#     print('end time:', now1)