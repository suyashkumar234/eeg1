#!/bin/bash
#SBATCH --job-name=eeg-train          # Job name
#SBATCH --partition=gpu               # Partition name on your cluster
#SBATCH --gres=gpu:1                   # Number of GPUs
#SBATCH --nodelist=gpu007
#SBATCH --cpus-per-task=8             # (optional) CPU cores per task                    
#SBATCH --time=4:00:00               # Max run time (hh:mm:ss)
#SBATCH --output=slurm-%j.out         # STDOUT file
#SBATCH --error=slurm-%j.err          # STDERR file

# --- Load modules / activate environment ---
module purge
module load cuda   # or the CUDA module your cluster uses

# Activate your conda env
source /home/apps/miniconda3/bin/activate
conda activate eegenv

# Move to the working directory
cd /scratch/$USER/EEG-AAD/eeg-aad-challenge2026-task1-baselines-master

# Optional: print diagnostic info
echo "Running on node: $HOSTNAME"
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

# --- Run your training script ---
export CUDA_VISIBLE_DEVICES=0
python trainer.py

