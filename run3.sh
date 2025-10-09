#!/bin/bash
#SBATCH --job-name=eeg-dan-dual       # Job name
#SBATCH --partition=gpu               # Partition name on your cluster
#SBATCH --gres=gpu:2                  # Number of GPUs
#SBATCH --nodelist=gpu006
#SBATCH --cpus-per-task=16            # CPU cores per task                    
#SBATCH --time=24:00:00                # Max run time (hh:mm:ss)
#SBATCH --output=slurm-%j.out         # STDOUT file
#SBATCH --error=slurm-%j.err          # STDERR file

  # --- Load modules / activate environment ---
module purge
module load cuda

  # Activate your conda env
source /home/apps/miniconda3/bin/activate
conda activate eegenv

  # Move to the working directory
cd /scratch/$USER/EEG-AAD/eeg-aad-challenge2026-task1-baselines-master

  # --- Run both training scripts ---
CUDA_VISIBLE_DEVICES=0 python trainer1.py &
CUDA_VISIBLE_DEVICES=1 python trainer2.py &
CUDA_VISIBLE_DEVICES=0 python trainer3.py &
CUDA_VISIBLE_DEVICES=1 python trainer4.py &
CUDA_VISIBLE_DEVICES=0 python trainer5.py &
CUDA_VISIBLE_DEVICES=1 python trainer6.py &


wait
