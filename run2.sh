#!/bin/bash
#SBATCH --job-name=eeg-train          # Job name
#SBATCH --partition=gpu               # Partition name on your cluster
#SBATCH --gres=gpu:1                 # Number of GPUs
#SBATCH --nodelist=gpu006
#SBATCH --cpus-per-task=16            # CPU cores per task                    
#SBATCH --time=48:00:00               # Max run time (48 hours)
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

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU allocation: $CUDA_VISIBLE_DEVICES"

# --- Run your training script ---
# Since you have 1 GPU, use GPU 0 (not GPU 1)
CUDA_VISIBLE_DEVICES=0 python -u trainer2.py

echo "Job finished at: $(date)"