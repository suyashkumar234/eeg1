#!/bin/bash
#SBATCH --job-name=eeg-train          # Job name
#SBATCH --partition=gpu               # Partition name on your cluster
#SBATCH --gres=gpu:2                  # Number of GPUs
#SBATCH --nodelist=gpu007
#SBATCH --cpus-per-task=8             # CPU cores per task                    
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
echo "Starting parallel training on 2 GPUs..."

# --- Run your training scripts in parallel with separate logs ---
echo "=== STARTING TRAINER2 ON GPU 1 ===" > trainer2_output.log
echo "Started at: $(date)" >> trainer2_output.log
CUDA_VISIBLE_DEVICES=1 python -u trainer2.py >> trainer2_output.log 2>&1 &
PID1=$!
echo "Started trainer2.py with PID: $PID1 on GPU 1"

echo "=== STARTING TRAINER3 ON GPU 0 ===" > trainer3_output.log  
echo "Started at: $(date)" >> trainer3_output.log
CUDA_VISIBLE_DEVICES=0 python -u trainer3.py >> trainer3_output.log 2>&1 &
PID2=$!
echo "Started trainer3.py with PID: $PID2 on GPU 0"

# Monitor both processes
echo "Both training jobs started. Monitoring progress..."

# Wait for both processes to complete
echo "Waiting for both training jobs to complete..."
wait $PID1
EXIT_CODE1=$?
echo "=== TRAINER2 FINISHED ===" >> trainer2_output.log
echo "Finished at: $(date)" >> trainer2_output.log
echo "trainer2.py completed with exit code: $EXIT_CODE1"

wait $PID2  
EXIT_CODE2=$?
echo "=== TRAINER3 FINISHED ===" >> trainer3_output.log
echo "Finished at: $(date)" >> trainer3_output.log
echo "trainer3.py completed with exit code: $EXIT_CODE2"

echo "Job finished at: $(date)"

# Final status report
echo "========== FINAL TRAINING REPORT =========="
echo "trainer2.py (GPU 1): Exit code $EXIT_CODE1"
echo "trainer3.py (GPU 0): Exit code $EXIT_CODE2"

if [ $EXIT_CODE1 -eq 0 ] && [ $EXIT_CODE2 -eq 0 ]; then
    echo "✅ Both training jobs completed successfully!"
    exit 0
else
    echo "❌ One or both training jobs failed!"
    exit 1
fi


# #!/bin/bash
# #SBATCH --job-name=eeg-train          # Job name
# #SBATCH --partition=gpu               # Partition name on your cluster
# #SBATCH --gres=gpu:2                 # Number of GPUs
# #SBATCH --nodelist=gpu007
# #SBATCH --cpus-per-task=8             # CPU cores per task                    
# #SBATCH --time=48:00:00               # Max run time (48 hours)
# #SBATCH --output=slurm-%j.out         # STDOUT file
# #SBATCH --error=slurm-%j.err          # STDERR file

# # --- Load modules / activate environment ---
# module purge
# module load cuda   # or the CUDA module your cluster uses

# # Activate your conda env
# source /home/apps/miniconda3/bin/activate
# conda activate eegenv

# # Move to the working directory
# cd /scratch/$USER/EEG-AAD/eeg-aad-challenge2026-task1-baselines-master

# echo "Job started at: $(date)"
# echo "Running on node: $(hostname)"
# echo "GPU allocation: $CUDA_VISIBLE_DEVICES"

# # --- Run your training script ---
# # Since you have 1 GPU, use GPU 0 (not GPU 1)
# CUDA_VISIBLE_DEVICES=1 python -u trainer2.py &
# CUDA_VISIBLE_DEVICES=0 python -u trainer3.py &


# echo "Job finished at: $(date)"