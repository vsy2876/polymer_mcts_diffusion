#!/bin/bash
#SBATCH --job-name=sft_diff
#SBATCH --account=gts-cdeo3              # Targets Professor Chaitanya Deo's active allocation account
#SBATCH --qos=embers                      # Standard production high-priority queue on Phoenix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12                # Requests the 12-core parallel processing sweet spot
#SBATCH --gres=gpu:A100:1                  # Explicitly requests 1x NVIDIA A100 GPU node
#SBATCH --mem=64G                          # Allocates 64GB of system RAM memory
#SBATCH --time=01:00:00                    # Sets a 1-hour wall-time (SFT is very fast)
#SBATCH --output=logs/sft_diff_%j.out
#SBATCH --error=logs/sft_diff_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vyadav68@gatech.edu

mkdir -p logs

module load anaconda3
conda activate polymers

# Force all temporary file operations to build inside high-throughput scratch space
if [ -n "$SCRM" ]; then
    export SCRATCH_DIR="$SCRM"
elif [ -n "$SCRATCH" ]; then
    export SCRATCH_DIR="$SCRATCH"
fi

# Asynchronous GPU kernel optimization 
export CUDA_LAUNCH_BLOCKING=0
unset TORCH_CUDNN_BENCHMARK                

# Core Thread-Splitting Strategy for SFT:
# SFT does NOT run xTB. Therefore, we dedicate all 12 requested CPU cores 
# directly to PyTorch to maximize DataLoader speed and matrix operations.
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export PYTHONUNBUFFERED=1     # Instantly forces print statements to write to the .out log

cd $SLURM_SUBMIT_DIR
export HF_HOME="/storage/scratch1/2/vyadav68/.hf_cache"

echo "--- Launching SFT Diffusion Model ---"
python3 finetune_offline_sft_diffusion.py