#!/bin/bash
#SBATCH -A <your_allocation>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -J E1.1_llama8b_tp1
#SBATCH -o logs/E1.1_%j.out
#SBATCH -e logs/E1.1_%j.err

# E1.1: Llama-8B | TP=1 | PP=1 | EP=1 | 1 GPU

module load python
module load cuda/12.4

# Activate environment
source activate vllm-profiling

# Create log directory
mkdir -p logs

# Set NCCL environment variables for profiling
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL,P2P,INIT

# Run profiling
srun python vllm_profiler.py \
    --config experiments/configs/E3.3_deepseek-v2-lite.yaml

echo "Experiment E1.1 completed!"
