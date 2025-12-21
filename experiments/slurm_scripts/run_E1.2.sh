#!/bin/bash
#SBATCH -A <your_allocation>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH -J E1.2_llama8b_tp2
#SBATCH -o logs/E1.2_%j.out
#SBATCH -e logs/E1.2_%j.err

# E1.2: Llama-8B | TP=2 | 2 GPUs

module load python
module load cuda/12.4

source activate vllm-profiling

mkdir -p logs

# NCCL settings for profiling
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL,P2P,INIT

# Run with Ray for distributed inference
srun python vllm_profiler.py \
    --config experiments/configs/E1.2_llama8b_tp2.yaml

echo "Experiment E1.2 completed!"
