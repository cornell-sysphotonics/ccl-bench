#!/bin/bash
#SBATCH -A <your_allocation>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -J E1.3_llama8b_tp4
#SBATCH -o logs/E1.3_%j.out
#SBATCH -e logs/E1.3_%j.err

# E1.3: Llama-8B | TP=4 | 4 GPUs

module load python
module load cuda/12.4

source activate vllm-profiling

mkdir -p logs

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL,P2P,INIT

srun python vllm_profiler.py \
    --config experiments/configs/E1.3_llama8b_tp4.yaml

echo "Experiment E1.3 completed!"
