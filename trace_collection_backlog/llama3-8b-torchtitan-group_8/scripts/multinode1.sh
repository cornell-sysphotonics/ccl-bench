#!/bin/bash
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00

export CUDA_VISIBLE_DEVICES=0

# Store trace by rank
export CONFIG_FILE=./torchtitan/models/llama3/train_configs/llama3_8b.toml
export HF_HOME=/pscratch/sd/k/kw746/torchtitan/data/hf_cache
export HF_DATASETS_CACHE=/pscratch/sd/k/kw746/torchtitan/data/hf_cache/datasets
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
export TORCH_NCCL_TRACE_FILE="./outputs/multinodes/nccl_rank_${SLURM_PROCID}.log"


# nsys per rank
# NSYS_CMD="nsys profile -t cuda,nvtx,osrt -o trace_rank${SLURM_PROCID}"

MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
MASTER_PORT=29500

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m torchtitan.train --job.config_file "./torchtitan/models/llama3/train_configs/llama3_8b.toml"