#!/bin/bash
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out

export CUDA_VISIBLE_DEVICES=0

# Make sure directories exist
mkdir -p outputs/multinodes
mkdir -p results

# NCCL trace prefix (NCCL will append .rank0, .rank1, ...)
export TORCH_NCCL_TRACE_FILE="outputs/multinodes/nccl_trace.log"
export CONFIG_FILE=./torchtitan/models/llama3/train_configs/llama3_8b.toml
export HF_HOME=/pscratch/sd/k/kw746/torchtitan/data/hf_cache
export HF_DATASETS_CACHE=/pscratch/sd/k/kw746/torchtitan/data/hf_cache/datasets

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL

CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml"

MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
MASTER_PORT=29500

# Train
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m torchtitan.train \
      --job.config_file $CONFIG_FILE

# After training: convert trace → JSON
# "python3 parse_nccl_trace_to_json.py \
#     outputs/multinodes/nccl_trace.log.rank${RANK} \
#     results/rank${RANK}.json"