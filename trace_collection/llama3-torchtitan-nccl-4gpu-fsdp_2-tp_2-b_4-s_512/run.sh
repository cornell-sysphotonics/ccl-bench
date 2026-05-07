#!/bin/bash
# Run script for llama3-torchtitan-nccl-4gpu-fsdp_2-tp_2-b_4-s_512
# Framework: TorchTitan  |  Model: llama-3.1-8b  |  TP=2  DP_shard=2  PP=1  EP=1
set -euo pipefail

NUM_NODES=1
GPUS_PER_NODE=4
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}

# Activate torchtitan environment and run
torchrun \
  --nproc_per_node=$GPUS_PER_NODE \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  torchtitan/train.py \
  --job.config_file trace_collection/llama3-torchtitan-nccl-4gpu-fsdp_2-tp_2-b_4-s_512/llama3-torchtitan-nccl-4gpu-fsdp_2-tp_2-b_4-s_512.toml \
  --parallelism.tensor_parallel_degree 2 \
  --parallelism.data_parallel_shard_degree 2 \
  --parallelism.data_parallel_replicate_degree 1 \
  --parallelism.pipeline_parallel_degree 1 \
  --parallelism.expert_parallel_degree 1 \
  --profiling.enable_profiling true \
  --profiling.save_traces_folder ./profile_traces
