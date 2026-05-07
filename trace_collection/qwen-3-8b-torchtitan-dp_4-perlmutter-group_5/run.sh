#!/bin/bash
# Run script for qwen-3-8b-torchtitan-dp_4-perlmutter-group_5
# Framework: TorchTitan  |  Model: qwen-3-8b  |  TP=1  DP_shard=4  PP=1  EP=1
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
  --job.config_file trace_collection/qwen-3-8b-torchtitan-dp_4-perlmutter-group_5/qwen-dp-4gpu-1batch.toml \
  --parallelism.tensor_parallel_degree 1 \
  --parallelism.data_parallel_shard_degree 4 \
  --parallelism.data_parallel_replicate_degree 1 \
  --parallelism.pipeline_parallel_degree 1 \
  --parallelism.expert_parallel_degree 1 \
  --profiling.enable_profiling true \
  --profiling.save_traces_folder ./profile_traces
