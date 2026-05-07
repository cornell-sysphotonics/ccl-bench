#!/bin/bash
# Run script for deepseek-v3-16b-torchtitan-ep8-dp8-perlmutter
# Framework: TorchTitan  |  Model: deepseek_v3_16b  |  TP=1  DP_shard=8  PP=1  EP=8
set -euo pipefail

NUM_NODES=2
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
  --job.config_file trace_collection/deepseek-v3-16b-torchtitan-ep8-dp8-perlmutter/deepseek_v3_16b.toml \
  --parallelism.tensor_parallel_degree 1 \
  --parallelism.data_parallel_shard_degree 8 \
  --parallelism.data_parallel_replicate_degree 1 \
  --parallelism.pipeline_parallel_degree 1 \
  --parallelism.expert_parallel_degree 8 \
  --profiling.enable_profiling true \
  --profiling.save_traces_folder ./profile_traces
