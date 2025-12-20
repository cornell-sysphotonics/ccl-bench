#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# --- This script is optimized for AWS with EFA
# --- adjust NCCL_BUFFSIZE if you encounter memory
# --- constraint issues or to tune for improved performance.
# ---

#SBATCH --job-name=torchtitan_multi_node

#SBATCH --ntasks=4

#SBATCH --nodes=4

#SBATCH --gpus-per-task=4

#SBATCH --cpus-per-task=48

#SBATCH --partition=train


if [ -n "$SLURM_JOB_NODELIST" ]; then
    nodes=( $( scontrol show hostnames "$SLURM_JOB_NODELIST" ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(getent hosts "$head_node" | awk '{print $1; exit}')
fi

if [ -z "$head_node_ip" ]; then
    # Fallback for interactive salloc shells where SLURM variables may be unset.
    head_node=$(hostname)
    head_node_ip=$(hostname --ip-address)
fi

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
# Enable for A100
export FI_PROVIDER="efa"
# Ensure that P2P is available
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
# optional debug settings
# export NCCL_DEBUG=INFO
# NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152
#export TORCH_DIST_INIT_BARRIER=1
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/qwen3/train_configs/qwen3_32b.toml"}

dcgmi profile --pause
# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.
srun torchrun --nnodes 4 --nproc_per_node 4 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" -m torchtitan.train --job.config_file ${CONFIG_FILE} "$@"
dcgmi profile --resume
