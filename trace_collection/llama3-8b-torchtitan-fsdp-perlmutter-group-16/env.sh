#!/bin/bash
# =============================================================================
# CCL-Bench Perlmutter-16 Workload Environment Configuration
# =============================================================================
# This file is sourced by install.sh, run.sh, and run.sbatch scripts
# to provide consistent environment configuration across all workloads.
#
# Usage:
#   source .env
#   # or automatically sourced by install.sh/run.sh/run.sbatch
# =============================================================================

# =============================================================================
# NERSC/Perlmutter Paths
# =============================================================================

# SCRATCH is set by NERSC environment, fallback to home if not available
export SCRATCH="${SCRATCH:-${HOME}/scratch}"

# =============================================================================
# UV Package Manager Configuration
# =============================================================================

# Store UV cache on scratch for better performance and quota management
export UV_CACHE_DIR="${SCRATCH}/cache/uv"

# Use copy mode for better reliability across compute nodes
export UV_LINK_MODE=copy

# =============================================================================
# HuggingFace Configuration
# =============================================================================

# HuggingFace cache and home directory (token stored at HF_HOME/token)
export HF_HOME="${SCRATCH}/cache/huggingface"

# Directory where model assets (tokenizer, config) are downloaded
export HF_ASSETS_ROOT="${SCRATCH}/ccl-bench-assets/models"

# =============================================================================
# Python Environment
# =============================================================================

# Virtual environment location
export VENV_DIR="${SCRATCH}/ccl-bench-venv"

# =============================================================================
# Trace Collection
# =============================================================================

# Base directory for all collected traces (nsys, torch profiler, etc.)
# Note: This is legacy/optional. Traces are now saved in <workload_dir>/traces
# by default via setup_trace_dir() in common.sh
export TRACE_BASE="${SCRATCH}/ccl-bench-traces"

# =============================================================================
# MPI Configuration
# =============================================================================

# GPU-aware MPI (needed for CUDA ptrs in MPI & GPUDirect RDMA)
export MPICH_GPU_SUPPORT_ENABLED=1

# =============================================================================
# OpenMP Configuration
# =============================================================================

# Reasonable number of CPU threads per rank (tune if needed)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# =============================================================================
# NCCL Configuration
# =============================================================================

# Debug level: WARN (default), INFO (verbose), or TRACE (very verbose)
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

# Enable async error handling for robustness
# Note: NCCL_ASYNC_ERROR_HANDLING is deprecated, use TORCH_NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"

# =============================================================================
# Distributed Training Configuration
# =============================================================================

# Master address and port for torch.distributed
# MASTER_ADDR is set dynamically from SLURM_JOB_NODELIST when running under Slurm
if [[ -n ${SLURM_JOB_NODELIST:-} ]]; then
	MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1 || true)
	export MASTER_ADDR
else
	export MASTER_ADDR="${MASTER_ADDR:-localhost}"
fi
export MASTER_PORT="${MASTER_PORT:-29500}"
