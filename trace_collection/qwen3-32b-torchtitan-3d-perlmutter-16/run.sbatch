#!/bin/bash
# =============================================================================
# Qwen3-32B Full 3D Parallelism (DP=2, TP=2, PP=2) - 2 Nodes, 8 GPUs
# =============================================================================
#SBATCH -C gpu
#SBATCH -A m4999
#SBATCH -q regular
#SBATCH -J qwen3_32b_3d
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 01:00:00
#SBATCH --output=logs/qwen3_32b_3d_%j.out
#SBATCH --error=logs/qwen3_32b_3d_%j.err

set -euo pipefail

# Source common configuration from perlmutter directory
# Navigate from workload folder to project root, then to perlmutter/
WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${WORKLOAD_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/perlmutter/common.sh"

# =============================================================================
# Workload Configuration
# =============================================================================
WORKLOAD_NAME="qwen3_32b_3d"
NSYS_PREFIX="qwen3_32b_3d"

# Profiling Mode: "both" (default), "nsys" (Nsys only), "torch" (Torch Profiler only)
# Set via: export PROFILE_MODE=nsys  (before sbatch, or uncomment below)
# export PROFILE_MODE="both"

# =============================================================================
# Job Execution
# =============================================================================

# Setup environment
setup_runtime_paths
setup_distributed
setup_trace_dir "${WORKLOAD_NAME}"

# Print job summary
print_job_summary "${WORKLOAD_NAME}"

# Create logs directory if needed
mkdir -p "${CCL_BENCH_HOME}/logs"

# Launch training with NSys profiling
cd "${TRACE_DIR}"
launch_torchtitan_with_nsys "${WORKLOAD_NAME}" "${NSYS_PREFIX}"

# Job complete
print_job_complete
