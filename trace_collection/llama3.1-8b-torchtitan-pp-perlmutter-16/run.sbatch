#!/bin/bash
# =============================================================================
# LLaMA-3.1-8B Pure Pipeline Parallelism (PP=4) - 1 Node, 4 GPUs
# =============================================================================
#SBATCH -C gpu
#SBATCH -A m4999
#SBATCH -q debug
#SBATCH -J llama3_8b_pp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 00:30:00
#SBATCH --output=logs/llama3_8b_pp_%j.out
#SBATCH --error=logs/llama3_8b_pp_%j.err

set -euo pipefail

# Source common configuration from perlmutter directory
# Navigate from workload folder to project root, then to perlmutter/
WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${WORKLOAD_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/perlmutter/common.sh"

# =============================================================================
# Workload Configuration
# =============================================================================
WORKLOAD_NAME="llama3_8b_pp"
NSYS_PREFIX="llama3_8b_pp"

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
