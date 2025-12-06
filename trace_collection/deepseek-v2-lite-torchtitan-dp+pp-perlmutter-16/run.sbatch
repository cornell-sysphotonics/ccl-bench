#!/bin/bash
# =============================================================================
# DeepSeek-V2-Lite DP+PP (DP=2, FSDP=2, PP=2) - 1 Node, 4 GPUs
# =============================================================================
#SBATCH -C gpu
#SBATCH -A m4999
#SBATCH -q regular
#SBATCH -J deepseek_dp_pp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 00:45:00
#SBATCH --output=logs/deepseek_v2_lite_dp_pp_%j.out
#SBATCH --error=logs/deepseek_v2_lite_dp_pp_%j.err

set -euo pipefail

# Source common configuration from perlmutter directory
# Navigate from workload folder to project root, then to perlmutter/
WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${WORKLOAD_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/perlmutter/common.sh"

# =============================================================================
# Workload Configuration
# =============================================================================
WORKLOAD_NAME="deepseek_v2_lite_dp_pp"
NSYS_PREFIX="deepseek_v2_lite_dp_pp"

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
