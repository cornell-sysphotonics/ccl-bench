#!/bin/bash
# =============================================================================
# Qwen3-32B DP+PP (DP=4, PP=2) - 2 Nodes, 8 GPUs
# =============================================================================
#SBATCH -C gpu
#SBATCH -A m4999
#SBATCH -q regular
#SBATCH -J qwen3_32b_dp_pp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 00:45:00
#SBATCH --output=logs/qwen3_32b_dp_pp_%j.out
#SBATCH --error=logs/qwen3_32b_dp_pp_%j.err

set -euo pipefail

# Source common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# =============================================================================
# Workload Configuration
# =============================================================================
WORKLOAD_NAME="qwen3_32b_dp_pp"
CONFIG_FILE="${TRAIN_CONFIG_DIR}/qwen3_32b_dp_pp.toml"
NSYS_PREFIX="qwen3_32b_dp_pp"

# =============================================================================
# Job Execution
# =============================================================================

# Setup environment
setup_runtime_paths
setup_distributed
setup_trace_dir "${WORKLOAD_NAME}"

# Print job summary
print_job_summary "${WORKLOAD_NAME}" "${CONFIG_FILE}"

# Create logs directory if needed
mkdir -p "${CCL_BENCH_HOME}/logs"

# Launch training with NSys profiling
cd "${TRACE_DIR}"
srun launch_torchtitan_with_nsys "${CONFIG_FILE}" "${NSYS_PREFIX}"

# Job complete
print_job_complete
