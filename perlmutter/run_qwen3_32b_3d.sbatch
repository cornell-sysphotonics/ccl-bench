#!/bin/bash
# =============================================================================
# Qwen3-32B Full 3D Parallelism (DP=2, TP=4, PP=2) - 4 Nodes, 16 GPUs
# =============================================================================
#SBATCH -C gpu
#SBATCH -A m4999
#SBATCH -q regular
#SBATCH -J qwen3_32b_3d
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 01:00:00
#SBATCH --output=logs/qwen3_32b_3d_%j.out
#SBATCH --error=logs/qwen3_32b_3d_%j.err

set -euo pipefail

# Source common configuration
# Use SLURM_SUBMIT_DIR since SLURM copies the script to a temp location
if [[ -f "${SLURM_SUBMIT_DIR}/common.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/perlmutter"
fi
source "${SCRIPT_DIR}/common.sh"

# =============================================================================
# Workload Configuration
# =============================================================================
WORKLOAD_NAME="qwen3_32b_3d"
CONFIG_FILE="${TRAIN_CONFIG_DIR}/qwen3_32b_3d.toml"
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
print_job_summary "${WORKLOAD_NAME}" "${CONFIG_FILE}"

# Create logs directory if needed
mkdir -p "${CCL_BENCH_HOME}/logs"

# Launch training with NSys profiling
cd "${TRACE_DIR}"
srun launch_torchtitan_with_nsys "${CONFIG_FILE}" "${NSYS_PREFIX}"

# Job complete
print_job_complete
