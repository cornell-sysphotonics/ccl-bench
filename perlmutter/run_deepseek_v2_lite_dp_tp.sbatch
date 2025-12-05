#!/bin/bash
# =============================================================================
# DeepSeek-V2-Lite DP+TP (DP=2, FSDP=2, TP=2) - 1 Node, 4 GPUs
# =============================================================================
#SBATCH -C gpu
#SBATCH -A m4999
#SBATCH -q regular
#SBATCH -J deepseek_dp_tp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 00:45:00
#SBATCH --output=logs/deepseek_v2_lite_dp_tp_%j.out
#SBATCH --error=logs/deepseek_v2_lite_dp_tp_%j.err

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
WORKLOAD_NAME="deepseek_v2_lite_dp_tp"
CONFIG_FILE="${TRAIN_CONFIG_DIR}/deepseek_v2_lite_dp_tp.toml"
NSYS_PREFIX="deepseek_v2_lite_dp_tp"

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
launch_torchtitan_with_nsys "${CONFIG_FILE}" "${NSYS_PREFIX}"

# Job complete
print_job_complete
