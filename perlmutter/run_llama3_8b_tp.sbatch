#!/bin/bash
# =============================================================================
# LLaMA-3.1-8B Pure Tensor Parallelism (TP=4) - 1 Node, 4 GPUs
# =============================================================================
#SBATCH -C gpu
#SBATCH -A m4999
#SBATCH -q regular
#SBATCH -J llama3_8b_tp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 00:30:00
#SBATCH --output=logs/llama3_8b_tp_%j.out
#SBATCH --error=logs/llama3_8b_tp_%j.err

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
WORKLOAD_NAME="llama3_8b_tp"
CONFIG_FILE="${TRAIN_CONFIG_DIR}/llama3_8b_tp.toml"
NSYS_PREFIX="llama3_8b_tp"

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
