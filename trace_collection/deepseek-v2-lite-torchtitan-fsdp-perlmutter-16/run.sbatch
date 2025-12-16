#!/bin/bash
# =============================================================================
# DeepSeek-V2-Lite (16B MoE) FSDP Only - 1 Node, 4 GPUs
# =============================================================================
#SBATCH -C gpu
#SBATCH -A m4999
#SBATCH -q debug
#SBATCH -J deepseek_v2_lite_fsdp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 00:30:00
#SBATCH --module=gpu,nccl-plugin
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# =============================================================================
# Load Required Modules
# =============================================================================
# Load CUDA toolkit for torch GPU support
module load cudatoolkit/12.9 2>/dev/null || true

# =============================================================================
# Setup Paths
# =============================================================================
# Use SLURM_SUBMIT_DIR when available (keeps original submission dir).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  WORKLOAD_DIR="${SLURM_SUBMIT_DIR}"
else
  WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Ensure logs directory exists (Slurm needs it before job starts)
mkdir -p "${WORKLOAD_DIR}/logs" 2>/dev/null || true

# Source environment configuration
# shellcheck disable=SC1091
source "${WORKLOAD_DIR}/env.sh"

# Source common functions
# shellcheck disable=SC1091
source "${WORKLOAD_DIR}/common.sh"

# =============================================================================
# Job Execution
# =============================================================================

# Setup trace directory
setup_trace_dir "${WORKLOAD_DIR}"

# Print job summary
print_job_summary

# Activate virtual environment
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
	# shellcheck disable=SC1091
	source "${VENV_DIR}/bin/activate"
fi

# Change to trace directory for output
cd "${TRACE_DIR}"

# Launch training with NSys profiling
launch_torchtitan_nsys "deepseek_v2_lite_fsdp" "${WORKLOAD_DIR}" "${WORKLOAD_DIR}/train_config.toml"

# Job complete
print_job_complete
