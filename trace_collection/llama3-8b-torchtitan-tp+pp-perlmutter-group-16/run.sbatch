#!/bin/bash
# =============================================================================
# LLaMA-3.1-8B TP+PP (TP=2, PP=2) - 1 Node, 4 GPUs
# =============================================================================
#SBATCH -C gpu
#SBATCH -A m4999
#SBATCH -q debug
#SBATCH -J llama3_8b_tp_pp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
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
launch_torchtitan_without_nsys "llama3_8b_tp_pp" "${WORKLOAD_DIR}" "${WORKLOAD_DIR}/train_config.toml"

# Print that the analysis is starting
echo "Starting analysis..."

# Run the analysis using absolute path
PROJECT_ROOT="$(cd "${WORKLOAD_DIR}/../.." && pwd)"
python "${PROJECT_ROOT}/tools/main.py" --workload-dir "${WORKLOAD_DIR}"

# Print that the analysis is complete
echo "Analysis complete"

# Job complete
print_job_complete
