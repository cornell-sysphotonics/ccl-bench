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
#SBATCH -t 00:30:00
#SBATCH --module=gpu,nccl-plugin

set -euo pipefail

# =============================================================================
# Setup Paths
# =============================================================================
WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="${WORKLOAD_DIR}/logs"
mkdir -p "${LOGS_DIR}"

# Redirect output to logs directory
exec > "${LOGS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2> "${LOGS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

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
launch_torchtitan_nsys "deepseek_v2_lite_dp_tp" "${WORKLOAD_DIR}" "${WORKLOAD_DIR}/train_config.toml"

# Job complete
print_job_complete
