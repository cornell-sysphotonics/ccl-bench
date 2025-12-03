#!/bin/bash
# =============================================================================
# Submit All CCL-Bench Workloads to Slurm
# =============================================================================
# This script submits all workload jobs in sequence.
# Edit NERSC_ALLOCATION in common.sh before running!
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# =============================================================================
# Configuration
# =============================================================================

# Check allocation is set
if [[ "${NERSC_ALLOCATION}" == "CHANGE_ME" ]]; then
    echo "ERROR: Please set NERSC_ALLOCATION in common.sh"
    echo "  Get your allocation with: sacctmgr show assoc user=\$USER"
    exit 1
fi

# Job scripts to submit
WORKLOADS=(
    "run_llama3_8b_tp.sbatch"
    "run_llama3_8b_pp.sbatch"
    "run_deepseek_v2_lite_dp_tp.sbatch"
    "run_deepseek_v2_lite_dp_pp.sbatch"
    "run_qwen3_32b_3d.sbatch"
    "run_qwen3_32b_dp_tp.sbatch"
    "run_qwen3_32b_dp_pp.sbatch"
)

# =============================================================================
# Main
# =============================================================================

echo "============================================================================="
echo "CCL-Bench Job Submission"
echo "============================================================================="
echo "Allocation: ${NERSC_ALLOCATION}"
echo "Workloads:  ${#WORKLOADS[@]} jobs"
echo ""

# Create logs directory
mkdir -p "${CCL_BENCH_HOME}/logs"

# Submit each workload
submitted_jobs=()
for workload in "${WORKLOADS[@]}"; do
    script_path="${SCRIPT_DIR}/${workload}"

    if [[ ! -f "${script_path}" ]]; then
        echo "WARNING: Script not found: ${script_path}"
        continue
    fi

    echo "Submitting: ${workload}..."
    job_id=$(sbatch --parsable -A "${NERSC_ALLOCATION}" "${script_path}")
    submitted_jobs+=("${job_id}:${workload}")
    echo "  Job ID: ${job_id}"
done

echo ""
echo "============================================================================="
echo "Submitted Jobs:"
echo "============================================================================="
for job_info in "${submitted_jobs[@]}"; do
    IFS=':' read -r job_id workload <<< "${job_info}"
    echo "  ${job_id}: ${workload}"
done

echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "View job output:   tail -f logs/<job_name>_<job_id>.out"
echo "============================================================================="
