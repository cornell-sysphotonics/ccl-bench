#!/bin/bash
# =============================================================================
# Submit All CCL-Bench Workloads to Slurm
# =============================================================================
# This script submits all workload jobs in sequence.
# Edit NERSC_ALLOCATION in common.sh before running!
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

# =============================================================================
# Configuration
# =============================================================================

# Check allocation is set
# shellcheck disable=SC2154
if [[ ${NERSC_ALLOCATION} == \"CHANGE_ME\" ]]; then
	echo \"ERROR: Please set NERSC_ALLOCATION in common.sh\"
	# shellcheck disable=SC2016
	echo '  Get your allocation with: sacctmgr show assoc user=$USER'
	exit 1
fi

# Workload folders (in trace_collection/)
WORKLOAD_FOLDERS=(
	"llama3.1-8b-torchtitan-tp-perlmutter-16"
	"llama3.1-8b-torchtitan-pp-perlmutter-16"
	"deepseek-v2-lite-torchtitan-dp+tp-perlmutter-16"
	"deepseek-v2-lite-torchtitan-dp+pp-perlmutter-16"
	"qwen3-32b-torchtitan-3d-perlmutter-16"
	"qwen3-32b-torchtitan-dp+tp-perlmutter-16"
	"qwen3-32b-torchtitan-dp+pp-perlmutter-16"
)

# =============================================================================
# Main
# =============================================================================

echo "============================================================================="
echo "CCL-Bench Job Submission"
echo "============================================================================="
echo "Allocation: ${NERSC_ALLOCATION}"
echo "Workloads:  ${#WORKLOAD_FOLDERS[@]} jobs"
echo ""

# Create logs directory
# shellcheck disable=SC2154
mkdir -p "${CCL_BENCH_HOME}/logs"

# Submit each workload
submitted_jobs=()
for workload_folder in "${WORKLOAD_FOLDERS[@]}"; do
	script_path="${PROJECT_ROOT}/trace_collection/${workload_folder}/run.sbatch"

	if [[ ! -f ${script_path} ]]; then
		echo "WARNING: Script not found: ${script_path}"
		continue
	fi

	echo "Submitting: ${workload_folder}..."
	job_id=$(sbatch --parsable -A "${NERSC_ALLOCATION}" "${script_path}")
	submitted_jobs+=("${job_id}:${workload_folder}")
	echo "  Job ID: ${job_id}"
done

echo ""
echo "============================================================================="
echo "Submitted Jobs:"
echo "============================================================================="
for job_info in "${submitted_jobs[@]}"; do
	IFS=':' read -r job_id workload_folder <<< "${job_info}"
	echo "  ${job_id}: ${workload_folder}"
done

echo ""
# shellcheck disable=SC2016
echo 'Monitor jobs with: squeue -u $USER'
echo "View job output:   tail -f logs/<job_name>_<job_id>.out"
echo "============================================================================="
