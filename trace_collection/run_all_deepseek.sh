#!/bin/bash
# =============================================================================
# Run All DeepSeek-V2-Lite Workloads
# =============================================================================
# This script finds all directories starting with "deepseek-v2-lite" and
# runs their respective run.sh scripts.
#
# Usage:
#   ./run_all_deepseek.sh              # Run all workloads
#   ./run_all_deepseek.sh --dry-run     # Show what would be run without executing
#   ./run_all_deepseek.sh --stop-on-error  # Stop on first error (default: continue)
#   ./run_all_deepseek.sh --parallel   # Run all in parallel (use with caution)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Parse command-line arguments
DRY_RUN=false
STOP_ON_ERROR=false
PARALLEL=false

while [[ $# -gt 0 ]]; do
	case $1 in
		--dry-run)
			DRY_RUN=true
			shift
			;;
		--stop-on-error)
			STOP_ON_ERROR=true
			shift
			;;
		--parallel)
			PARALLEL=true
			shift
			;;
		-h|--help)
			echo "Usage: $0 [--dry-run] [--stop-on-error] [--parallel]"
			echo ""
			echo "Options:"
			echo "  --dry-run        Show what would be run without executing"
			echo "  --stop-on-error  Stop on first error (default: continue)"
			echo "  --parallel       Run all workloads in parallel"
			echo "  -h, --help       Show this help message"
			exit 0
			;;
		*)
			echo "Unknown option: $1"
			echo "Use --help for usage information"
			exit 1
			;;
	esac
done

# Find all directories starting with "deepseek-v2-lite"
mapfile -t WORKLOAD_DIRS < <(find . -maxdepth 1 -type d -name "deepseek-v2-lite*" | sort)

if [[ ${#WORKLOAD_DIRS[@]} -eq 0 ]]; then
	echo "ERROR: No directories found starting with 'deepseek-v2-lite'"
	exit 1
fi

echo "=============================================================================="
echo "Found ${#WORKLOAD_DIRS[@]} DeepSeek-V2-Lite workload(s)"
echo "=============================================================================="
echo ""

# Display workloads that will be run
for dir in "${WORKLOAD_DIRS[@]}"; do
	echo "  - $(basename "${dir}")"
done
echo ""

if [[ ${DRY_RUN} == true ]]; then
	echo "[DRY-RUN MODE] Would run the following:"
	echo ""
fi

# Track results
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_WORKLOADS=()

# Function to run a single workload
run_workload() {
	local workload_dir="$1"
	local workload_name
	workload_name="$(basename "${workload_dir}")"
	
	# Convert to absolute path
	local abs_workload_dir
	abs_workload_dir="$(cd "${SCRIPT_DIR}/${workload_dir}" && pwd)"
	
	echo "------------------------------------------------------------------------------"
	echo "Running: ${workload_name}"
	echo "------------------------------------------------------------------------------"
	
	if [[ ! -f "${abs_workload_dir}/run.sh" ]]; then
		echo "ERROR: run.sh not found in ${abs_workload_dir}"
		return 1
	fi
	
	if [[ ! -x "${abs_workload_dir}/run.sh" ]]; then
		echo "Making run.sh executable..."
		chmod +x "${abs_workload_dir}/run.sh"
	fi
	
	if [[ ${DRY_RUN} == true ]]; then
		echo "[DRY-RUN] Would execute: cd ${abs_workload_dir} && ./run.sh"
		return 0
	fi
	
	# Run the workload (use subshell to preserve current directory)
	if (cd "${abs_workload_dir}" && ./run.sh); then
		echo "✓ Successfully submitted: ${workload_name}"
		return 0
	else
		local exit_code=$?
		echo "✗ Failed to run: ${workload_name} (exit code: ${exit_code})"
		return 1
	fi
}

# Run workloads
if [[ ${PARALLEL} == true ]]; then
	echo "Running workloads in parallel..."
	echo ""
	
	# Run all workloads in background
	PIDS=()
	for dir in "${WORKLOAD_DIRS[@]}"; do
		run_workload "${dir}" &
		PIDS+=($!)
	done
	
	# Wait for all background jobs and collect results
	for i in "${!PIDS[@]}"; do
		if wait "${PIDS[$i]}"; then
			((SUCCESS_COUNT++)) || true
		else
			((FAILED_COUNT++)) || true
			FAILED_WORKLOADS+=("$(basename "${WORKLOAD_DIRS[$i]}")")
			if [[ ${STOP_ON_ERROR} == true ]]; then
				echo ""
				echo "Stopping due to error (--stop-on-error flag)"
				# Kill remaining background jobs
				for pid in "${PIDS[@]:$((i+1))}"; do
					kill "${pid}" 2>/dev/null || true
				done
				break
			fi
		fi
	done
else
	# Run workloads sequentially
	for dir in "${WORKLOAD_DIRS[@]}"; do
		if run_workload "${dir}"; then
			((SUCCESS_COUNT++)) || true
		else
			((FAILED_COUNT++)) || true
			FAILED_WORKLOADS+=("$(basename "${dir}")")
			if [[ ${STOP_ON_ERROR} == true ]]; then
				echo ""
				echo "Stopping due to error (--stop-on-error flag)"
				break
			fi
		fi
		echo ""
	done
fi

# Print summary
echo "=============================================================================="
echo "Summary"
echo "=============================================================================="
echo "Total workloads: ${#WORKLOAD_DIRS[@]}"
echo "Successful: ${SUCCESS_COUNT}"
echo "Failed: ${FAILED_COUNT}"

if [[ ${FAILED_COUNT} -gt 0 ]]; then
	echo ""
	echo "Failed workloads:"
	for workload in "${FAILED_WORKLOADS[@]}"; do
		echo "  - ${workload}"
	done
	exit 1
else
	echo ""
	echo "All workloads processed successfully!"
	exit 0
fi

