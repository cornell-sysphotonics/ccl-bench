#!/bin/bash
# =============================================================================
# Generic Perlmutter-16 Workload - Run Script
# =============================================================================
# Simple wrapper to submit the SLURM batch job for this workload.
# Automatically detects workload name from directory.
#
# Usage:
#   ./run.sh              # Submit job with default settings
#   ./run.sh --dry-run    # Show what would be submitted without running
#
# Environment variables:
#   PROFILE_MODE    - "both" (default), "nsys", or "torch"
# =============================================================================

set -euo pipefail

WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${WORKLOAD_DIR}/../.." && pwd)"
WORKLOAD_NAME="$(basename "${WORKLOAD_DIR}")"

# Source environment configuration
if [[ -f "${WORKLOAD_DIR}/env.sh" ]]; then
	# shellcheck disable=SC1091
	source "${WORKLOAD_DIR}/env.sh"
fi

# Check for dry-run flag
DRY_RUN=false
if [[ ${1:-} == "--dry-run" ]]; then
	DRY_RUN=true
fi

echo "========================================"
echo "Workload: ${WORKLOAD_NAME}"
echo "========================================"
echo "Workload directory: ${WORKLOAD_DIR}"
echo "Project root: ${PROJECT_ROOT}"
echo "Profile mode: ${PROFILE_MODE:-both}"
echo ""

# Verify required files exist
if [[ ! -f "${WORKLOAD_DIR}/run.sbatch" ]]; then
	echo "ERROR: run.sbatch not found in ${WORKLOAD_DIR}"
	exit 1
fi

if [[ ! -f "${WORKLOAD_DIR}/train_config.toml" ]]; then
	echo "ERROR: train_config.toml not found in ${WORKLOAD_DIR}"
	exit 1
fi

if [[ ${DRY_RUN} == true ]]; then
	echo "[DRY-RUN] Would submit: sbatch ${WORKLOAD_DIR}/run.sbatch"
	echo ""
	echo "SBATCH configuration:"
	grep "^#SBATCH" "${WORKLOAD_DIR}/run.sbatch"
else
	echo "Submitting job..."
	cd "${WORKLOAD_DIR}"
	sbatch run.sbatch
fi
