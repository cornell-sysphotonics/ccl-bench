#!/bin/bash
# =============================================================================
# CCL-Bench Environment Activation Script
# =============================================================================
# Source this file to activate the CCL-Bench environment.
#
# Usage:
#   source perlmutter/activate.sh
#
# This script:
#   - Loads Perlmutter modules (python, cudatoolkit)
#   - Activates the project virtual environment
#   - Sets environment variables (CCL_BENCH_HOME, TORCHTITAN_HOME, PYTHONPATH)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load Perlmutter modules
if command -v module &> /dev/null; then
	module load python 2> /dev/null || true
	module load cudatoolkit 2> /dev/null || true
fi

# Activate virtual environment
if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
	source "$PROJECT_ROOT/.venv/bin/activate"
	echo "Activated CCL-Bench environment"
else
	echo "Error: Virtual environment not found at $PROJECT_ROOT/.venv"
	echo "Run: ./perlmutter/setup_env.sh"
	return 1
fi

# Set useful environment variables
export CCL_BENCH_HOME="$PROJECT_ROOT"
export TORCHTITAN_HOME="${SCRATCH:-$HOME}/torchtitan"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "CCL_BENCH_HOME=$CCL_BENCH_HOME"
echo "TORCHTITAN_HOME=$TORCHTITAN_HOME"
