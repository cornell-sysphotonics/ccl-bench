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
#   - Sets environment variables (CCL_BENCH_HOME, PYTHONPATH)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${SCRATCH:-$PROJECT_ROOT}/ccl-bench-venv"

# Load Perlmutter modules
# Note: Temporarily disable 'set -u' if active, because module/conda scripts have unbound variables
_u_was_set=false
if [[ $- == *u* ]]; then
	_u_was_set=true
	set +u
fi

if command -v module &> /dev/null; then
	module load python 2> /dev/null || true
	module load cudatoolkit/12.9 2> /dev/null || true
fi

if [[ $_u_was_set == true ]]; then
	set -u
fi
unset _u_was_set

# Activate virtual environment
if [[ -f "$VENV_DIR/bin/activate" ]]; then
	source "$VENV_DIR/bin/activate"
	echo "Activated CCL-Bench environment"
else
	echo "Error: Virtual environment not found at $VENV_DIR"
	echo "Run: ./perlmutter/setup_env.sh"
	return 1
fi

# Set useful environment variables
export CCL_BENCH_HOME="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Set cache directories on SCRATCH for better performance
export HF_HOME="${SCRATCH:-$HOME}/cache/huggingface"
export XDG_CACHE_HOME="${SCRATCH:-$HOME}/.cache"
export TORCH_EXTENSIONS_DIR="${SCRATCH:-$HOME}/.torch_extensions"
export CUDA_CACHE_PATH="${SCRATCH:-$HOME}/.nv/ComputeCache"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" 2>/dev/null || true

echo "CCL_BENCH_HOME=$CCL_BENCH_HOME"
echo "VENV_DIR=$VENV_DIR"
