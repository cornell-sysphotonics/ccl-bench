# =============================================================================
# CCL-Bench Perlmutter-16 Workload Environment Configuration
# =============================================================================
# This file is sourced by install.sh, run.sh, and run.sbatch scripts
# to provide consistent environment configuration across all workloads.
#
# Usage:
#   source .env
#   # or automatically sourced by install.sh/run.sh/run.sbatch
# =============================================================================

# =============================================================================
# NERSC/Perlmutter Paths
# =============================================================================

# SCRATCH is set by NERSC environment, fallback to home if not available
export SCRATCH="${SCRATCH:-${HOME}/scratch}"

# =============================================================================
# UV Package Manager Configuration
# =============================================================================

# Store UV cache on scratch for better performance and quota management
export UV_CACHE_DIR="${SCRATCH}/cache/uv"

# Use copy mode for better reliability across compute nodes
export UV_LINK_MODE=copy

# =============================================================================
# HuggingFace Configuration
# =============================================================================

# HuggingFace cache and home directory (token stored at HF_HOME/token)
export HF_HOME="${SCRATCH}/cache/huggingface"

# Directory where model assets (tokenizer, config) are downloaded
export HF_ASSETS_ROOT="${SCRATCH}/ccl-bench-assets/models"

# =============================================================================
# Python Environment
# =============================================================================

# Virtual environment location
export VENV_DIR="${SCRATCH}/ccl-bench-venv"

# =============================================================================
# Trace Collection
# =============================================================================

# Base directory for all collected traces (nsys, torch profiler, etc.)
export TRACE_BASE="${SCRATCH}/ccl-bench-traces"
