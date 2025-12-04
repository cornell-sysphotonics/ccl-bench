#!/bin/bash
# =============================================================================
# Perlmutter Environment Setup for CCL-Bench + TorchTitan
# =============================================================================
# This script sets up the Python environment for running TorchTitan training
# jobs on Perlmutter with trace collection.
#
# Supports both uv (preferred) and standard venv as fallback.
#
# Usage:
#   ./perlmutter/setup_env.sh [--with-models]
#
# Options:
#   --with-models    Also download HuggingFace model weights
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() {
	echo -e "${RED}[ERROR]${NC} $*"
	exit 1
}
success() { echo -e "${GREEN}[OK]${NC} $*"; }

# =============================================================================
# Configuration
# =============================================================================

# Detect script location and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Environment paths
VENV_DIR="${SCRATCH:-$PROJECT_ROOT}/ccl-bench-venv"
# Parse arguments
DOWNLOAD_MODELS=false
while [[ $# -gt 0 ]]; do
	case $1 in
		--with-models)
			DOWNLOAD_MODELS=true
			shift
			;;
		*)
			warn "Unknown option: $1"
			shift
			;;
	esac
done

# =============================================================================
# Environment Detection
# =============================================================================

info "Setting up CCL-Bench environment on Perlmutter..."
info "Project root: ${PROJECT_ROOT}"

# Check if we're on Perlmutter
if [[ -n ${NERSC_HOST:-} ]]; then
	info "Detected NERSC host: ${NERSC_HOST}"
else
	warn "Not running on NERSC - some features may not work"
fi

# Load required modules on Perlmutter
# Note: Temporarily disable 'set -u' because module/conda scripts have unbound variables
if command -v module &> /dev/null; then
	info "Loading Perlmutter modules..."
	set +u  # Disable unbound variable check for module commands
	module load python 2> /dev/null || true
	module load cudatoolkit 2> /dev/null || true
	set -u  # Re-enable unbound variable check
	success "Modules loaded"
fi

# =============================================================================
# Python Environment Setup
# =============================================================================

setup_with_uv() {
	info "Setting up environment with uv..."

	# Check if uv is available
	if ! command -v uv &> /dev/null; then
		info "Installing uv..."
		curl -LsSf https://astral.sh/uv/install.sh | sh
		export PATH="$HOME/.local/bin:$PATH"
	fi

	success "uv version: $(uv --version)"

	export UV_LINK_MODE=copy

	# Create venv and install dependencies
	cd "$PROJECT_ROOT"

	info "Creating virtual environment with uv..."
	uv venv "$VENV_DIR"

	# shellcheck disable=SC1091
	source "$VENV_DIR/bin/activate"

	# Use --active to sync to the activated venv (not the default .venv)
	uv sync --active

	success "Environment setup complete via uv."
}

# Try uv first, fall back to venv
if command -v uv &> /dev/null || [[ -f "$HOME/.local/bin/uv" ]]; then
	setup_with_uv
else
	warn "uv not found, attempting to install..."
	if curl -LsSf https://astral.sh/uv/install.sh | sh 2> /dev/null; then
		export PATH="$HOME/.local/bin:$PATH"
		setup_with_uv
	fi
fi

# =============================================================================
# Model Weights Download (Optional)
# =============================================================================

if [[ $DOWNLOAD_MODELS == true ]]; then
	info "Downloading model weights..."

	# Root for models + HF cache on SCRATCH (recommended by NERSC)
	export HF_HOME="$SCRATCH/cache/huggingface"
	export HF_ASSETS_ROOT="$SCRATCH/ccl-bench-assets/models"
	mkdir -p "$HF_HOME" "$HF_ASSETS_ROOT"

	# Determine which HF CLI to use (prefer uvx hf, then hf, then huggingface-cli)
	HF_CLI=""
	if command -v uv &> /dev/null; then
		HF_CLI="uvx hf"
		info "Using: uvx hf (recommended)"
	elif command -v hf &> /dev/null; then
		HF_CLI="hf"
		info "Using: hf CLI"
	fi

	if [[ -n $HF_CLI ]]; then
		# Check if logged in
		if ! $HF_CLI auth whoami &> /dev/null; then
			warn "Not logged in to Hugging Face. Run: $HF_CLI auth login"
			warn "Some models (like LLaMA) require authentication."
		fi

		info "Downloading LLaMA-3.1-8B..."
		$HF_CLI download meta-llama/Llama-3.1-8B \
		  --local-dir "$HF_ASSETS_ROOT/Llama-3.1-8B" \
			|| warn "LLaMA download failed (may need HF token and model access)"

		info "Downloading DeepSeek-V2-Lite..."
		$HF_CLI download deepseek-ai/DeepSeek-V2-Lite \
		  --local-dir "$HF_ASSETS_ROOT/DeepSeek-V2-Lite" \
			|| warn "DeepSeek download failed"

		info "Downloading Qwen3-32B..."
		$HF_CLI download Qwen/Qwen3-32B  \
		  --local-dir "$HF_ASSETS_ROOT/Qwen3-32B" \
			|| warn "Qwen download failed"
	else
		warn "No Hugging Face CLI found."
		warn "Install with one of:"
		warn "  curl -LsSf https://hf.co/cli/install.sh | bash  (recommended)"
		warn "  pip install huggingface_hub"
		warn "Then run: hf auth login"
	fi
fi

# =============================================================================
# Summary
# =============================================================================

ACTIVATE_SCRIPT="${PROJECT_ROOT}/perlmutter/activate.sh"

echo ""
echo "============================================================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "============================================================================="
echo ""
echo "Environment location: ${VENV_DIR}"
echo ""
echo "To activate the environment:"
echo "  source ${ACTIVATE_SCRIPT}"
echo ""
echo "To run a training job:"
echo "  sbatch perlmutter/run_llama3_8b_tp.sbatch"
echo ""
echo "To analyze traces:"
echo "  ccl-metrics --trace <trace_dir> --metric <metric_name>"
echo ""
if [[ $DOWNLOAD_MODELS == false ]]; then
	echo -e "${YELLOW}Note: Model weights were not downloaded.${NC}"
	echo "Run with --with-models to download"
fi
