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
VENV_DIR="${PROJECT_ROOT}/.venv"
TORCHTITAN_DIR="${SCRATCH:-$HOME}/torchtitan"

# Python version requirement
PYTHON_MIN_VERSION="3.10.12"

# PyTorch version (CUDA 12.4 for Perlmutter A100s)
PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"

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
if command -v module &> /dev/null; then
	info "Loading Perlmutter modules..."
	module load python 2> /dev/null || true
	module load cudatoolkit 2> /dev/null || true
	success "Modules loaded"
fi

# =============================================================================
# Python Environment Setup
# =============================================================================

check_python_version() {
	local python_cmd=$1
	local version
	version=$($python_cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2> /dev/null)
	if [[ -z $version ]]; then
		return 1
	fi
	# Compare versions
	if [[ "$(printf '%s\n' "$PYTHON_MIN_VERSION" "$version" | sort -V | head -n1)" == "$PYTHON_MIN_VERSION" ]]; then
		return 0
	fi
	return 1
}

setup_with_uv() {
	info "Setting up environment with uv..."

	# Check if uv is available
	if ! command -v uv &> /dev/null; then
		info "Installing uv..."
		curl -LsSf https://astral.sh/uv/install.sh | sh
		export PATH="$HOME/.local/bin:$PATH"
	fi

	success "uv version: $(uv --version)"

	# Create venv and install dependencies
	cd "$PROJECT_ROOT"

	info "Creating virtual environment with uv..."
	uv venv "$VENV_DIR" --python "$PYTHON_MIN_VERSION"

	# shellcheck disable=SC1091
	source "$VENV_DIR/bin/activate"

	uv sync

	success "Environment setup complete via uv."
}

setup_with_venv() {
	info "Setting up environment with standard venv..."

	# Find suitable Python
	local python_cmd=""
	for cmd in python3.10 python3.11 python3; do
		if command -v "$cmd" &> /dev/null && check_python_version "$cmd"; then
			python_cmd="$cmd"
			break
		fi
	done

	if [[ -z $python_cmd ]]; then
		error "No suitable Python >= ${PYTHON_MIN_VERSION} found"
	fi

	info "Using Python: $python_cmd ($($python_cmd --version))"

	# Create virtual environment
	info "Creating virtual environment..."
	$python_cmd -m venv "$VENV_DIR"

	# Activate and install
	source "$VENV_DIR/bin/activate"

	info "Upgrading pip..."
	pip install --upgrade pip

	info "Installing PyTorch with CUDA 12.4..."
	pip install torch torchvision --index-url "$PYTORCH_INDEX"

	info "Installing project dependencies..."
	pip install -e "$PROJECT_ROOT"

	success "venv environment setup complete"
}

# Try uv first, fall back to venv
if command -v uv &> /dev/null || [[ -f "$HOME/.local/bin/uv" ]]; then
	setup_with_uv
else
	warn "uv not found, attempting to install..."
	if curl -LsSf https://astral.sh/uv/install.sh | sh 2> /dev/null; then
		export PATH="$HOME/.local/bin:$PATH"
		setup_with_uv
	else
		warn "Could not install uv, falling back to standard venv"
		setup_with_venv
	fi
fi

# =============================================================================
# Model Weights Download (Optional)
# =============================================================================

if [[ $DOWNLOAD_MODELS == true ]]; then
	info "Downloading model weights..."

	ASSETS_DIR="${PROJECT_ROOT}/assets/hf"
	mkdir -p "$ASSETS_DIR"

	# Determine which HF CLI to use (prefer uvx hf, then hf, then huggingface-cli)
	HF_CLI=""
	if command -v uv &> /dev/null; then
		HF_CLI="uvx hf"
		info "Using: uvx hf (recommended)"
	elif command -v hf &> /dev/null; then
		HF_CLI="hf"
		info "Using: hf CLI"
	elif command -v huggingface-cli &> /dev/null; then
		HF_CLI="huggingface-cli"
		info "Using: huggingface-cli (legacy)"
	fi

	if [[ -n $HF_CLI ]]; then
		# Check if logged in
		if ! $HF_CLI auth whoami &> /dev/null; then
			warn "Not logged in to Hugging Face. Run: $HF_CLI auth login"
			warn "Some models (like LLaMA) require authentication."
		fi

		info "Downloading LLaMA-3.1-8B..."
		$HF_CLI download meta-llama/Llama-3.1-8B --local-dir "$ASSETS_DIR/Llama-3.1-8B" || warn "LLaMA download failed (may need HF token and model access)"

		info "Downloading DeepSeek-V2-Lite..."
		$HF_CLI download deepseek-ai/DeepSeek-V2-Lite --local-dir "$ASSETS_DIR/DeepSeek-V2-Lite" || warn "DeepSeek download failed"

		info "Downloading Qwen3-32B..."
		$HF_CLI download Qwen/Qwen3-32B --local-dir "$ASSETS_DIR/Qwen3-32B" || warn "Qwen download failed"
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
echo "TorchTitan location:  ${TORCHTITAN_DIR}"
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
	echo "Run with --with-models to download, or manually place weights in:"
	echo "  ${PROJECT_ROOT}/assets/hf/"
	echo ""
fi
