#!/bin/bash
# =============================================================================
# Perlmutter-16 Workload Install Script
# =============================================================================
# This script sets up the environment for running torchtitan training workloads
# on Perlmutter. It handles:
#   - UV package manager installation/update
#   - Python virtual environment creation
#   - HuggingFace authentication check
#   - Model asset downloads (tokenizer, config)
#
# Usage:
#   ./install.sh              # Install everything
#   ./install.sh --skip-model # Skip model download
#
# This script should be run from the workload directory (e.g., llama3.1-8b-pp)
# =============================================================================

set -euo pipefail

# =============================================================================
# Colors and Output Helpers
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }

# =============================================================================
# Path Configuration
# =============================================================================

WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${WORKLOAD_DIR}/../.." && pwd)"
WORKLOAD_NAME="$(basename "${WORKLOAD_DIR}")"

# Source env.sh file to get shared configuration
if [[ -f "${WORKLOAD_DIR}/env.sh" ]]; then
	# shellcheck disable=SC1091
	source "${WORKLOAD_DIR}/env.sh"
fi

# Set defaults if not already set by env.sh
export SCRATCH="${SCRATCH:-${HOME}/scratch}"
export HF_HOME="${HF_HOME:-${SCRATCH}/cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH}/cache/uv}"
export HF_ASSETS_ROOT="${HF_ASSETS_ROOT:-${SCRATCH}/ccl-bench-assets/models}"
export VENV_DIR="${VENV_DIR:-${SCRATCH}/ccl-bench-venv}"

# =============================================================================
# Argument Parsing
# =============================================================================

SKIP_MODEL=false
while [[ $# -gt 0 ]]; do
	case $1 in
		--skip-model)
			SKIP_MODEL=true
			shift
			;;
		-h | --help)
			echo "Usage: $0 [--skip-model]"
			echo "  --skip-model  Skip downloading model assets"
			exit 0
			;;
		*)
			warn "Unknown option: $1"
			shift
			;;
	esac
done

# =============================================================================
# System Checks
# =============================================================================

info "=========================================="
info "CCL-Bench Workload Installation"
info "=========================================="
info "Workload:     ${WORKLOAD_NAME}"
info "Workload Dir: ${WORKLOAD_DIR}"
info "Project Root: ${PROJECT_ROOT}"
info ""

# Check if we're on Perlmutter/NERSC
if [[ -n ${NERSC_HOST:-} ]]; then
	info "Detected NERSC host: ${NERSC_HOST}"
else
	warn "Not running on NERSC - some features may not work"
fi

# =============================================================================
# Load Modules (Perlmutter)
# =============================================================================

if command -v module &> /dev/null; then
	info "Loading Perlmutter modules..."
	# Disable unbound variable check for module commands (they have unset vars)
	set +u
	module load cudatoolkit/12.9 2> /dev/null || true
	set -u
	success "Modules loaded (cudatoolkit)"
fi

# =============================================================================
# UV Installation and Update
# =============================================================================

info "Checking UV package manager..."

# Create cache directory
mkdir -p "${UV_CACHE_DIR}"
export UV_CACHE_DIR

if command -v uv &> /dev/null; then
	info "UV found, checking for updates..."
	uv self update || warn "Could not update UV (may require manual update)"
	# shellcheck disable=SC2312
	success "UV version: $(uv --version)"
elif [[ -f "${HOME}/.local/bin/uv" ]]; then
	export PATH="${HOME}/.local/bin:${PATH}"
	info "UV found in ~/.local/bin, checking for updates..."
	uv self update || warn "Could not update UV"
	# shellcheck disable=SC2312
	success "UV version: $(uv --version)"
else
	info "Installing UV package manager..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="${HOME}/.local/bin:${PATH}"
	# shellcheck disable=SC2312
	success "UV installed: $(uv --version)"
fi

# =============================================================================
# Python Virtual Environment Setup
# =============================================================================

info "Setting up Python virtual environment..."

# Create directories
mkdir -p "${HF_HOME}" "${HF_ASSETS_ROOT}" "${UV_CACHE_DIR}"

cd "${WORKLOAD_DIR}"

if [[ -d ${VENV_DIR}   ]]; then
	info "Virtual environment exists at ${VENV_DIR}"
	info "Syncing dependencies..."
else
	info "Creating virtual environment at ${VENV_DIR}..."
	uv venv "${VENV_DIR}"
fi

# Activate and sync
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
uv sync --active

success "Python environment ready"

# =============================================================================
# HuggingFace Authentication Check
# =============================================================================

info "Checking HuggingFace authentication..."

check_hf_login() {
	# Try to check via uvx hf auth whoami
	if uvx hf auth whoami &> /dev/null; then
		return 0
	fi
	return 1
}

check_hf_login
hf_login_status=$?
if [[ ${hf_login_status} -eq 0 ]]; then
	success "HuggingFace authentication found"
else
	warn "=========================================="
	warn "HuggingFace Authentication Required"
	warn "=========================================="
	warn ""
	warn "You are not logged in to HuggingFace."
	warn "Some models require authentication."
	warn ""
	warn "To log in, run:"
	warn "  uvx hf auth login"
	warn ""
	warn "Get your token from: https://huggingface.co/settings/tokens"
	warn "=========================================="

	if [[ ${SKIP_MODEL} == false ]]; then
		error "Cannot download models without HuggingFace authentication."
		error "Please log in first, or run with --skip-model"
		exit 1
	fi
fi

# =============================================================================
# Model Assets Download
# =============================================================================

if [[ ${SKIP_MODEL} == true ]]; then
	info "Skipping model download (--skip-model specified)"
else
	info "Downloading model assets..."

	# Parse the workload_card.yaml to get the HF URL
	WORKLOAD_CARD="${WORKLOAD_DIR}/workload_card.yaml"
	if [[ ! -f ${WORKLOAD_CARD} ]]; then
		error "workload_card.yaml not found at ${WORKLOAD_CARD}"
		exit 1
	fi

	# Extract hf_url from workload_card.yaml
	# Format: hf_url: https://huggingface.co/meta-llama/Llama-3.1-8B
	HF_URL=$(grep -E "^hf_url:" "${WORKLOAD_CARD}" | sed 's/hf_url:[[:space:]]*//' | tr -d ' ')

	if [[ -z ${HF_URL} || ${HF_URL} == "#"* ]]; then
		warn "No hf_url found in workload_card.yaml, skipping model download"
	else
		# Extract repo_id from URL (e.g., meta-llama/Llama-3.1-8B)
		REPO_ID="${HF_URL#https://huggingface.co/}"
		MODEL_NAME="${REPO_ID#*/}"

		info "Repository: ${REPO_ID}"
		info "Model name: ${MODEL_NAME}"
		info "Destination: ${HF_ASSETS_ROOT}/${MODEL_NAME}"

		# Check if model requires gated access (LLaMA, etc.)
		GATED_MODELS=("meta-llama")
		REPO_ORG="${REPO_ID%%/*}"
		IS_GATED=false
		for gated in "${GATED_MODELS[@]}"; do
			if [[ ${REPO_ORG} == "${gated}" ]]; then
				IS_GATED=true
				break
			fi
		done

		if [[ ${IS_GATED} == true ]]; then
			warn "=========================================="
			warn "This model requires access approval"
			warn "=========================================="
			warn ""
			warn "Model: ${REPO_ID}"
			warn ""
			warn "You must request access at:"
			warn "  ${HF_URL}"
			warn ""
			warn "After approval, ensure you're logged in:"
			warn "  uvx hf auth login"
			warn "=========================================="
		fi

		# Download tokenizer and config files using huggingface-hub CLI
		# Note: torchtitan's scripts/ folder is NOT included in the pip package
		info "Downloading tokenizer and config..."

		MODEL_DIR="${HF_ASSETS_ROOT}/${MODEL_NAME}"
		mkdir -p "${MODEL_DIR}"

		# Download tokenizer files (tokenizer.json, tokenizer_config.json, tokenizer.model, etc.)
		# and config files (config.json, generation_config.json)
		# Each pattern needs its own --include flag
		if uvx hf download "${REPO_ID}" \
			--local-dir "${MODEL_DIR}" \
			--include "tokenizer.json" \
			--include "tokenizer_config.json" \
			--include "tokenizer.model" \
			--include "vocab.txt" \
			--include "vocab.json" \
			--include "merges.txt" \
			--include "special_tokens_map.json" \
			--include "config.json" \
			--include "generation_config.json"; then
			success "Model assets downloaded to ${MODEL_DIR}"
		else
			if [[ ${IS_GATED} == true ]]; then
				error "=========================================="
				error "Model download failed!"
				error "=========================================="
				error ""
				error "This is likely because you don't have access to ${REPO_ID}"
				error ""
				error "Please:"
				error "1. Visit ${HF_URL} and request access"
				error "2. Wait for approval"
				error "3. Run this script again"
				error "=========================================="
				exit 1
			else
				warn "Download failed - check the errors above"
			fi
		fi
	fi
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=========================================="
echo -e "${GREEN}Installation Complete!${NC}"
echo "=========================================="
echo ""
echo "Environment:"
echo "  Virtual Environment: ${VENV_DIR}"
echo "  HF Home:            ${HF_HOME}"
echo "  HF Assets:          ${HF_ASSETS_ROOT}"
echo "  UV Cache:           ${UV_CACHE_DIR}"
echo ""
echo "To activate the environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To submit a training job:"
echo "  cd ${WORKLOAD_DIR}"
echo "  sbatch run.sbatch"
echo ""
echo "Or use the run.sh wrapper:"
echo "  ./run.sh"
echo ""
