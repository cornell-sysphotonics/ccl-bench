#!/bin/bash
# =============================================================================
# Common Configuration for CCL-Bench Perlmutter Jobs
# =============================================================================
# This file is sourced by all Slurm batch scripts to provide shared
# configuration and helper functions.
#
# IMPORTANT: Set your NERSC allocation below before running jobs!
# =============================================================================

# =============================================================================
# USER CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# Your NERSC allocation (REQUIRED - get this from `sacctmgr show assoc user=$USER`)
export NERSC_ALLOCATION="m4999"

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Auto-detect paths based on script location
if [[ -n ${SLURM_SUBMIT_DIR:-} ]]; then
	# Running under Slurm - check if submitted from perlmutter/ or project root
	if [[ -f "${SLURM_SUBMIT_DIR}/common.sh" ]]; then
		SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
	else
		SCRIPT_DIR="${SLURM_SUBMIT_DIR}/perlmutter"
	fi
else
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

CCL_BENCH_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"
export CCL_BENCH_HOME
export VENV_DIR="${SCRATCH:-$CCL_BENCH_HOME}/ccl-bench-venv"

# Trace output directory base
# Use $SCRATCH for traces to avoid filling home/project directories
# Nsys .qdrep/.nsys-rep files and Torch profiler traces can be large
export TRACE_BASE="${SCRATCH:-$CCL_BENCH_HOME}/ccl-bench-traces"

# Train config directory
export TRAIN_CONFIG_DIR="${CCL_BENCH_HOME}/train_configs"

# HuggingFace model assets directory (where downloaded models are stored)
export HF_ASSETS_ROOT="${SCRATCH:-$CCL_BENCH_HOME}/ccl-bench-assets/models"

# =============================================================================
# MODEL PATH RESOLUTION
# =============================================================================

# Get the HF assets path for a model based on config file name
# This maps config files to the correct model directory
# Usage: get_model_hf_path <config_file>
get_model_hf_path() {
	local config_file="$1"
	local config_name
	config_name=$(basename "$config_file" .toml)

	# Map config names to model directories
	case "$config_name" in
		llama3_8b_*)
			echo "${HF_ASSETS_ROOT}/Llama-3.1-8B"
			;;
		deepseek_v2_lite_*)
			echo "${HF_ASSETS_ROOT}/DeepSeek-V2-Lite"
			;;
		qwen3_32b_*)
			echo "${HF_ASSETS_ROOT}/Qwen3-32B"
			;;
		*)
			# Fallback: try to extract from config file
			echo "ERROR: Unknown model config: $config_name" >&2
			echo "Please add a mapping in common.sh get_model_hf_path()" >&2
			return 1
			;;
	esac
}

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

setup_runtime_paths() {
	# Load required modules
	# Note: Temporarily disable 'set -u' if active, because module/conda scripts have unbound variables
	local u_was_set=false
	if [[ $- == *u* ]]; then
		u_was_set=true
		set +u
	fi
	module load python 2> /dev/null || true
	module load cudatoolkit/12.9 2> /dev/null || true
	if [[ $u_was_set == true ]]; then
		set -u
	fi

	# Activate virtual environment
	if [[ -f "${VENV_DIR}/bin/activate" ]]; then
		source "${VENV_DIR}/bin/activate"
	else
		echo "ERROR: Virtual environment not found at ${VENV_DIR}"
		echo "Run: ./perlmutter/setup_env.sh"
		exit 1
	fi

	# Add project to Python path
	export PYTHONPATH="${CCL_BENCH_HOME}:${PYTHONPATH:-}"

	# Set cache directories on SCRATCH for better performance
	export HF_HOME="${SCRATCH:-$HOME}/cache/huggingface"
	export XDG_CACHE_HOME="${SCRATCH:-$HOME}/.cache"
	export TORCH_EXTENSIONS_DIR="${SCRATCH:-$HOME}/.torch_extensions"
	export CUDA_CACHE_PATH="${SCRATCH:-$HOME}/.nv/ComputeCache"
	mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH"
}

# =============================================================================
# DISTRIBUTED TRAINING CONFIGURATION
# =============================================================================

setup_distributed() {
	# Get master address from Slurm
	MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
	export MASTER_ADDR
	export MASTER_PORT="${MASTER_PORT:-29500}"

	# Calculate world size
	export WORLD_SIZE=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))

	# OpenMP threads (conservative to avoid oversubscription)
	export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

	# NCCL configuration for Perlmutter Slingshot network
	export NCCL_NET_GDR_LEVEL=PHB
	export NCCL_IB_QPS_PER_CONNECTION=4

	# Enable NCCL debug output (set to WARN or INFO for debugging)
	export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

	echo "=========================================="
	echo "Distributed Configuration:"
	echo "  MASTER_ADDR:  ${MASTER_ADDR}"
	echo "  MASTER_PORT:  ${MASTER_PORT}"
	echo "  WORLD_SIZE:   ${WORLD_SIZE}"
	echo "  NUM_NODES:    ${SLURM_JOB_NUM_NODES}"
	echo "  GPUS_PER_NODE: ${SLURM_GPUS_PER_NODE}"
	echo "=========================================="
}

# =============================================================================
# PROFILING MODE CONFIGURATION
# =============================================================================
# Controls what kind of profiling traces to collect:
#   "both"          - Nsys + Torch Profiler (default, may hit CUPTI conflicts)
#   "nsys"          - Nsys only (disables Torch Profiler via CLI override)
#   "torch"         - Torch Profiler only (no Nsys wrapper)
#
# If you see CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED, run two passes:
#   1. PROFILE_MODE="nsys" for Nsight traces
#   2. PROFILE_MODE="torch" for Torch ET + Kineto traces
# =============================================================================

export PROFILE_MODE="${PROFILE_MODE:-both}"

# =============================================================================
# TRACE COLLECTION SETUP
# =============================================================================

setup_trace_dir() {
	local workload_name="$1"

	export TRACE_DIR="${TRACE_BASE}/${workload_name}"
	mkdir -p "${TRACE_DIR}"

	echo "Trace output directory: ${TRACE_DIR}"
	echo "Profiling mode: ${PROFILE_MODE}"

	# Create a run timestamp for unique trace names
	RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
	export RUN_TIMESTAMP
}

# =============================================================================
# POST-JOB TRACE PROCESSING
# =============================================================================

# Copy/symlink rank-0 traces to a consistent location for metric tools
# TorchTitan outputs traces to profile_trace/ with varying names
copy_kineto_traces() {
	local trace_dir="${TRACE_DIR}"
	local profile_trace_dir="${trace_dir}/profile_trace"

	echo "Organizing trace files for analysis..."

	# Find the first rank-0 trace file (various naming conventions)
	local rank0_trace=""

	# Check TorchTitan's profile_trace directory first
	if [[ -d ${profile_trace_dir} ]]; then
		# Look for rank 0 trace with common patterns
		for pattern in "*rank0*.json" "*local_rank0*.json" "*trace_0.json"; do
			rank0_trace=$(find "${profile_trace_dir}" -maxdepth 2 -name "${pattern}" -type f 2> /dev/null | head -n 1)
			if [[ -n ${rank0_trace} ]]; then
				break
			fi
		done

		# If no rank-specific file found, take the first JSON trace
		if [[ -z ${rank0_trace} ]]; then
			rank0_trace=$(find "${profile_trace_dir}" -maxdepth 2 -name "*.json" -type f 2> /dev/null | head -n 1)
		fi
	fi

	# Create symlink for backward compatibility with metric tools
	if [[ -n ${rank0_trace} && -f ${rank0_trace} ]]; then
		local dest="${trace_dir}/kineto_trace_0.json"
		if [[ ! -e ${dest} ]]; then
			ln -sf "${rank0_trace}" "${dest}"
			echo "  Linked: ${rank0_trace} -> kineto_trace_0.json"
		fi
	else
		echo "  Note: No rank-0 trace found in profile_trace/"
	fi

	# List all trace files for reference
	echo ""
	echo "Trace files in ${trace_dir}:"
	find "${trace_dir}" -name "*.json" -o -name "*.nsys-rep" 2> /dev/null | head -20 || true
}

# =============================================================================
# NSYS PROFILING CONFIGURATION
# =============================================================================

# Verify the TOML config file exists and print a summary
# Usage: verify_config <config_file>
verify_config() {
	local config_file="$1"

	if [[ ! -f ${config_file} ]]; then
		echo "ERROR: Config file not found: ${config_file}"
		echo "Available configs in ${TRAIN_CONFIG_DIR}:"
		ls -1 "${TRAIN_CONFIG_DIR}"/*.toml 2> /dev/null || echo "  (none found)"
		exit 1
	fi

	echo "Config file: ${config_file}"

	# Extract and display parallelism settings if grep/awk available
	if command -v grep &> /dev/null; then
		echo "Parallelism settings:"
		grep -E "^(data_parallel|tensor_parallel|pipeline_parallel|enable_async)" "${config_file}" 2> /dev/null | sed 's/^/  /' || true
	fi
}

# NSys profile command wrapper
# Usage: nsys_profile <output_prefix> <command...>
nsys_profile() {
	local output_prefix="$1"
	shift

	nsys profile \
		--stats=true \
		--trace=cuda,osrt \
		--cuda-memory-usage=true \
		--gpuctxsw=true \
		--output="${TRACE_DIR}/${output_prefix}_${RUN_TIMESTAMP}" \
		--force-overwrite=true \
		"$@"
}

# =============================================================================
# TORCHRUN LAUNCHER
# =============================================================================

# Launch distributed training with torchrun
# Usage: launch_torchtitan <config_file> [extra_args...]
# Respects PROFILE_MODE for Torch Profiler control
launch_torchtitan() {
	local config_file="$1"
	shift
	local extra_args=("$@")

	if [[ ! -f ${config_file} ]]; then
		echo "ERROR: Config file not found: ${config_file}"
		exit 1
	fi

	# Get the correct model path for this config
	local model_path
	model_path=$(get_model_hf_path "${config_file}")
	if [[ $? -ne 0 ]]; then
		exit 1
	fi

	echo "Launching TorchTitan with config: ${config_file}"
	echo "  Model path: ${model_path}"

	# Build profiler args based on PROFILE_MODE
	local profiler_args=()
	if [[ ${PROFILE_MODE} == "nsys" ]]; then
		# Disable Torch Profiler when running Nsys-only
		profiler_args+=(--profiling.enable_profiling False)
		echo "  Torch Profiler: DISABLED (nsys-only mode)"
	else
		echo "  Torch Profiler: ENABLED"
	fi

	python -m torch.distributed.run \
		--nnodes="${SLURM_JOB_NUM_NODES}" \
		--nproc-per-node="${SLURM_GPUS_PER_NODE}" \
		--rdzv-backend=c10d \
		--rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
		-m torchtitan.train \
		--job.config_file "${config_file}" \
		--job.dump_folder "${TRACE_DIR}" \
		--model.hf_assets_path "${model_path}" \
		"${profiler_args[@]}" \
		"${extra_args[@]}"
}

# Launch with NSys profiling
# Usage: launch_torchtitan_with_nsys <config_file> <nsys_output_prefix>
# Respects PROFILE_MODE: "both" wraps with nsys, "torch" skips nsys
launch_torchtitan_with_nsys() {
	local config_file="$1"
	local nsys_prefix="$2"

	if [[ ! -f ${config_file} ]]; then
		echo "ERROR: Config file not found: ${config_file}"
		exit 1
	fi

	# Get the correct model path for this config
	local model_path
	model_path=$(get_model_hf_path "${config_file}")
	if [[ $? -ne 0 ]]; then
		exit 1
	fi

	# Build profiler args based on PROFILE_MODE
	local profiler_args=()
	if [[ ${PROFILE_MODE} == "nsys" ]]; then
		# Disable Torch Profiler when running Nsys-only
		profiler_args+=(--profiling.enable_profiling False)
	fi

	# Handle different profiling modes
	case "${PROFILE_MODE}" in
		torch)
			# Torch Profiler only - no Nsys wrapper
			echo "Launching TorchTitan with Torch Profiler only (no Nsys)..."
			echo "  Config: ${config_file}"
			echo "  Model path: ${model_path}"

			python -m torch.distributed.run \
				--nnodes="${SLURM_JOB_NUM_NODES}" \
				--nproc-per-node="${SLURM_GPUS_PER_NODE}" \
				--rdzv-backend=c10d \
				--rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
				-m torchtitan.train \
				--job.config_file "${config_file}" \
				--job.dump_folder "${TRACE_DIR}" \
				--model.hf_assets_path "${model_path}"
			;;
		nsys)
			# Nsys only - disable Torch Profiler
			echo "Launching TorchTitan with Nsys only (Torch Profiler disabled)..."
			echo "  Config: ${config_file}"
			echo "  Model path: ${model_path}"
			echo "  NSys output: ${TRACE_DIR}/${nsys_prefix}_${RUN_TIMESTAMP}"

			nsys_profile "${nsys_prefix}" \
				python -m torch.distributed.run \
				--nnodes="${SLURM_JOB_NUM_NODES}" \
				--nproc-per-node="${SLURM_GPUS_PER_NODE}" \
				--rdzv-backend=c10d \
				--rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
				-m torchtitan.train \
				--job.config_file "${config_file}" \
				--job.dump_folder "${TRACE_DIR}" \
				--model.hf_assets_path "${model_path}" \
				--profiling.enable_profiling False
			;;
		both|*)
			# Both Nsys + Torch Profiler (default)
			echo "Launching TorchTitan with Nsys + Torch Profiler..."
			echo "  Config: ${config_file}"
			echo "  Model path: ${model_path}"
			echo "  NSys output: ${TRACE_DIR}/${nsys_prefix}_${RUN_TIMESTAMP}"
			echo ""
			echo "  NOTE: If you see CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED,"
			echo "        run two separate passes with PROFILE_MODE=nsys and PROFILE_MODE=torch"

			nsys_profile "${nsys_prefix}" \
				python -m torch.distributed.run \
				--nnodes="${SLURM_JOB_NUM_NODES}" \
				--nproc-per-node="${SLURM_GPUS_PER_NODE}" \
				--rdzv-backend=c10d \
				--rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
				-m torchtitan.train \
				--job.config_file "${config_file}" \
				--job.dump_folder "${TRACE_DIR}" \
				--model.hf_assets_path "${model_path}"
			;;
	esac
}

# =============================================================================
# JOB SUMMARY
# =============================================================================

print_job_summary() {
	local workload_name="$1"
	local config_file="$2"

	echo ""
	echo "============================================================================="
	echo "CCL-Bench Training Job: ${workload_name}"
	echo "============================================================================="
	echo "Job ID:         ${SLURM_JOB_ID}"
	echo "Job Name:       ${SLURM_JOB_NAME}"
	echo "Nodes:          ${SLURM_JOB_NUM_NODES}"
	echo "GPUs per Node:  ${SLURM_GPUS_PER_NODE}"
	echo "Total GPUs:     ${WORLD_SIZE}"
	echo "Trace Dir:      ${TRACE_DIR}"
	echo "Start Time:     $(date)"
	echo "-----------------------------------------------------------------------------"

	# Verify config and show parallelism settings
	verify_config "${config_file}"

	echo "============================================================================="
	echo ""
}

print_job_complete() {
	# Organize traces for metric tools
	copy_kineto_traces

	echo ""
	echo "============================================================================="
	echo "Job Complete: ${SLURM_JOB_NAME}"
	echo "End Time:     $(date)"
	echo "Traces saved to: ${TRACE_DIR}"
	echo "============================================================================="
	echo ""
	echo "To analyze traces, run:"
	echo "  source perlmutter/activate.sh"
	echo "  ccl-metrics --trace ${TRACE_DIR} --metric coll_call_num"
	echo "  ccl-metrics --trace ${TRACE_DIR} --metric throughput_tokens"
	echo ""
}
