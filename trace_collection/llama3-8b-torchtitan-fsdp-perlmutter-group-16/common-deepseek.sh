#!/bin/bash
# =============================================================================
# Common Functions for Perlmutter Workloads
# =============================================================================
# This file provides shared functions for running TorchTitan training workloads
# on NERSC Perlmutter. It is symlinked to each workload directory.
#
# Usage:
#   source common.sh
#
# Required: Source env.sh first to set environment variables.
# =============================================================================

# =============================================================================
# TRACE DIRECTORY SETUP
# =============================================================================

# Setup trace output directory for a workload
# Usage: setup_trace_dir <workload_dir>
#
# Creates the trace directory in the workload directory under a "traces" subdirectory.
# Sets TRACE_DIR and RUN_TIMESTAMP environment variables.
setup_trace_dir() {
	local workload_dir="$1"

	# Validate workload directory
	if [[ -z ${workload_dir} ]]; then
		echo "ERROR: workload_dir not provided to setup_trace_dir"
		return 1
	fi

	# Create trace directory in workload directory
	export TRACE_DIR="${workload_dir}/traces"
	mkdir -p "${TRACE_DIR}"

	# Create a run timestamp for unique trace names
	RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
	export RUN_TIMESTAMP

	echo "Trace output directory: ${TRACE_DIR}"
	echo "Run timestamp: ${RUN_TIMESTAMP}"
}

# =============================================================================
# NSYS PROFILING
# =============================================================================

# NSys profile command wrapper
# Usage: nsys_profile <output_prefix> <command...>
nsys_profile() {
	local output_prefix="$1"
	shift

	# Ensure TRACE_DIR and RUN_TIMESTAMP are set
	if [[ -z ${TRACE_DIR:-} ]]; then
		echo "ERROR: TRACE_DIR not set. Run setup_trace_dir first."
		return 1
	fi
	if [[ -z ${RUN_TIMESTAMP:-} ]]; then
		RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
		export RUN_TIMESTAMP
	fi

	local output_file="${TRACE_DIR}/${output_prefix}_${RUN_TIMESTAMP}"
	echo "NSys output: ${output_file}.nsys-rep"

	nsys profile \
		--stats=true \
		--trace=mpi,cuda,nvtx,osrt,openmp \
		--cuda-memory-usage=true \
		--mpi-impl=mpich \
		--output="${output_file}" \
		--force-overwrite=true \
		"$@"
}

# =============================================================================
# TORCHRUN LAUNCHER
# =============================================================================

# Launch distributed training with torchrun
# Usage: torchrun_launcher <command...>
#
# Automatically sets up distributed training using SLURM environment variables:
#   --nnodes:         Number of nodes from SLURM_JOB_NUM_NODES
#   --nproc-per-node: GPUs per node from SLURM_GPUS_PER_NODE
#   --rdzv-backend:   c10d rendezvous backend
#   --rdzv-endpoint:  MASTER_ADDR:MASTER_PORT from env.sh
#
# Example:
#   torchrun_launcher -m torchtitan.train --job.config_file config.toml
torchrun_launcher() {
	# Validate required environment variables
	if [[ -z ${SLURM_JOB_NUM_NODES:-} ]]; then
		echo "ERROR: SLURM_JOB_NUM_NODES not set. Are you running under Slurm?"
		return 1
	fi
	if [[ -z ${SLURM_GPUS_PER_NODE:-} ]]; then
		echo "ERROR: SLURM_GPUS_PER_NODE not set. Are you running under Slurm?"
		return 1
	fi
	if [[ -z ${MASTER_ADDR:-} ]]; then
		echo "ERROR: MASTER_ADDR not set. Source env.sh first."
		return 1
	fi
	if [[ -z ${MASTER_PORT:-} ]]; then
		echo "ERROR: MASTER_PORT not set. Source env.sh first."
		return 1
	fi

	echo "=========================================="
	echo "Distributed Training Configuration:"
	echo "  NNODES:         ${SLURM_JOB_NUM_NODES}"
	echo "  NPROC_PER_NODE: ${SLURM_GPUS_PER_NODE}"
	echo "  MASTER_ADDR:    ${MASTER_ADDR}"
	echo "  MASTER_PORT:    ${MASTER_PORT}"
	echo "=========================================="

	# Use srun to launch across all nodes for multi-node training
	srun --ntasks-per-node=1 \
		python -m torch.distributed.run \
		--nnodes="${SLURM_JOB_NUM_NODES}" \
		--nproc-per-node="${SLURM_GPUS_PER_NODE}" \
		--rdzv-backend=c10d \
		--rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
		"$@"
}

# =============================================================================
# TORCHTITAN LAUNCHER
# =============================================================================

# Get model HF assets path from workload_card.yaml
# Usage: get_model_hf_path <workload_dir>
#
# Extracts the hf_url from workload_card.yaml and constructs the local path:
#   hf_url: https://huggingface.co/meta-llama/Llama-3.1-8B
#   -> HF_ASSETS_ROOT/Llama-3.1-8B
#
# Returns the path to the model assets directory.
get_model_hf_path() {
	local workload_dir="$1"
	local yaml_file="${workload_dir}/workload_card.yaml"

	if [[ ! -f ${yaml_file} ]]; then
		echo "ERROR: workload_card.yaml not found at ${yaml_file}" >&2
		return 1
	fi

	if [[ -z ${HF_ASSETS_ROOT:-} ]]; then
		echo "ERROR: HF_ASSETS_ROOT not set. Source env.sh first." >&2
		return 1
	fi

	# Extract hf_url from workload_card.yaml
	# Format: hf_url: https://huggingface.co/meta-llama/Llama-3.1-8B
	local hf_url
	hf_url=$(grep -E "^hf_url:" "${yaml_file}" | sed 's/hf_url:[[:space:]]*//' | tr -d ' ' || true)

	if [[ -z ${hf_url} || ${hf_url} == "#"* ]]; then
		echo "ERROR: No hf_url found in ${yaml_file}" >&2
		return 1
	fi

	# Extract repo_id from URL (e.g., meta-llama/Llama-3.1-8B)
	local repo_id="${hf_url#https://huggingface.co/}"
	# Extract model name (last part after /)
	local model_name="${repo_id#*/}"

	echo "${HF_ASSETS_ROOT}/${model_name}"
}

# Launch TorchTitan training with configuration
# Usage: launch_torchtitan <workload_dir> <config_file> [extra_args...]
#
# Required environment variables (set by env.sh and Slurm):
#   TRACE_DIR:       Output directory for traces and checkpoints
#   HF_ASSETS_ROOT:  Directory containing HuggingFace model assets
#
# The function automatically:
#   - Sets --job.dump_folder to TRACE_DIR
#   - Sets --model.hf_assets_path from workload_card.yaml hf_url
#   - Passes through any extra arguments
#
# Example:
#   launch_torchtitan . train_config.toml
#   launch_torchtitan /path/to/workload train_config.toml --training.steps 100
launch_torchtitan() {
	local workload_dir="$1"
	local config_file="$2"
	shift 2
	local extra_args=("$@")

	# Validate config file exists
	if [[ ! -f ${config_file} ]]; then
		echo "ERROR: Config file not found: ${config_file}"
		return 1
	fi

	# Validate required environment variables
	if [[ -z ${TRACE_DIR:-} ]]; then
		echo "ERROR: TRACE_DIR not set. Run setup_trace_dir first."
		return 1
	fi

	# Get model path from workload_card.yaml
	local model_hf_path
	model_hf_path=$(get_model_hf_path "${workload_dir}") || return 1

	echo "=========================================="
	echo "TorchTitan Configuration:"
	echo "  Workload dir:   ${workload_dir}"
	echo "  Config file:    ${config_file}"
	echo "  HF assets:      ${model_hf_path}"
	echo "  Dump folder:    ${TRACE_DIR}"
	echo "=========================================="

	torchrun_launcher \
		-m torchtitan.train \
		--job.config_file "${config_file}" \
		--job.dump_folder "${TRACE_DIR}" \
		--model.hf_assets_path "${model_hf_path}" \
		"${extra_args[@]}"
}

# Launch TorchTitan with NSys profiling
# Usage: launch_torchtitan_nsys <nsys_prefix> <workload_dir> <config_file> [extra_args...]
#
# Wraps launch_torchtitan with nsys profiling.
# The nsys output will be saved to TRACE_DIR/<nsys_prefix>_<timestamp>.nsys-rep
#
# Example:
#   launch_torchtitan_nsys "llama3_8b_pp" . train_config.toml
launch_torchtitan_nsys() {
	local nsys_prefix="$1"
	local workload_dir="$2"
	local config_file="$3"
	shift 3
	local extra_args=("$@")

	# Validate config file exists
	if [[ ! -f ${config_file} ]]; then
		echo "ERROR: Config file not found: ${config_file}"
		return 1
	fi

	# Validate required environment variables
	if [[ -z ${TRACE_DIR:-} ]]; then
		echo "ERROR: TRACE_DIR not set. Run setup_trace_dir first."
		return 1
	fi

	# Get model path from workload_card.yaml
	local model_hf_path
	model_hf_path=$(get_model_hf_path "${workload_dir}") || return 1

	# Get the trace_collection directory (parent of common/) for custom model registration
	local common_dir
	common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	local trace_collection_dir
	trace_collection_dir="$(dirname "${common_dir}")"

	# Create a wrapper script that imports the custom model registration
	local wrapper_script="${TRACE_DIR}/train_wrapper.py"
	cat > "${wrapper_script}" << 'WRAPPER_EOF'
#!/usr/bin/env python3
"""Wrapper script to register custom models before running TorchTitan training."""
import runpy

# Import the registration module to register DeepSeek-V2-Lite model
# This auto-registers the model when imported (see register_deepseek_v2_lite.py)
import register_deepseek_v2_lite

# Run torchtitan.train as __main__
runpy.run_module('torchtitan.train', run_name='__main__')
WRAPPER_EOF

	echo "=========================================="
	echo "TorchTitan Configuration (with NSys):"
	echo "  Workload dir:   ${workload_dir}"
	echo "  Config file:    ${config_file}"
	echo "  HF assets:      ${model_hf_path}"
	echo "  Dump folder:    ${TRACE_DIR}"
	echo "  NSys prefix:    ${nsys_prefix}"
	echo "  Custom models:  ${trace_collection_dir}/register_deepseek_v2_lite.py"
	echo "=========================================="

	# Use srun to launch across all nodes for multi-node training
	# PYTHONPATH is set to include the trace_collection directory for custom model imports
	PYTHONPATH="${trace_collection_dir}:${PYTHONPATH:-}" \
		srun --ntasks-per-node=1 \
		nsys profile \
		--stats=true \
		--trace=mpi,cuda,nvtx,osrt,openmp \
		--cuda-memory-usage=true \
		--mpi-impl=mpich \
		--output="${TRACE_DIR}/${nsys_prefix}_${RUN_TIMESTAMP}_%h" \
		--force-overwrite=true \
		python -m torch.distributed.run \
		--nnodes="${SLURM_JOB_NUM_NODES}" \
		--nproc-per-node="${SLURM_GPUS_PER_NODE}" \
		--rdzv-backend=c10d \
		--rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
		"${wrapper_script}" \
		--job.config_file "${config_file}" \
		--job.dump_folder "${TRACE_DIR}" \
		--model.hf_assets_path "${model_hf_path}" \
		"${extra_args[@]}"
}

launch_torchtitan_without_nsys() {
	local nsys_prefix="$1"
	local workload_dir="$2"
	local config_file="$3"
	shift 3
	local extra_args=("$@")

	# Validate config file exists
	if [[ ! -f ${config_file} ]]; then
		echo "ERROR: Config file not found: ${config_file}"
		return 1
	fi

	# Validate required environment variables
	if [[ -z ${TRACE_DIR:-} ]]; then
		echo "ERROR: TRACE_DIR not set. Run setup_trace_dir first."
		return 1
	fi

	# Get model path from workload_card.yaml
	local model_hf_path
	model_hf_path=$(get_model_hf_path "${workload_dir}") || return 1

	# Get the trace_collection directory (parent of common/) for custom model registration
	local common_dir
	common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	local trace_collection_dir
	trace_collection_dir="$(dirname "${common_dir}")"

	# Create a wrapper script that imports the custom model registration
	local wrapper_script="${TRACE_DIR}/train_wrapper.py"
	cat > "${wrapper_script}" << 'WRAPPER_EOF'
#!/usr/bin/env python3
"""Wrapper script to register custom models before running TorchTitan training."""
import runpy

# Import the registration module to register DeepSeek-V2-Lite model
# This auto-registers the model when imported (see register_deepseek_v2_lite.py)
import register_deepseek_v2_lite

# Run torchtitan.train as __main__
runpy.run_module('torchtitan.train', run_name='__main__')
WRAPPER_EOF

	echo "=========================================="
	echo "TorchTitan Configuration (with NSys):"
	echo "  Workload dir:   ${workload_dir}"
	echo "  Config file:    ${config_file}"
	echo "  HF assets:      ${model_hf_path}"
	echo "  Dump folder:    ${TRACE_DIR}"
	echo "  NSys prefix:    ${nsys_prefix}"
	echo "  Custom models:  ${trace_collection_dir}/register_deepseek_v2_lite.py"
	echo "=========================================="

	# Use srun to launch across all nodes for multi-node training
	# PYTHONPATH is set to include the trace_collection directory for custom model imports
	PYTHONPATH="${trace_collection_dir}:${PYTHONPATH:-}" \
		srun --ntasks-per-node=1 \
		python -m torch.distributed.run \
		--nnodes="${SLURM_JOB_NUM_NODES}" \
		--nproc-per-node="${SLURM_GPUS_PER_NODE}" \
		--rdzv-backend=c10d \
		--rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
		"${wrapper_script}" \
		--job.config_file "${config_file}" \
		--job.dump_folder "${TRACE_DIR}" \
		--model.hf_assets_path "${model_hf_path}" \
		"${extra_args[@]}"
}

# =============================================================================
# JOB SUMMARY
# =============================================================================

# Print job summary at the start of a job
# Usage: print_job_summary
print_job_summary() {
	echo ""
	echo "============================================================================="
	echo "CCL-Bench Training Job"
	echo "============================================================================="
	echo "Job ID:         ${SLURM_JOB_ID:-N/A}"
	echo "Job Name:       ${SLURM_JOB_NAME:-N/A}"
	echo "Nodes:          ${SLURM_JOB_NUM_NODES:-N/A}"
	echo "GPUs per Node:  ${SLURM_GPUS_PER_NODE:-N/A}"
	echo "Total GPUs:     $((${SLURM_JOB_NUM_NODES:-1} * ${SLURM_GPUS_PER_NODE:-1}))"
	echo "Trace Dir:      ${TRACE_DIR:-N/A}"
	echo "Start Time:     $(date || true)"
	echo "============================================================================="
	echo ""
}

# Print job completion message
# Usage: print_job_complete
print_job_complete() {
	echo ""
	echo "============================================================================="
	echo "Job Complete"
	echo "End Time:     $(date || true)"
	echo "Traces saved to: ${TRACE_DIR:-N/A}"
	echo "============================================================================="
	echo ""
}

# =============================================================================
# METRICS COLLECTION
# =============================================================================

# All available metrics from the tools module
# Keep in sync with tools/main.py _METRIC_MODULES
METRICS=(
	"coll_call_num_16"
	"comm_comp_overlap_16"
	"comm_volume_16"
	"config_metadata_16"
	"hardware_saturation_16"
	"iter_time_16"
	"pipeline_bubble_16"
	"straggler_lag_16"
	"throughput_tokens_16"
	"traffic_distribution_16"
	"training_quality_16"
	"variability_metrics_16"
)

# Run all metrics and save JSON outputs
# Usage: run_all_metrics <trace_dir> [output_dir]
#
# Arguments:
#   trace_dir:  Path to the trace directory containing kineto/torch_et traces
#   output_dir: (Optional) Directory to save JSON outputs. Defaults to current directory.
#
# Output files are named: <metric_name>.json
#
# Example:
#   run_all_metrics /pscratch/sd/g/gb555/ccl-bench-traces/llama3_8b_fsdp
#   run_all_metrics /path/to/traces ./metrics_output
run_all_metrics() {
	local trace_dir="$1"
	local output_dir="${2:-.}"

	# Validate trace directory
	if [[ -z ${trace_dir} ]]; then
		echo "ERROR: Trace directory not specified."
		echo "Usage: run_all_metrics <trace_dir> [output_dir]"
		return 1
	fi

	if [[ ! -d ${trace_dir} ]]; then
		echo "ERROR: Trace directory does not exist: ${trace_dir}"
		return 1
	fi

	# Create output directory if it doesn't exist
	mkdir -p "${output_dir}"

	# Get the project root (parent of trace_collection)
	local common_dir
	common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	local trace_collection_dir
	trace_collection_dir="$(dirname "${common_dir}")"
	local project_root
	project_root="$(dirname "${trace_collection_dir}")"

	echo "=========================================="
	echo "Running All Metrics"
	echo "  Trace dir:   ${trace_dir}"
	echo "  Output dir:  ${output_dir}"
	echo "  Project root: ${project_root}"
	echo "=========================================="

	local failed_metrics=()
	local success_count=0

	for metric in "${METRICS[@]}"; do
		local output_file="${output_dir}/${metric}.json"
		echo ""
		echo "Running metric: ${metric}"
		echo "  Output: ${output_file}"

		if python -m tools.main \
			--trace "${trace_dir}" \
			--output-json \
			--metric "${metric}" \
			> "${output_file}" 2>&1; then
			echo "  ✓ Success"
			((success_count++))
		else
			echo "  ✗ Failed (see ${output_file} for details)"
			failed_metrics+=("${metric}")
		fi
	done

	echo ""
	echo "=========================================="
	echo "Metrics Summary"
	echo "  Total:   ${#METRICS[@]}"
	echo "  Success: ${success_count}"
	echo "  Failed:  ${#failed_metrics[@]}"
	if [[ ${#failed_metrics[@]} -gt 0 ]]; then
		echo "  Failed metrics: ${failed_metrics[*]}"
	fi
	echo "  Output directory: ${output_dir}"
	echo "=========================================="

	# Return non-zero if any metrics failed
	[[ ${#failed_metrics[@]} -eq 0 ]]
}
