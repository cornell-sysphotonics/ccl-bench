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
export NERSC_ALLOCATION="${NERSC_ALLOCATION:-CHANGE_ME}"

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Auto-detect paths based on script location
if [[ -n ${SLURM_SUBMIT_DIR:-}   ]]; then
	# Running under Slurm
	SCRIPT_DIR="${SLURM_SUBMIT_DIR}/perlmutter"
else
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

CCL_BENCH_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"
export CCL_BENCH_HOME
export TORCHTITAN_HOME="${TORCHTITAN_HOME:-${SCRATCH}/torchtitan}"
export VENV_DIR="${CCL_BENCH_HOME}/.venv"

# Trace output directory base
export TRACE_BASE="${CCL_BENCH_HOME}/trace_collection"

# Train config directory
export TRAIN_CONFIG_DIR="${CCL_BENCH_HOME}/train_configs"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

setup_environment() {
	# Load required modules
	module load python 2> /dev/null || true
	module load cudatoolkit 2> /dev/null || true

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
# TRACE COLLECTION SETUP
# =============================================================================

setup_trace_dir() {
	local workload_name="$1"

	export TRACE_DIR="${TRACE_BASE}/${workload_name}"
	mkdir -p "${TRACE_DIR}"

	echo "Trace output directory: ${TRACE_DIR}"

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
		--trace=nvtx,cuda,osrt \
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
# Usage: launch_torchtitan <config_file>
launch_torchtitan() {
	local config_file="$1"

	if [[ ! -f ${config_file} ]]; then
		echo "ERROR: Config file not found: ${config_file}"
		exit 1
	fi

	echo "Launching TorchTitan with config: ${config_file}"

	python -m torch.distributed.run \
		--nnodes="${SLURM_JOB_NUM_NODES}" \
		--nproc-per-node="${SLURM_GPUS_PER_NODE}" \
		--rdzv-backend=c10d \
		--rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
		-m torchtitan.train \
		--job.config_file "${config_file}" \
		--job.dump_folder "${TRACE_DIR}"
}

# Launch with NSys profiling
# Usage: launch_torchtitan_with_nsys <config_file> <nsys_output_prefix>
launch_torchtitan_with_nsys() {
	local config_file="$1"
	local nsys_prefix="$2"

	if [[ ! -f ${config_file} ]]; then
		echo "ERROR: Config file not found: ${config_file}"
		exit 1
	fi

	echo "Launching TorchTitan with NSys profiling..."
	echo "  Config: ${config_file}"
	echo "  NSys output: ${TRACE_DIR}/${nsys_prefix}_${RUN_TIMESTAMP}"

	nsys_profile "${nsys_prefix}" \
		python -m torch.distributed.run \
		--nnodes="${SLURM_JOB_NUM_NODES}" \
		--nproc-per-node="${SLURM_GPUS_PER_NODE}" \
		--rdzv-backend=c10d \
		--rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
		-m torchtitan.train \
		--job.config_file "${config_file}" \
		--job.dump_folder "${TRACE_DIR}"
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
