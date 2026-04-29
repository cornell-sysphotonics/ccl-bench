#!/bin/bash
# train_llama8b.sh — Llama-3.1-8B torchtitan benchmark for CCL-Bench agent
#
# Called by execute.py with env vars injected from the config dict:
#   TP, DP, PP            — parallelism degrees
#   MICRO_BATCH_SIZE      — pipeline microbatch size (also accepts MICRO_BATCH)
#   COMPILE_MODE          — "eager" | "compile"
#   ACTIVATION_CHECKPOINTING — "true" | "false"
#   ENABLE_OPUS_BACKEND   — "true" | "false" (requires the opus Python package)
#   TRACE_DIR             — output root for profiler traces and metric YAML
#   WORKLOAD_NAME         — informational label

set -e

# ── Config from environment ────────────────────────────────────────────────────
TP=${TP:-4}
DP=${DP:-1}
PP=${PP:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-${MICRO_BATCH:-1}}
COMPILE_MODE=${COMPILE_MODE:-"eager"}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-"false"}
ENABLE_OPUS_BACKEND=${ENABLE_OPUS_BACKEND:-"false"}
TRACE_DIR=${TRACE_DIR:-"/pscratch/sd/e/ericding/ccl-bench/perlmutter_llama8b"}

# ── Workload constants (from workload card) ────────────────────────────────────
GLOBAL_BATCH=32
SEQ_LEN=1024
GPUS_PER_NODE=4

# ── Derived parameters ─────────────────────────────────────────────────────────
TOTAL_GPUS=$(( TP * DP * PP ))
NPROC_PER_NODE=$(( TOTAL_GPUS < GPUS_PER_NODE ? TOTAL_GPUS : GPUS_PER_NODE ))
NNODES=$(( TOTAL_GPUS / NPROC_PER_NODE ))
LOCAL_BATCH=$(( GLOBAL_BATCH / DP ))

# micro_batch_size has no effect when pp=1; normalize so the agent sees consistent results
if [ "$PP" -eq 1 ]; then
    MICRO_BATCH_SIZE=1
fi

if [ "$ACTIVATION_CHECKPOINTING" = "true" ]; then
    AC_MODE="full"
else
    AC_MODE="selective"
fi

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_DIR="$(cd "$SCRIPT_DIR/../../torchtitan" && pwd)"
TOKENIZER_PATH="$TORCHTITAN_DIR/tests/assets/tokenizer"

# ── Lustre fix: redirect HF cache to /tmp ─────────────────────────────────────
# Perlmutter's $HOME and $PSCRATCH are on Lustre, which does not support
# fcntl.flock.  huggingface_hub uses flock for cache locking and raises
# OSError 524 on Lustre.  /tmp is node-local and supports flock.
export HF_HOME="/tmp/hf_home_${USER}"
export HF_DATASETS_CACHE="/tmp/hf_datasets_${USER}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# ── Profiler schedule: Opus TorchTitan uses profile_freq with fixed warmup ─────
TRAINING_STEPS=10
PROFILE_FREQ=10

# Traces land at $TRACE_DIR/profile_traces/iteration_N/rank{r}_trace.json
SAVE_TRACES_FOLDER="profile_traces"

mkdir -p "$TRACE_DIR"

# ── Build CLI overrides ────────────────────────────────────────────────────────
OVERRIDES=(
    --model.name llama3
    --model.flavor 8B
    --model.hf_assets_path "$TOKENIZER_PATH"
    --training.local_batch_size "$LOCAL_BATCH"
    --training.seq_len "$SEQ_LEN"
    --training.steps "$TRAINING_STEPS"
    --training.dataset c4_test
    --parallelism.tensor_parallel_degree "$TP"
    --parallelism.data_parallel_shard_degree "$DP"
    --parallelism.data_parallel_replicate_degree 1
    --parallelism.fsdp_reshard_after_forward always
    --parallelism.pipeline_parallel_degree "$PP"
    --parallelism.pipeline_parallel_microbatch_size "$MICRO_BATCH_SIZE"
    --activation_checkpoint.mode "$AC_MODE"
    --profiling.enable_profiling
    --profiling.profile_freq "$PROFILE_FREQ"
    --profiling.save_traces_folder "$SAVE_TRACES_FOLDER"
    --job.dump_folder "$TRACE_DIR"
    --metrics.disable_color_printing
)

if [ "$ENABLE_OPUS_BACKEND" = "true" ]; then
    OVERRIDES+=(--training.enable_opus_backend)
fi

if [ "$COMPILE_MODE" != "eager" ]; then
    OVERRIDES+=(--compile.enable)
fi

# ── Launch ─────────────────────────────────────────────────────────────────────
cd "$TORCHTITAN_DIR"

export PYTORCH_ALLOC_CONF="expandable_segments:True"

if [ "$NNODES" -gt 1 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    # Multi-node inside a SLURM allocation (Perlmutter)
    MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
    MASTER_PORT=29510
    srun --ntasks-per-node=1 --nodes="$NNODES" --gpus-per-node="$NPROC_PER_NODE" \
        torchrun \
            --nnodes="$NNODES" \
            --nproc_per_node="$NPROC_PER_NODE" \
            --rdzv_id=42 \
            --rdzv_backend=c10d \
            --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
            -m torchtitan.train \
            "${OVERRIDES[@]}"
else
    # Single-node
    torchrun \
        --nproc_per_node="$NPROC_PER_NODE" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="localhost:0" \
        -m torchtitan.train \
        "${OVERRIDES[@]}"
fi

# ── Flatten traces into TRACE_DIR for metric tools ─────────────────────────────
# avg_step_time scans the flat directory for *.json files; torchtitan writes
# them nested under profile_traces/iteration_N/.  Copy them up to TRACE_DIR.
PROFILE_DIR="$TRACE_DIR/$SAVE_TRACES_FOLDER"
if [ -d "$PROFILE_DIR" ]; then
    ITER_DIR=$(ls -td "$PROFILE_DIR"/iteration_* 2>/dev/null | head -1)
    if [ -n "$ITER_DIR" ] && [ -d "$ITER_DIR" ]; then
        cp "$ITER_DIR"/*.json "$TRACE_DIR/" 2>/dev/null || true
    fi
fi

# Write minimal YAML so avg_step_time dispatches to the json backend
cat > "$TRACE_DIR/trace_meta.yaml" << 'YAML'
metric_source:
  traces:
    - json
YAML
