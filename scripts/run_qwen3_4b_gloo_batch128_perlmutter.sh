#!/usr/bin/env bash
set -uo pipefail

# Qwen3-4B vLLM Kineto/NSYS run with Gloo as the collective library.
#
# Single-node, TP=4, batch=128, input=1024, output=128 — matches the
# existing Qwen3-4B NCCL / NCCL+MSCCL++ / pure-MSCCL++ Perlmutter rows so
# Gloo slots into the comm-library comparison cleanly.
#
# Gloo is a torch.distributed TCP backend; it has no RDMA/NVLink fast path,
# so it acts as the bandwidth-floor baseline for the website.
#
# Requires the ccl-bench vLLM patch to be installed (see
# scripts/vllm_gloo_patch/README.md). The patch is gated by two env vars so
# normal runs are unaffected:
#   CCL_BENCH_DIST_BACKEND=gloo        -> init_process_group uses gloo
#   CCL_BENCH_FORCE_TORCH_DISTRIBUTED=1 -> pynccl is skipped; all collectives
#                                         go through torch.distributed
#
# When both are set, vLLM's CUDA communicator falls back to
# torch.distributed.{all_reduce, all_gather, reduce_scatter_tensor,
# broadcast}, which in turn dispatch to the Gloo backend.

TRACE_ROOT="${TRACE_ROOT:-$PSCRATCH/ccl-bench-traces/qwen3-4b-gloo-batch128}"
HF_HOME="${HF_HOME:-$PSCRATCH/huggingface}"
MODEL_ID="${MODEL_ID:-$PSCRATCH/models/Qwen3-4B}"

INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1152}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_BATCHED_TOKENS_CAP="${MAX_NUM_BATCHED_TOKENS_CAP:-16384}"
# Gloo is TCP-bound; keep iteration counts low to fit in a 2 h allocation.
NUM_ITERS_WARMUP="${NUM_ITERS_WARMUP:-1}"
NUM_ITERS="${NUM_ITERS:-3}"
PROFILE_WARMUP="${PROFILE_WARMUP:-1}"

export HF_HOME
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
# Gloo path: force torch.distributed + gloo backend. These two vars are what
# the vLLM patch reads.
export CCL_BENCH_DIST_BACKEND=gloo
export CCL_BENCH_FORCE_TORCH_DISTRIBUTED=1
# Unused for Gloo but harmless; keeps env consistent with NCCL variants.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
# Gloo does not care about expandable segments.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$TRACE_ROOT"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

scheduled_batched_tokens() {
  local full_tokens=$((BATCH_SIZE * (INPUT_LEN + OUTPUT_LEN)))
  if [[ "$MAX_NUM_BATCHED_TOKENS_CAP" =~ ^[0-9]+$ ]] && \
     (( MAX_NUM_BATCHED_TOKENS_CAP > 0 )) && \
     (( full_tokens > MAX_NUM_BATCHED_TOKENS_CAP )); then
    echo "$MAX_NUM_BATCHED_TOKENS_CAP"
  else
    echo "$full_tokens"
  fi
}

verify_vllm_patch() {
  # Confirm the vLLM patch is installed: the cuda_communicator.py should
  # reference CCL_BENCH_FORCE_TORCH_DISTRIBUTED.
  local cc_path
  cc_path=$(python -c 'import vllm.distributed.device_communicators.cuda_communicator as m; print(m.__file__)' 2>/dev/null) || {
    echo "cannot import vllm cuda_communicator — is the cvllm env active?" >&2
    return 1
  }
  if ! grep -q "CCL_BENCH_FORCE_TORCH_DISTRIBUTED" "$cc_path"; then
    echo "vLLM patch not detected at $cc_path" >&2
    echo "(expected token CCL_BENCH_FORCE_TORCH_DISTRIBUTED)" >&2
    echo "See scripts/vllm_gloo_patch/README.md to install it." >&2
    return 1
  fi
  echo "  vLLM patch OK ($cc_path)"
}

verify_gloo_in_log() {
  local log=$1
  # If 'backend=gloo' does not show up in the worker init log, Gloo is not
  # being used. Treat this as a hard failure so we never silently record a
  # mislabeled trace.
  if ! grep -q "backend=gloo" "$log"; then
    echo "  ! no 'backend=gloo' line in $log — Gloo was not engaged" >&2
    return 1
  fi
  if grep -qE "pynccl|ncclCommInit" "$log"; then
    echo "  ! unexpected NCCL init in $log:" >&2
    grep -nE "pynccl|ncclCommInit" "$log" | head -5 >&2
    return 1
  fi
  echo "  Gloo engagement verified in $log"
}

run_one() {
  local name=$1
  local model_id=$2
  local tp=$3

  local batched_tokens
  batched_tokens=$(scheduled_batched_tokens)

  local out_dir="$TRACE_ROOT/$name"
  local kineto_dir="$out_dir/kineto"
  local nsys_prefix="$out_dir/${name}"
  local latency_json="$out_dir/${name}.latency.json"
  local bench_log="$out_dir/${name}.bench.log"
  local profile_log="$out_dir/${name}.profile.log"

  rm -rf "$out_dir"
  mkdir -p "$kineto_dir"

  echo
  echo "============================================================"
  echo "RUN $name  (Gloo via torch.distributed)"
  echo "  model: $model_id  tp=$tp  batch=$BATCH_SIZE"
  echo "  input=$INPUT_LEN output=$OUTPUT_LEN max_num_batched_tokens=$batched_tokens"
  echo "  CCL_BENCH_DIST_BACKEND=$CCL_BENCH_DIST_BACKEND"
  echo "  CCL_BENCH_FORCE_TORCH_DISTRIBUTED=$CCL_BENCH_FORCE_TORCH_DISTRIBUTED"
  echo "  out: $out_dir"
  echo "============================================================"

  local common_args=(
    --model "$model_id"
    --input-len "$INPUT_LEN"
    --output-len "$OUTPUT_LEN"
    --batch-size "$BATCH_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$BATCH_SIZE"
    --max-num-batched-tokens "$batched_tokens"
    --enforce-eager
    --disable-detokenize
    --disable-custom-all-reduce
    --tensor-parallel-size "$tp"
  )

  # Plain env: CCL_BENCH_* already exported above. No LD_PRELOAD, no
  # VLLM_NCCL_SO_PATH.
  local env_cmd=(env)

  echo "[1/2] latency run..."
  "${env_cmd[@]}" python -m vllm.entrypoints.cli.main bench latency \
    "${common_args[@]}" \
    --num-iters-warmup "$NUM_ITERS_WARMUP" \
    --num-iters "$NUM_ITERS" \
    --output-json "$latency_json" \
    > "$bench_log" 2>&1
  local latency_rc=$?
  if [[ "$latency_rc" != 0 ]]; then
    echo "  ! latency run failed rc=$latency_rc; see $bench_log"
    return "$latency_rc"
  fi
  verify_gloo_in_log "$bench_log" || return 1

  echo "[2/2] kineto + nsys profile run..."
  nsys profile \
    -t cuda,nvtx,osrt \
    -s none --cpuctxsw=none \
    --trace-fork-before-exec=true \
    --force-overwrite=true \
    --stats=true \
    --export=sqlite \
    -o "$nsys_prefix" \
    "${env_cmd[@]}" python -m vllm.entrypoints.cli.main bench latency \
      "${common_args[@]}" \
      --num-iters-warmup "$PROFILE_WARMUP" \
      --num-iters 1 \
      --profile \
      --profiler-config.profiler torch \
      --profiler-config.torch_profiler_dir "$kineto_dir" \
      --profiler-config.torch_profiler_with_stack false \
      --profiler-config.torch_profiler_record_shapes true \
      --profiler-config.torch_profiler_with_memory true \
      --profiler-config.torch_profiler_with_flops true \
      --profiler-config.torch_profiler_use_gzip false \
      > "$profile_log" 2>&1
  local profile_rc=$?
  if [[ "$profile_rc" != 0 ]]; then
    echo "  ! profile run failed rc=$profile_rc; see $profile_log"
    return "$profile_rc"
  fi
  verify_gloo_in_log "$profile_log" || return 1

  echo "  produced:"
  find "$out_dir" -maxdepth 2 -type f | sed "s#^#    #"
}

main() {
  require_cmd python
  require_cmd nsys
  require_cmd nvidia-smi

  echo "Host: $(hostname)"
  echo "GPU count: $(nvidia-smi -L | wc -l)"
  echo "Trace root: $TRACE_ROOT"
  echo "Model: $MODEL_ID"

  verify_vllm_patch || exit 1

  if [[ ! -d "$MODEL_ID" ]]; then
    echo "ERROR: model directory not found: $MODEL_ID" >&2
    exit 1
  fi

  run_one "qwen3-4b-vllm-tp${TP}-batch${BATCH_SIZE}-gloo-perlmutter" \
    "$MODEL_ID" "$TP" || echo "FAILED: qwen tp${TP} gloo"

  echo
  echo "==== DONE ===="
  find "$TRACE_ROOT" -maxdepth 2 -type f | sed "s#^#  #"
}

main "$@"
