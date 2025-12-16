#!/bin/bash

set -e

###############################################
# DISTRIBUTED CONFIGURATION
###############################################
NNODES=2
MASTER_PORT=30001

: "${NODE_RANK:?Environment variable NODE_RANK must be set (0..1)}"
: "${MASTER_ADDR:?Environment variable MASTER_ADDR must be set (rank-0 hostname/IP)}"

# Basic sanity check
if [ "$NODE_RANK" -lt 0 ] || [ "$NODE_RANK" -ge "$NNODES" ]; then
  echo "ERROR: NODE_RANK (${NODE_RANK}) must be in [0, $((NNODES - 1))]"
  exit 1
fi

BASE_DIR="$(pwd)"

###############################################
# SERVER CONFIGURATION
###############################################
MODEL_PATH="Qwen/Qwen3-32B"  # HF name, not a local folder
TP=4
PP=2
PORT=30000
HOST="0.0.0.0"
CUDA_VISIBLE_DEVICES="0,1,2,3"

###############################################
# PER-NODE WORKDIR (avoids NFS file collisions)
# Important: this isolates cuda_graph_runner_memory_usage.pickle
###############################################
NODE_WORKDIR="${BASE_DIR}/node${NODE_RANK}"
mkdir -p "${NODE_WORKDIR}"
cd "${NODE_WORKDIR}"

###############################################
# SYNC FILES FOR CROSS-NODE COORDINATION (NFS)
###############################################
SYNC_DIR="${BASE_DIR}/sglang_multi_node_sync_tp${TP}_pp${PP}"
mkdir -p "${SYNC_DIR}"

HEALTH_DONE_FILE="${SYNC_DIR}/health_done_node0"
BENCH_DONE_FILE="${SYNC_DIR}/bench_done_node0"
PROFILE_STARTED_NODE1="${SYNC_DIR}/profile_started_node1"

###############################################
# PROFILING ENVIRONMENT
###############################################
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE="./tp_${TP}_pp_${PP}_node${NODE_RANK}_nccl_debug_log_%p.txt"

export SGLANG_TORCH_PROFILER_DIR="./tp_${TP}_pp_${PP}_node${NODE_RANK}_torch_profile_dir"
mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"

###############################################
# BENCHMARK CONFIGURATION
###############################################
NUM_PROMPTS=32
INPUT_LEN=1024
OUTPUT_LEN=256
RANGE_RATIO=0.5
MAX_CONCURRENCY=4

###############################################
# NSYS CONFIGURATION
###############################################
NSYS_SESSION="sglang_node${NODE_RANK}"
OUTPUT_DIR="sglang_profile_$(date +%Y%m%d_%H%M%S)_tp${TP}_pp${PP}_node${NODE_RANK}"
PROFILE_OUTPUT="${OUTPUT_DIR}/sglang_profile_tp${TP}_pp${PP}_node${NODE_RANK}.nsys-rep"

# Comprehensive trace options
NSYS_TRACE="cuda,nvtx,mpi,osrt,cudnn,cublas"

###############################################
# SETUP
###############################################
export CUDA_VISIBLE_DEVICES

mkdir -p "$OUTPUT_DIR"

echo "=== Starting SGLang server with nsys (profiling disabled initially) ==="
echo "BASE_DIR             = ${BASE_DIR}"
echo "NODE_WORKDIR         = ${NODE_WORKDIR}"
echo "SYNC_DIR             = ${SYNC_DIR}"
echo "MODEL_PATH           = ${MODEL_PATH}"
echo "TP/PP                = ${TP}/${PP}"
echo "NNODES               = ${NNODES}"
echo "NODE_RANK            = ${NODE_RANK}"
echo "MASTER_ADDR          = ${MASTER_ADDR}"
echo "MASTER_PORT          = ${MASTER_PORT}"
echo "PORT                 = ${PORT}"
echo "HOST                 = ${HOST}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "OUTPUT_DIR           = ${OUTPUT_DIR}"
echo "NSYS_SESSION         = ${NSYS_SESSION}"
echo "NSYS_TRACE           = ${NSYS_TRACE}"
echo ""

# Pause DCGM to avoid conflicts with nsys (ignore failure)
dcgmi profile --pause || true

###############################################
# LAUNCH SERVER WITH NSYS (PROFILING PAUSED)
###############################################
nsys launch \
  --session="${NSYS_SESSION}" \
  --trace="${NSYS_TRACE}" \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --cuda-memory-usage=true \
  python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --nnodes "${NNODES}" \
    --node-rank "${NODE_RANK}" \
    --dist-init-addr "${MASTER_ADDR}:${MASTER_PORT}" \
    --host "$HOST" \
    --port "$PORT" \
    --enable-profile-cuda-graph \
    --enable-layerwise-nvtx-marker \
  > "$OUTPUT_DIR/server_stdout_node${NODE_RANK}.log" \
  2> "$OUTPUT_DIR/server_stderr_node${NODE_RANK}.log" &

SERVER_PID=$!
echo "Server launched on node ${NODE_RANK} with PID: $SERVER_PID"

###############################################
# WAIT FOR SERVER HEALTH (NODE 0 ONLY)
###############################################
if [ "$NODE_RANK" -eq 0 ]; then
  echo "=== Waiting for server to be ready (node 0) ==="
  MAX_RETRIES=60
  RETRY_COUNT=0
  HEALTH_URL="http://127.0.0.1:${PORT}/health"

  while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null || echo "000")

    if [ "$HTTP_CODE" = "200" ]; then
      echo "✓ Server on node 0 is healthy!"
      break
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for server on node 0... ($RETRY_COUNT/$MAX_RETRIES) [HTTP: $HTTP_CODE]"
    sleep 5

    # Check if server process is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
      echo "ERROR: Server process on node 0 died. Check $OUTPUT_DIR/server_stdout_node${NODE_RANK}.log"
      exit 1
    fi
  done

  if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "ERROR: Server on node 0 failed to become healthy after $MAX_RETRIES attempts"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
  fi

  # Give it a moment to fully stabilize
  echo "Server on node 0 is ready. Waiting 5s for full stabilization..."
  sleep 5

  # Signal to node 1 that health check is complete
  touch "${HEALTH_DONE_FILE}"
  echo "Node 0: wrote health completion flag at ${HEALTH_DONE_FILE}"
else
  echo "Node ${NODE_RANK}: Skipping server health check (only node 0)."
fi

###############################################
# START PROFILING (ALL NODES, ORDERED)
###############################################
if [ "$NODE_RANK" -ne 0 ]; then
  echo "Node ${NODE_RANK}: Waiting for node 0 health check before starting profiling..."
  while [ ! -f "${HEALTH_DONE_FILE}" ]; do
    sleep 2
  done
  echo "Node ${NODE_RANK}: Detected health completion flag. Starting profiling..."
else
  echo "Node 0: Starting profiling after successful health check."
fi

echo "=== Starting nsys profiling (node ${NODE_RANK}) ==="
nsys start \
  --session="${NSYS_SESSION}" \
  --sample=none \
  -o "${PROFILE_OUTPUT}" \
  --force-overwrite=true \
  --gpu-metrics-device=all
  # \--gpu-metrics-set=all

# New requirement: node 0 should run benchmark AFTER node 1 has started profiling
if [ "$NODE_RANK" -eq 1 ]; then
  touch "${PROFILE_STARTED_NODE1}"
  echo "Node 1: wrote profile-start flag at ${PROFILE_STARTED_NODE1}"
fi

###############################################
# RUN BENCHMARK (NODE 0 ONLY, AFTER NODE 1 STARTS PROFILING)
###############################################
if [ "$NODE_RANK" -eq 0 ]; then
  echo "Node 0: Waiting for node 1 to start profiling before running benchmark..."
  while [ ! -f "${PROFILE_STARTED_NODE1}" ]; do
    sleep 2
  done
  echo "Node 0: Detected that node 1 has started profiling. Running benchmark..."

  echo "=== Running benchmark on node 0 ==="
  python -m sglang.bench_serving \
    --backend sglang \
    --host "127.0.0.1" \
    --port "$PORT" \
    --model "$MODEL_PATH" \
    --dataset-name random \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio "$RANGE_RATIO" \
    --request-rate inf \
    --max-concurrency "$MAX_CONCURRENCY" \
    --output-file "$OUTPUT_DIR/bench_results_node${NODE_RANK}.jsonl" \
    --output-details \
    --profile \
    | tee "$OUTPUT_DIR/benchmark_stdout_node${NODE_RANK}.log"

  # Signal to node 1 that benchmark is done
  touch "${BENCH_DONE_FILE}"
  echo "Node 0: wrote benchmark completion flag at ${BENCH_DONE_FILE}"
else
  echo "Node ${NODE_RANK}: Skipping benchmark (only node 0 runs it)."
fi

###############################################
# STOP PROFILING (ALL NODES, ORDERED)
###############################################
if [ "$NODE_RANK" -ne 0 ]; then
  echo "Node ${NODE_RANK}: Waiting for node 0 benchmark completion before stopping profiling..."
  while [ ! -f "${BENCH_DONE_FILE}" ]; do
    sleep 2
  done
  echo "Node ${NODE_RANK}: Detected benchmark completion flag. Stopping profiling..."
else
  echo "Node 0: Stopping profiling after benchmark."
fi

echo "=== Stopping nsys profiling (node ${NODE_RANK}) ==="
nsys stop --session="${NSYS_SESSION}"

###############################################
# SHUTDOWN
###############################################
echo "=== Shutting down server on node ${NODE_RANK} ==="
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

###############################################
# SUMMARY
###############################################
echo ""
echo "=== Profiling complete on node ${NODE_RANK}! ==="
echo "Profile saved to:       $PROFILE_OUTPUT"
if [ "$NODE_RANK" -eq 0 ]; then
  echo "Benchmark results:      $OUTPUT_DIR/bench_results_node${NODE_RANK}.jsonl"
  echo "Benchmark stdout:       $OUTPUT_DIR/benchmark_stdout_node${NODE_RANK}.log"
fi
echo "Server stdout:          $OUTPUT_DIR/server_stdout_node${NODE_RANK}.log"
echo "Server stderr:          $OUTPUT_DIR/server_stderr_node${NODE_RANK}.log"
echo "NCCL debug logs:        tp_${TP}_pp_${PP}_node${NODE_RANK}_nccl_debug_log_*.txt"
echo "Torch profiler output:  $SGLANG_TORCH_PROFILER_DIR/"
echo ""
echo "Note: per-node workdir is ${NODE_WORKDIR}, so files like cuda_graph_runner_memory_usage.pickle"
echo "are isolated per node and no longer collide on NFS."
echo ""
echo "To view the nsys profile (node ${NODE_RANK}):"
echo "  nsys-ui $PROFILE_OUTPUT"
echo ""
echo "Or export to SQLite for analysis:"
echo "  nsys export --type sqlite --output $OUTPUT_DIR/profile_node${NODE_RANK}.sqlite $PROFILE_OUTPUT"
echo ""
echo "To analyze torch profiles:"
echo "  python -m torch.profiler.analyze $SGLANG_TORCH_PROFILER_DIR"
echo ""
