#!/bin/bash

set -e


###############################################
# SERVER CONFIGURATION
###############################################
MODEL_PATH="meta-llama/Llama-3.1-8B"
TP=1
PP=4
PORT=30000
HOST="0.0.0.0"
CUDA_VISIBLE_DEVICES="0,1,2,3"


###############################################
# PROFILING ENVIRONMENT
###############################################
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE="./pp_${PP}_nccl_debug_log_%p.txt"
# export SGLANG_TORCH_PROFILER_DIR="./pp_${PP}torch_profile_dir"
# mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"

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
NSYS_SESSION="sglang"
OUTPUT_DIR="sglang_profile_$(date +%Y%m%d_%H%M%S)_pp${PP}"
PROFILE_OUTPUT="nsys_0.nsys-rep"

# Comprehensive trace options
NSYS_TRACE="cuda,nvtx,mpi,osrt,cudnn,cublas"

###############################################
# SETUP
###############################################
export CUDA_VISIBLE_DEVICES

mkdir -p "$OUTPUT_DIR"

echo "=== Starting SGLang server with nsys (profiling disabled initially) ==="
echo "MODEL_PATH           = ${MODEL_PATH}"
echo "TP/PP                = ${TP}/${PP}"
echo "PORT                 = ${PORT}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "OUTPUT_DIR           = ${OUTPUT_DIR}"
echo "NSYS_SESSION         = ${NSYS_SESSION}"
echo "NSYS_TRACE           = ${NSYS_TRACE}"
echo ""

# Pause DCGM to avoid conflicts with nsys
dcgmi profile --pause

###############################################
# LAUNCH SERVER WITH NSYS (PROFILING PAUSED)
###############################################
# Launch server with nsys but profiling stopped initially
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
    --attention-backend torch_native \
    --host "$HOST" \
    --port "$PORT" \
    --enable-profile-cuda-graph \
    --enable-layerwise-nvtx-marker \
  > "$OUTPUT_DIR/server_stdout.log" 2> "$OUTPUT_DIR/server_stderr.log" &

SERVER_PID=$!
echo "Server launched with PID: $SERVER_PID"

###############################################
# WAIT FOR SERVER HEALTH
###############################################
echo "=== Waiting for server to be ready ==="
MAX_RETRIES=60
RETRY_COUNT=0
HEALTH_URL="http://127.0.0.1:${PORT}/health"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null || echo "000")
  
  if [ "$HTTP_CODE" = "200" ]; then
    echo "✓ Server is healthy!"
    break
  fi
  
  RETRY_COUNT=$((RETRY_COUNT + 1))
  echo "Waiting for server... ($RETRY_COUNT/$MAX_RETRIES) [HTTP: $HTTP_CODE]"
  sleep 5
  
  # Check if server process is still running
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server process died. Check $OUTPUT_DIR/server_stdout.log"
    exit 1
  fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
  echo "ERROR: Server failed to become healthy after $MAX_RETRIES attempts"
  kill $SERVER_PID 2>/dev/null || true
  exit 1
fi

# Give it a moment to fully stabilize
echo "Server is ready. Waiting 5s for full stabilization..."
sleep 5

###############################################
# START PROFILING
###############################################
echo "=== Starting nsys profiling ==="
nsys start \
  --session="${NSYS_SESSION}" \
  --sample=none \
  -o "${PROFILE_OUTPUT}" \
  --force-overwrite=true \
  --gpu-metrics-device=all 
  #\--gpu-metrics-set=all

###############################################
# RUN BENCHMARK
###############################################
echo "=== Running benchmark ==="
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
  --output-file "bench_results.jsonl" \
  --output-details \
  | tee "$OUTPUT_DIR/benchmark_stdout.log"
  # --profile \

###############################################
# STOP PROFILING
###############################################
echo "=== Stopping nsys profiling ==="
nsys stop --session="${NSYS_SESSION}"

###############################################
# SHUTDOWN
###############################################
echo "=== Shutting down server ==="
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

###############################################
# SUMMARY
###############################################
echo ""
echo "=== Profiling complete! ==="
echo "Profile saved to:       $PROFILE_OUTPUT"
echo "Benchmark results:      bench_results.jsonl"
echo "Benchmark stdout:       $OUTPUT_DIR/benchmark_stdout.log"
echo "Server stdout:          $OUTPUT_DIR/server_stdout.log"
echo "Server stderr:          $OUTPUT_DIR/server_stderr.log"
echo "NCCL debug logs:        nccl_debug_log_*.txt"
# echo "Torch profiler output:  $SGLANG_TORCH_PROFILER_DIR/"
echo ""
echo "To view the nsys profile:"
echo "  nsys-ui $PROFILE_OUTPUT"
echo ""
echo "Or export to SQLite for analysis:"
echo "  nsys export --type sqlite --output $OUTPUT_DIR/profile.sqlite $PROFILE_OUTPUT"
echo ""
echo "To analyze torch profiles:"
# echo "  python -m torch.profiler.analyze $SGLANG_TORCH_PROFILER_DIR"