#!/bin/bash
#
# Orchestrated profiling of vLLM server with nsys + NVLink utilization
#
# Workflow:
# 1. Launch vLLM server (no nsys yet, let it warm up)
# 2. Wait for server to initialize
# 3. Start nsys profiling + NVLink polling simultaneously
# 4. Run client workload
# 5. Stop both, export nsys to SQLite
#
# Key: We capture nsys session start time (wall-clock) which aligns with
# nsys internal timestamps, allowing correlation with NVLink (NVML wall-clock).
#

set -e

# Use specific nsys version
export PATH=~/CS5470/assignment_1/nsys_new/opt/nvidia/nsight-systems-cli/2025.5.1/bin:$PATH
echo "Using nsys: $(which nsys)"
echo "  Version: $(nsys --version 2>&1 | head -1)"

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/nsys_profile}"
NVLINK_INTERVAL_MS="${NVLINK_INTERVAL_MS:-1.0}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
TP_SIZE="${TP_SIZE:-4}"
SERVER_PORT="${SERVER_PORT:-8000}"
NSYS_OUTPUT="${OUTPUT_DIR}/vllm_profile"
PROFILE_DURATION="${PROFILE_DURATION:-60}"  # Max profiling duration in seconds

# Batch size experiment parameters
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-$((MAX_NUM_SEQS * 4096))}"
NUM_PROMPTS="${NUM_PROMPTS:-$((MAX_NUM_SEQS * 5))}"
REQUEST_RATE="${REQUEST_RATE:-inf}"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "=== vLLM + NVLink Profiling Session ==="
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Batch Size (max-num-seqs): $MAX_NUM_SEQS"
echo "Max Batched Tokens: $MAX_BATCHED_TOKENS"
echo "Num Prompts: $NUM_PROMPTS"
echo "Request Rate: $REQUEST_RATE"
echo ""

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Cleanup function
cleanup() {
    echo ""
    echo "[$(date)] Cleaning up..."
    
    # Stop NVLink poller
    if [ -n "$NVLINK_PID" ] && kill -0 $NVLINK_PID 2>/dev/null; then
        echo "  Stopping NVLink poller..."
        kill -INT $NVLINK_PID 2>/dev/null || true
        wait $NVLINK_PID 2>/dev/null || true
    fi
    
    # Stop nsys (which wraps the server)
    if [ -n "$NSYS_PID" ] && kill -0 $NSYS_PID 2>/dev/null; then
        echo "  Stopping nsys + server..."
        kill -INT $NSYS_PID 2>/dev/null || true
        wait $NSYS_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ============================================================
# PHASE 1: Start server under nsys (captures everything including model load)
# ============================================================
echo "[$(date)] === Phase 1: Starting vLLM Server under nsys ==="

# Record nsys start time - used for timestamp correlation
NSYS_START_WALL_NS=$(python3 -c "import time; print(time.time_ns())")
NSYS_START_MONO_NS=$(python3 -c "import time; print(time.monotonic_ns())")
NSYS_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%S.%NZ")

cat > "$OUTPUT_DIR/timing_nsys_start.json" << EOF
{
  "name": "nsys_start",
  "wall_clock_ns": $NSYS_START_WALL_NS,
  "wall_clock_us": $(($NSYS_START_WALL_NS / 1000)),
  "monotonic_ns": $NSYS_START_MONO_NS,
  "monotonic_us": $(($NSYS_START_MONO_NS / 1000)),
  "wall_clock_iso": "$NSYS_START_ISO",
  "description": "Wall-clock when nsys started - nsys timestamps are relative to this"
}
EOF

echo "[$(date)] nsys start time: $NSYS_START_ISO"

nsys profile \
    --output="$NSYS_OUTPUT" \
    --force-overwrite=true \
    --trace=cuda,nvtx,cudnn,cublas \
    --cuda-memory-usage=true \
    --sample=none \
    --cpuctxsw=none \
    -- python3 -m vllm.entrypoints.openai.api_server \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len 4096 \
        --model "$MODEL" \
        --swap-space 16 \
        --disable-log-requests \
        --enforce-eager \
        --enable-chunked-prefill \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
        --port "$SERVER_PORT" \
        --disable-sliding-window \
    > logs/server_nsys.log 2>&1 &

NSYS_PID=$!
echo "[$(date)] nsys PID: $NSYS_PID"

# Wait for server to be ready
echo "[$(date)] Waiting for server to initialize (this may take a few minutes)..."
MAX_WAIT=600
WAITED=0
while ! curl -s "http://localhost:$SERVER_PORT/health" > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "[ERROR] Server did not start within $MAX_WAIT seconds"
        exit 1
    fi
    if ! kill -0 $NSYS_PID 2>/dev/null; then
        echo "[ERROR] nsys/server process died"
        cat logs/server_nsys.log | tail -50
        exit 1
    fi
    echo "  Waiting... ($WAITED s)"
done

echo "[$(date)] Server is ready!"
echo ""

# ============================================================
# PHASE 2: Start NVLink polling + record inference start
# ============================================================
echo "[$(date)] === Phase 2: Starting NVLink Polling ==="

# Record inference start time - THIS IS THE KEY REFERENCE FOR CORRELATION
PROFILE_START_WALL_NS=$(python3 -c "import time; print(time.time_ns())")
PROFILE_START_MONO_NS=$(python3 -c "import time; print(time.monotonic_ns())")
PROFILE_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%S.%NZ")

cat > "$OUTPUT_DIR/timing_profile_start.json" << EOF
{
  "name": "profile_start",
  "wall_clock_ns": $PROFILE_START_WALL_NS,
  "wall_clock_us": $(($PROFILE_START_WALL_NS / 1000)),
  "monotonic_ns": $PROFILE_START_MONO_NS,
  "monotonic_us": $(($PROFILE_START_MONO_NS / 1000)),
  "wall_clock_iso": "$PROFILE_START_ISO",
  "description": "Wall-clock when inference profiling started (after model load)"
}
EOF

echo "[$(date)] Inference start time: $PROFILE_START_ISO"
echo "  Wall-clock (us): $(($PROFILE_START_WALL_NS / 1000))"

# Start NVLink poller
NVLINK_BIN="./utilization"
if [ -x "$NVLINK_BIN" ]; then
    echo "[$(date)] Starting NVLink poller..."
    $NVLINK_BIN --gpu 0 --interval-ms "$NVLINK_INTERVAL_MS" --out "$OUTPUT_DIR/nvlink_trace.bin" --quiet &
    NVLINK_PID=$!
    echo "  NVLink poller PID: $NVLINK_PID"
else
    echo "[WARNING] NVLink binary not found at $NVLINK_BIN"
    NVLINK_PID=""
fi

# Give NVLink poller a moment to start
sleep 1
echo ""

# ============================================================
# PHASE 3: Run client workload
# ============================================================
echo "[$(date)] === Phase 3: Running Client Workload ==="

python3 benchmark.py --backend vllm \
    --model "$MODEL" \
    --request-rate "$REQUEST_RATE" \
    --num-prompts "$NUM_PROMPTS" \
    --dataset-name dummy \
    --long-prompts 0 \
    --long-prompt-len 32000

echo ""

# ============================================================
# PHASE 4: Stop profiling
# ============================================================
echo "[$(date)] === Phase 4: Stopping Profiling ==="

# Record profiling end time
PROFILE_END_WALL_NS=$(python3 -c "import time; print(time.time_ns())")
PROFILE_END_MONO_NS=$(python3 -c "import time; print(time.monotonic_ns())")
PROFILE_END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%S.%NZ")

cat > "$OUTPUT_DIR/timing_profile_end.json" << EOF
{
  "name": "profile_end",
  "wall_clock_ns": $PROFILE_END_WALL_NS,
  "wall_clock_us": $(($PROFILE_END_WALL_NS / 1000)),
  "monotonic_ns": $PROFILE_END_MONO_NS,
  "monotonic_us": $(($PROFILE_END_MONO_NS / 1000)),
  "wall_clock_iso": "$PROFILE_END_ISO"
}
EOF

echo "[$(date)] Profile end time: $PROFILE_END_ISO"

# Stop NVLink poller
if [ -n "$NVLINK_PID" ]; then
    echo "[$(date)] Stopping NVLink poller..."
    kill -INT $NVLINK_PID 2>/dev/null || true
    wait $NVLINK_PID 2>/dev/null || true
    NVLINK_PID=""
fi

# Stop nsys/server (nsys wraps the server process)
if [ -n "$NSYS_PID" ] && kill -0 $NSYS_PID 2>/dev/null; then
    echo "[$(date)] Stopping nsys + server..."
    # Send SIGINT to nsys, which will gracefully stop and generate report
    kill -INT $NSYS_PID 2>/dev/null || true
    echo "  Waiting for nsys to finish (generating report)..."
    wait $NSYS_PID 2>/dev/null || true
    NSYS_PID=""
fi

echo ""

# ============================================================
# PHASE 5: Export nsys to SQLite
# ============================================================
echo "[$(date)] === Phase 5: Exporting nsys to SQLite ==="

# Wait for nsys to finish writing
sleep 2

if [ -f "${NSYS_OUTPUT}.nsys-rep" ]; then
    echo "[$(date)] Exporting to SQLite (this may take a while for large traces)..."
    nsys export \
        --type sqlite \
        --output "${NSYS_OUTPUT}.sqlite" \
        "${NSYS_OUTPUT}.nsys-rep"
    echo "[$(date)] SQLite export complete!"
else
    echo "[WARNING] nsys report not found: ${NSYS_OUTPUT}.nsys-rep"
    echo "  Check logs/nsys.log for errors"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "=== Profiling Complete ==="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Files created:"
ls -lh "$OUTPUT_DIR/" 2>/dev/null || true
echo ""

DURATION_S=$((($PROFILE_END_WALL_NS - $PROFILE_START_WALL_NS) / 1000000000))
echo "Profiling duration: ${DURATION_S} seconds"
echo ""
echo "Next steps:"
echo "  python3 correlate_nsys_nvlink.py $OUTPUT_DIR"

