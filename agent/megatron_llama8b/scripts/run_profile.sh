#!/bin/bash
set -e

# ============================================================
# CCL-Bench Profiling Script for SM Utilization & MFU Analysis
# Supports: Llama-3.1-8B, DeepSeek-MoE-16B, DeepSeek-V3-236B
#
# Usage:
#   bash run_profile.sh --model llama8b   [--nodes 1]
#   bash run_profile.sh --model ds_moe16b [--nodes 2]
#   bash run_profile.sh --model ds_v3     [--nodes 16]
# ============================================================

# ---------- parse args ----------
MODEL=""
NODES_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --nodes) NODES_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "Usage: bash run_profile.sh --model {llama8b|ds_moe16b|ds_v3} [--nodes N]"
    exit 1
fi

# ---------- common env ----------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=hsn0
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=6000

BASE_DIR="/pscratch/sd/b/byungsoo"
REPO_DIR="$BASE_DIR/Megatron-LM"
TOKENIZER_PATH="$BASE_DIR/tokenizers/llama-3.1-8b"
TRACE_BASE="/pscratch/sd/b/byungsoo/ccl-bench-traces/profiling"
GPUS_PER_NODE=4
TRAIN_ITERS=5

cd "$REPO_DIR"

# ============================================================
# Model configurations
# ============================================================
case "$MODEL" in

llama8b)
    # Llama-3.1-8B: 32 layers, 4096 hidden, 32 heads, 8 KV heads
    # Batch=4, Seq=512
    TRACE_DIR="$TRACE_BASE/llama8b"
    TP=4; PP=4; DP=1; EP=1
    MBS=1; GBS=4; SEQ_LEN=512
    NUM_LAYERS=32; HIDDEN=4096; FFN_HIDDEN=14336
    NUM_HEADS=32; NUM_KV_HEADS=8; HEAD_DIM=128
    NUM_PARAMS=8030261248
    NUM_PARAMS_ACTIVE=$NUM_PARAMS
    NUM_PARAMS_EMBEDDING=524288000  # vocab_size * hidden_dim ≈ 128256 * 4096

    MODEL_ARGS="--use-mcore-models \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN \
        --ffn-hidden-size $FFN_HIDDEN \
        --num-attention-heads $NUM_HEADS \
        --group-query-attention \
        --num-query-groups $NUM_KV_HEADS \
        --kv-channels $HEAD_DIM \
        --seq-length $SEQ_LEN \
        --max-position-embeddings $SEQ_LEN \
        --position-embedding-type rope \
        --swiglu \
        --bf16 \
        --no-gradient-accumulation-fusion \
        --no-masked-softmax-fusion \
        --no-bias-gelu-fusion \
        --no-bias-dropout-fusion"
    MOE_ARGS=""
    ;;

ds_moe16b)
    # DeepSeek-MoE-16B: 28 layers, 2048 hidden, 16 heads
    # 8 experts, top-2 routing (Mixtral-like architecture)
    # Batch=8, Seq=1024
    TRACE_DIR="$TRACE_BASE/ds_moe16b"
    TP=4; PP=4; DP=1; EP=1
    MBS=1; GBS=8; SEQ_LEN=1024
    NUM_LAYERS=28; HIDDEN=2048; FFN_HIDDEN=5504
    NUM_HEADS=16; NUM_KV_HEADS=16; HEAD_DIM=128
    NUM_PARAMS=12900000000
    NUM_PARAMS_ACTIVE=3800000000   # ~2.8B active per token
    NUM_PARAMS_EMBEDDING=262144000 # vocab_size * hidden_dim ≈ 128000 * 2048

    MODEL_ARGS="--use-mcore-models \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN \
        --ffn-hidden-size $FFN_HIDDEN \
        --num-attention-heads $NUM_HEADS \
        --kv-channels $HEAD_DIM \
        --seq-length $SEQ_LEN \
        --max-position-embeddings $SEQ_LEN \
        --position-embedding-type rope \
        --swiglu \
        --bf16 \
        --normalization RMSNorm \
        --disable-bias-linear \
        --no-masked-softmax-fusion \
        --no-position-embedding \
        --untie-embeddings-and-output-weights \
        --no-gradient-accumulation-fusion \
        --no-bias-gelu-fusion \
        --no-bias-dropout-fusion"
    MOE_ARGS="--num-experts 8 \
        --moe-router-topk 2 \
        --moe-router-load-balancing-type aux_loss \
        --moe-aux-loss-coeff 1e-2 \
        --moe-grouped-gemm \
        --moe-token-dispatcher-type alltoall"
    ;;

ds_v3)
    # DeepSeek-V3-236B (simplified): 60 layers, 7168 hidden, 56 heads
    # 256 experts, top-8 routing, ~37B active params per token
    # Batch=64, Seq=1024
    TRACE_DIR="$TRACE_BASE/ds_v3"
    TP=4; PP=8; DP=1; EP=4
    MBS=1; GBS=64; SEQ_LEN=1024
    NUM_LAYERS=60; HIDDEN=7168; FFN_HIDDEN=18432
    NUM_HEADS=56; NUM_KV_HEADS=8; HEAD_DIM=128
    NUM_PARAMS=236000000000
    NUM_PARAMS_ACTIVE=37000000000   # ~37B active per token
    NUM_PARAMS_EMBEDDING=917504000  # vocab_size * hidden_dim ≈ 128000 * 7168

    MODEL_ARGS="--use-mcore-models \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN \
        --ffn-hidden-size $FFN_HIDDEN \
        --num-attention-heads $NUM_HEADS \
        --group-query-attention \
        --num-query-groups $NUM_KV_HEADS \
        --kv-channels $HEAD_DIM \
        --seq-length $SEQ_LEN \
        --max-position-embeddings $SEQ_LEN \
        --position-embedding-type rope \
        --swiglu \
        --bf16 \
        --normalization RMSNorm \
        --disable-bias-linear \
        --no-masked-softmax-fusion \
        --no-position-embedding \
        --untie-embeddings-and-output-weights"
    MOE_ARGS="--num-experts 256 \
        --moe-router-topk 8 \
        --moe-router-load-balancing-type aux_loss \
        --moe-aux-loss-coeff 1e-2 \
        --moe-grouped-gemm \
        --moe-token-dispatcher-type alltoall"
    ;;

llama8b_xlarge)
    # Llama-3.1-8B with very large batch + long sequence
    TRACE_DIR="$TRACE_BASE/llama8b_xlarge"
    TP=4; PP=4; DP=1; EP=1
    MBS=1; GBS=64; SEQ_LEN=4096
    NUM_LAYERS=32; HIDDEN=4096; FFN_HIDDEN=14336
    NUM_HEADS=32; NUM_KV_HEADS=8; HEAD_DIM=128
    NUM_PARAMS=8030261248
    NUM_PARAMS_ACTIVE=$NUM_PARAMS
    NUM_PARAMS_EMBEDDING=524288000

    MODEL_ARGS="--use-mcore-models \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN \
        --ffn-hidden-size $FFN_HIDDEN \
        --num-attention-heads $NUM_HEADS \
        --group-query-attention \
        --num-query-groups $NUM_KV_HEADS \
        --kv-channels $HEAD_DIM \
        --seq-length $SEQ_LEN \
        --max-position-embeddings $SEQ_LEN \
        --position-embedding-type rope \
        --swiglu \
        --bf16 \
        --no-gradient-accumulation-fusion \
        --no-masked-softmax-fusion \
        --no-bias-gelu-fusion \
        --no-bias-dropout-fusion"
    MOE_ARGS=""
    ;;
llama8b_large)
    # Llama-3.1-8B with large batch + long sequence for higher MFU
    # BS=32, Seq=2048 gives much higher arithmetic intensity
    TRACE_DIR="$TRACE_BASE/llama8b_large"
    TP=4; PP=4; DP=1; EP=1
    MBS=1; GBS=32; SEQ_LEN=2048
    NUM_LAYERS=32; HIDDEN=4096; FFN_HIDDEN=14336
    NUM_HEADS=32; NUM_KV_HEADS=8; HEAD_DIM=128
    NUM_PARAMS=8030261248
    NUM_PARAMS_ACTIVE=$NUM_PARAMS
    NUM_PARAMS_EMBEDDING=524288000

    MODEL_ARGS="--use-mcore-models \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN \
        --ffn-hidden-size $FFN_HIDDEN \
        --num-attention-heads $NUM_HEADS \
        --group-query-attention \
        --num-query-groups $NUM_KV_HEADS \
        --kv-channels $HEAD_DIM \
        --seq-length $SEQ_LEN \
        --max-position-embeddings $SEQ_LEN \
        --position-embedding-type rope \
        --swiglu \
        --bf16 \
        --no-gradient-accumulation-fusion \
        --no-masked-softmax-fusion \
        --no-bias-gelu-fusion \
        --no-bias-dropout-fusion"
    MOE_ARGS=""
    ;;

mixtral8x7b)
    # Mixtral 8x7B: 32 layers, 4096 hidden, 32 heads, 8 KV heads
    # 8 experts, top-2 routing
    # TP=1, PP=4, DP=4, EP=4 = 16 GPUs
    TRACE_DIR="$TRACE_BASE/mixtral8x7b"
    TP=2; PP=2; DP=4; EP=4
    MBS=1; GBS=8; SEQ_LEN=1024
    NUM_LAYERS=32; HIDDEN=4096; FFN_HIDDEN=14336
    NUM_HEADS=32; NUM_KV_HEADS=8; HEAD_DIM=128
    NUM_PARAMS=46700000000
    NUM_PARAMS_ACTIVE=12900000000
    NUM_PARAMS_EMBEDDING=524288000

    MODEL_ARGS="--use-mcore-models \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN \
        --ffn-hidden-size $FFN_HIDDEN \
        --num-attention-heads $NUM_HEADS \
        --group-query-attention \
        --num-query-groups $NUM_KV_HEADS \
        --kv-channels $HEAD_DIM \
        --seq-length $SEQ_LEN \
        --max-position-embeddings $SEQ_LEN \
        --position-embedding-type rope \
        --swiglu \
        --bf16 \
        --normalization RMSNorm \
        --disable-bias-linear \
        --no-gradient-accumulation-fusion \
        --no-masked-softmax-fusion \
        --no-bias-gelu-fusion \
        --no-bias-dropout-fusion \
        --no-position-embedding \
        --untie-embeddings-and-output-weights"
    MOE_ARGS="--num-experts 8 \
        --moe-router-topk 2 \
        --moe-router-load-balancing-type aux_loss \
        --moe-aux-loss-coeff 1e-2 \
        --moe-grouped-gemm \
        --moe-token-dispatcher-type alltoall"
    ;;
dsv2lite)
    # DeepSeek-V2-Lite: 27 layers, 2048 hidden, MLA, 64 experts top-6
    # 15.7B total, 2.4B active per token
    TRACE_DIR="$TRACE_BASE/dsv2lite"
    TP=4; PP=1; DP=4; EP=4
    MBS=1; GBS=8; SEQ_LEN=1024
    NUM_LAYERS=27; HIDDEN=2048; FFN_HIDDEN=10944
    NUM_HEADS=16; NUM_KV_HEADS=16; HEAD_DIM=128
    NUM_PARAMS=15700000000
    NUM_PARAMS_ACTIVE=2400000000
    NUM_PARAMS_EMBEDDING=262144000

    MODEL_ARGS="--use-mcore-models \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN \
        --ffn-hidden-size $FFN_HIDDEN \
        --num-attention-heads $NUM_HEADS \
        --kv-channels 16 \
        --seq-length $SEQ_LEN \
        --max-position-embeddings $SEQ_LEN \
        --position-embedding-type rope \
        --no-rope-fusion \
        --swiglu \
        --bf16 \
        --normalization RMSNorm \
        --norm-epsilon 1e-6 \
        --disable-bias-linear \
        --no-masked-softmax-fusion \
        --no-gradient-accumulation-fusion \
        --no-bias-gelu-fusion \
        --no-bias-dropout-fusion \
        --untie-embeddings-and-output-weights \
        --no-position-embedding \
        --multi-latent-attention \
        --kv-lora-rank 512 \
        --v-head-dim 128 \
        --qk-head-dim 128 \
        --qk-layernorm \
        --qk-pos-emb-head-dim 64 \
        --rotary-base 10000 \
        --rotary-scaling-factor 40 \
        --mscale 0.707 \
        --mscale-all-dim 0.707 \
        --make-vocab-size-divisible-by 3200 \
        --attention-softmax-in-fp32 \
        --sequence-parallel"
    MOE_ARGS="--num-experts 64 \
        --moe-layer-freq '([0]+[1]*26)' \
        --moe-ffn-hidden-size 1408 \
        --moe-grouped-gemm \
        --moe-router-topk 6 \
        --moe-router-score-function softmax \
        --moe-router-topk-scaling-factor 1.0 \
        --moe-router-pre-softmax \
        --moe-shared-expert-intermediate-size 2816 \
        --moe-aux-loss-coeff 1e-3 \
        --moe-token-dispatcher-type alltoall \
        --moe-token-drop-policy probs \
        --moe-router-load-balancing-type seq_aux_loss"
    ;;
llama8b_match)
    # Llama-3.1-8B: BS=32, S=1024 matching TPU config
    TRACE_DIR="$TRACE_BASE/llama8b_match"
    TP=4; PP=1; DP=2; EP=1
    MBS=1; GBS=32; SEQ_LEN=1024
    NUM_LAYERS=32; HIDDEN=4096; FFN_HIDDEN=14336
    NUM_HEADS=32; NUM_KV_HEADS=8; HEAD_DIM=128
    NUM_PARAMS=8030261248
    NUM_PARAMS_ACTIVE=$NUM_PARAMS
    NUM_PARAMS_EMBEDDING=524288000

    MODEL_ARGS="--use-mcore-models \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN \
        --ffn-hidden-size $FFN_HIDDEN \
        --num-attention-heads $NUM_HEADS \
        --group-query-attention \
        --num-query-groups $NUM_KV_HEADS \
        --kv-channels $HEAD_DIM \
        --seq-length $SEQ_LEN \
        --max-position-embeddings $SEQ_LEN \
        --position-embedding-type rope \
        --swiglu \
        --bf16 \
        --no-gradient-accumulation-fusion \
        --no-masked-softmax-fusion \
        --no-bias-gelu-fusion \
        --no-bias-dropout-fusion \
        --use-distributed-optimizer"
    MOE_ARGS=""
    ;;
*)
    echo "Unknown model: $MODEL. Use llama8b, ds_moe16b, or ds_v3"
    exit 1
    ;;
esac

# ---------- derived values ----------
TOTAL_GPUS=$(( TP * DP * PP ))
if [ -n "$NODES_OVERRIDE" ]; then
    NNODES=$NODES_OVERRIDE
else
    NNODES=$(( (TOTAL_GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
fi
NPROC_PER_NODE=$(( TOTAL_GPUS / NNODES ))

# For MoE with expert parallelism
if [ "$EP" -gt 1 ]; then
    MOE_ARGS="$MOE_ARGS --expert-model-parallel-size $EP"
fi

echo "=== CCL-Bench Profiling: $MODEL ==="
echo "TP=$TP  DP=$DP  PP=$PP  EP=$EP  total_gpus=$TOTAL_GPUS"
echo "NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE"
echo "MBS=$MBS  GBS=$GBS  SEQ_LEN=$SEQ_LEN"
echo "TRAIN_ITERS=$TRAIN_ITERS"
echo "TRACE_DIR=$TRACE_DIR"
echo "======================================"

# ---------- clean & prepare trace dir ----------
mkdir -p "$TRACE_DIR"
rm -rf "$TRACE_DIR"/*.json "$TRACE_DIR"/*.yaml "$TRACE_DIR"/tb_logs "$TRACE_DIR"/torch_profile "$TRACE_DIR"/gpu_count.txt

echo "$TOTAL_GPUS" > "$TRACE_DIR/gpu_count.txt"

# ---------- write workload YAML for metric tools ----------
TB_DIR="$TRACE_DIR/tb_logs"

cat > "$TRACE_DIR/trace_meta.yaml" << YAML
metric_source:
  traces:
    - json
workload:
  model:
    phase: training
    model_family: $MODEL
    precision: bf16
    model_arch:
      num_params: $NUM_PARAMS
      num_params_active: $NUM_PARAMS_ACTIVE
      num_params_embedding: $NUM_PARAMS_EMBEDDING
      num_layers: $NUM_LAYERS
      num_heads: $NUM_HEADS
      head_dim: $HEAD_DIM
  data:
    batch_size: $GBS
    seq_len: $SEQ_LEN
  hardware:
    xpu_spec:
      type: GPU
      model: nvidia_a100
      memory_gb: 40
      total_count: $TOTAL_GPUS
      count_per_node: $GPUS_PER_NODE
YAML

# ---------- build megatron args ----------
MEGATRON_ARGS="pretrain_gpt.py \
    $MODEL_ARGS \
    $MOE_ARGS \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --train-iters $TRAIN_ITERS \
    --lr 0.00015 \
    --lr-decay-style constant \
    --lr-warmup-iters 0 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --mock-data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    --log-interval 1 \
    --log-throughput \
    --eval-interval 100 \
    --eval-iters 0 \
    --tensorboard-dir $TB_DIR \
    --profile \
    --use-pytorch-profiler \
    --profile-step-start 2 \
    --profile-step-end 4 \
    --recompute-activations"

# ---------- launch ----------
if [ "$NNODES" -eq 1 ]; then
    torchrun \
        --nproc_per_node "$NPROC_PER_NODE" \
        --nnodes 1 \
        --master_addr localhost \
        --master_port "$MASTER_PORT" \
        $MEGATRON_ARGS
else
    srun --export=ALL bash -c "
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export NCCL_SOCKET_IFNAME=hsn0
        torchrun \
            --nproc_per_node $NPROC_PER_NODE \
            --nnodes $NNODES \
            --node_rank \$SLURM_PROCID \
            --master_addr $MASTER_ADDR \
            --master_port $MASTER_PORT \
            $MEGATRON_ARGS
    " || true
fi

# ---------- collect traces ----------
echo "Searching for profiler traces..."

PROFILE_DIR="$TRACE_DIR/torch_profile"
FOUND=0

if [ -d "$PROFILE_DIR" ]; then
    for f in "$PROFILE_DIR"/rank-*.json.gz; do
        if [ -f "$f" ]; then
            RANK=$(basename "$f" | grep -oP '\d+' | head -1)
            gunzip -c "$f" > "$TRACE_DIR/rank${RANK}_trace.json"
            FOUND=$((FOUND + 1))
            echo "  Extracted: $f -> rank${RANK}_trace.json"
        fi
    done
    for f in "$PROFILE_DIR"/rank-*.json; do
        if [ -f "$f" ]; then
            RANK=$(basename "$f" | grep -oP '\d+' | head -1)
            cp "$f" "$TRACE_DIR/rank${RANK}_trace.json"
            FOUND=$((FOUND + 1))
            echo "  Copied: $f -> rank${RANK}_trace.json"
        fi
    done
fi

if [ "$FOUND" -eq 0 ]; then
    echo "WARNING: No profiler traces found!"
    exit 1
fi

echo "Collected $FOUND trace files into $TRACE_DIR"

# ---------- verify ProfilerStep markers ----------
python3 -c "
import json, sys
f = '$TRACE_DIR/rank0_trace.json'
data = json.load(open(f))
events = data.get('traceEvents', [])
steps = [e for e in events if 'ProfilerStep' in e.get('name', '')]
print(f'ProfilerStep events: {len(steps)}')
if steps:
    for s in steps:
        print(f'  {s[\"name\"]} dur={s.get(\"dur\",\"?\")}us')
else:
    print('WARNING: No ProfilerStep markers found!')
    sys.exit(1)
" || echo "Warning: could not verify ProfilerStep markers"

# ---------- run metrics ----------
TOOLS_DIR="/global/u1/b/byungsoo/cclbench-agent/ccl-bench/tools"
RESULTS_FILE="$TRACE_DIR/metrics_summary.txt"

run_metric() {
    local name=$1
    local result
    result=$(python3 "$TOOLS_DIR/main.py" --trace "$TRACE_DIR" --metric "$name" 2>/dev/null) || result="FAILED"
    echo "$name: $result"
    echo "$name: $result" >> "$RESULTS_FILE"
}

echo ""
echo "============================================================"
echo "  Hardware Efficiency Analysis: $MODEL"
echo "  GPUs: $TOTAL_GPUS (TP=$TP PP=$PP DP=$DP EP=$EP)"
echo "  Batch=$GBS  Seq=$SEQ_LEN  MBS=$MBS"
echo "============================================================"

cat /dev/null > "$RESULTS_FILE"
echo "model: $MODEL" >> "$RESULTS_FILE"
echo "total_gpus: $TOTAL_GPUS" >> "$RESULTS_FILE"
echo "tp: $TP  pp: $PP  dp: $DP  ep: $EP" >> "$RESULTS_FILE"
echo "batch: $GBS  seq: $SEQ_LEN  mbs: $MBS" >> "$RESULTS_FILE"
echo "---" >> "$RESULTS_FILE"

echo ""
echo "--- 1. Overall Efficiency ---"
run_metric avg_step_time
run_metric mfu

echo ""
echo "--- 2. Time Breakdown ---"
run_metric aggregate_gpu_utilization
run_metric communication_fraction
run_metric communication_overlap_ratio

echo ""
echo "--- 3. Kernel Analysis ---"
run_metric total_kernel_time
run_metric total_communication_time
run_metric dominant_kernel_concentration
run_metric load_imbalance_ratio

echo ""
echo "============================================================"
echo "  Results saved to: $RESULTS_FILE"
echo "  Traces: $TRACE_DIR"
echo "============================================================"
