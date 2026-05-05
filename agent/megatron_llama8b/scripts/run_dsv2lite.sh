#!/bin/bash
set -e
# ============================================================
# CCL-Bench Agent – DeepSeek-V2-Lite run script
# ============================================================
# ---------- defaults (overridden by agent env vars) ----------
TP=${TP:-4}
DP=${DP:-4}
PP=${PP:-1}
EP=${EP:-4}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-true}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
SEQ_LEN=${SEQ_LEN:-1024}
TRAIN_ITERS=${TRAIN_ITERS:-5}
TRACE_DIR=${TRACE_DIR:-"/pscratch/sd/b/byungsoo/ccl-bench-traces/dsv2lite_agent"}

# ---------- derived values ----------
TOTAL_GPUS=$(( TP * DP * PP ))
GPUS_PER_NODE=4
NNODES=$(( (TOTAL_GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
NPROC_PER_NODE=$(( TOTAL_GPUS / NNODES ))
GLOBAL_BATCH_SIZE=8

echo "=== CCL-Bench DeepSeek-V2-Lite Run ==="
echo "TP=$TP  DP=$DP  PP=$PP  EP=$EP  total_gpus=$TOTAL_GPUS"
echo "NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE"
echo "MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE  GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
echo "ACTIVATION_CHECKPOINTING=$ACTIVATION_CHECKPOINTING"
echo "TRACE_DIR=$TRACE_DIR"
echo "======================================="

# ---------- environment ----------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=hsn0
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=6000

# ---------- paths ----------
BASE_DIR="/pscratch/sd/b/byungsoo"
REPO_DIR="$BASE_DIR/Megatron-LM"
TOKENIZER_PATH="$BASE_DIR/tokenizers/llama-3.1-8b"
TB_DIR="$TRACE_DIR/tb_logs"

cd "$REPO_DIR"

# ---------- clean & prepare trace dir ----------
mkdir -p "$TRACE_DIR"
rm -rf "$TRACE_DIR"/*.json "$TRACE_DIR"/*.yaml "$TRACE_DIR"/tb_logs "$TRACE_DIR"/torch_profile

# Write gpu_count for composite score metric
echo "$TOTAL_GPUS" > "$TRACE_DIR/gpu_count.txt"

# Write gpu_step_score config for composite metric
cat > "$TRACE_DIR/gpu_step_score_config.yaml" << SCORECONF
G0: 16
T0: 2.633
w: 0.5
SCORECONF

# Write trace metadata YAML for metric tools
cat > "$TRACE_DIR/trace_meta.yaml" << YAML
metric_source:
  traces:
    - json
workload:
  model:
    phase: training
    model_family: deepseek-v2-lite
    precision: bf16
    model_arch:
      num_params: 15700000000
      num_params_active: 2400000000
      num_params_embedding: 262144000
      num_layers: 27
      num_heads: 16
      head_dim: 128
  data:
    batch_size: $GLOBAL_BATCH_SIZE
    seq_len: $SEQ_LEN
  hardware:
    xpu_spec:
      type: GPU
      model: nvidia_a100
      memory_gb: 40
      total_count: $TOTAL_GPUS
      count_per_node: $GPUS_PER_NODE
YAML

# ---------- build model args ----------
MODEL_ARGS="--use-mcore-models \
    --num-layers 27 \
    --hidden-size 2048 \
    --ffn-hidden-size 10944 \
    --num-attention-heads 16 \
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

# ---------- build MoE args ----------
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

# Add expert parallelism if EP > 1
if [ "$EP" -gt 1 ]; then
    MOE_ARGS="$MOE_ARGS --expert-model-parallel-size $EP"
fi

# ---------- build training args ----------
TRAIN_ARGS="--micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
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
    --profile-step-end 4"

# Add activation checkpointing
if [ "$ACTIVATION_CHECKPOINTING" = "true" ] || [ "$ACTIVATION_CHECKPOINTING" = "True" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --recompute-activations"
fi

MEGATRON_ARGS="pretrain_gpt.py $MODEL_ARGS $MOE_ARGS $TRAIN_ARGS"

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

echo "Done."
