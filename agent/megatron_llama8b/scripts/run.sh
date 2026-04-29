#!/bin/bash
set -e

# ============================================================
# CCL-Bench Agent – Megatron-LM Llama-3.1-8B run script
# ============================================================

# ---------- defaults (overridden by agent env vars) ----------
TP=${TP:-4}
DP=${DP:-2}
PP=${PP:-1}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-false}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
SEQ_LEN=${SEQ_LEN:-1024}
TRAIN_ITERS=${TRAIN_ITERS:-5}
TRACE_DIR=${TRACE_DIR:-"/pscratch/sd/b/byungsoo/ccl-bench-traces/megatron_llama8b"}

# ---------- derived values ----------
TOTAL_GPUS=$(( TP * DP * PP ))
GPUS_PER_NODE=4
NNODES=$(( (TOTAL_GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
NPROC_PER_NODE=$(( TOTAL_GPUS / NNODES ))
GLOBAL_BATCH_SIZE=$(( DP * MICRO_BATCH_SIZE ))

echo "=== CCL-Bench Megatron Run ==="
echo "TP=$TP  DP=$DP  PP=$PP  total_gpus=$TOTAL_GPUS"
echo "NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE"
echo "MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE  GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
echo "ACTIVATION_CHECKPOINTING=$ACTIVATION_CHECKPOINTING"
echo "TRACE_DIR=$TRACE_DIR"
echo "=============================="

# ---------- environment ----------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=hsn0
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=6000

# ---------- paths ----------
BASE_DIR="/pscratch/sd/b/byungsoo"
REPO_DIR="$BASE_DIR/Megatron-LM"
TOKENIZER_PATH="$BASE_DIR/tokenizers/llama-3.1-8b"

# Megatron writes profiler traces to {tensorboard_dir}/../torch_profile/
# We set tensorboard_dir so traces land in a predictable place.
TB_DIR="$TRACE_DIR/tb_logs"

cd "$REPO_DIR"

# ---------- clean trace dir ----------
mkdir -p "$TRACE_DIR"
rm -rf "$TRACE_DIR"/*.json "$TRACE_DIR"/trace_meta.yaml "$TRACE_DIR"/tb_logs "$TRACE_DIR"/torch_profile

# ---------- build args ----------
MEGATRON_ARGS="pretrain_gpt.py \
    --use-mcore-models \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 128 \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --position-embedding-type rope \
    --swiglu \
    --bf16 \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters $TRAIN_ITERS \
    --lr 0.00015 \
    --lr-decay-style constant \
    --lr-warmup-iters 0 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
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

if [ "$ACTIVATION_CHECKPOINTING" = "true" ]; then
    MEGATRON_ARGS="$MEGATRON_ARGS --recompute-activations"
fi

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
    "
fi

# ---------- collect traces ----------
# Megatron writes to {tensorboard_dir}/../torch_profile/rank-{R}.json.gz
# We need flat rank{R}_trace.json files in $TRACE_DIR

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

# Also check if traces ended up in other locations
if [ "$FOUND" -eq 0 ]; then
    for search_dir in \
        "$REPO_DIR/torch_profile" \
        "$REPO_DIR/profile_traces"; do
        if [ -d "$search_dir" ]; then
            while IFS= read -r -d '' f; do
                RANK=$(basename "$f" | grep -oP '\d+' | head -1)
                if [[ "$f" == *.gz ]]; then
                    gunzip -c "$f" > "$TRACE_DIR/rank${RANK}_trace.json"
                else
                    cp "$f" "$TRACE_DIR/rank${RANK}_trace.json"
                fi
                FOUND=$((FOUND + 1))
                echo "  Found: $f"
            done < <(find "$search_dir" -maxdepth 3 \( -name "*.json" -o -name "*.json.gz" \) -print0 2>/dev/null)
        fi
    done
fi

if [ "$FOUND" -eq 0 ]; then
    echo "WARNING: No profiler traces found!"
    echo "Checked: $PROFILE_DIR, $REPO_DIR/torch_profile, $REPO_DIR/profile_traces"
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

# ---------- write trace metadata ----------
cat > "$TRACE_DIR/trace_meta.yaml" << 'YAML'
metric_source:
  traces:
    - json
YAML

echo "Done."