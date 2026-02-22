#!/bin/bash
#SBATCH --nodes=2                  # 2 nodes (4 GPUs each = 8 total)
#SBATCH --ntasks-per-node=1        # 1 task (torchrun) per node
#SBATCH --gpus-per-node=4          # Perlmutter: 4 GPUs per node
#SBATCH --cpus-per-task=64
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=00:15:00
#SBATCH --account=m4999
#SBATCH --job-name=mistral_8gpu
#SBATCH --output=logs/mistral_8gpu_%j_%N.out
#SBATCH --error=logs/mistral_8gpu_%j_%N.err
#SBATCH --image=nvcr.io/nvidia/pytorch:25.03-py3

set -euo pipefail
mkdir -p logs

# Master node/port for torchrun
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=6000

# Launch one container per node, and run torchrun inside.
srun --export=ALL shifter \
  --volume=/pscratch/sd/a/ab2352:/scratch \
  bash -c '
    set -euo pipefail

    # ---- install HF tokenizer support (pin to your config.json)
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet transformers==4.36.0 sentencepiece protobuf

    # ---- disable compile/JIT (matches what worked for you)
    export PYTORCH_JIT=0
    export TORCH_COMPILE_DISABLE=1

    # ---- NCCL / memory knobs
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_DEBUG=INFO
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # =========================
    # PATHS (SAME AS YOUR 4GPU SCRIPT)
    # =========================
    cd /scratch/megatron-lm-new
    TOKENIZER_PATH="/scratch/tokenizers/mistral-7b"
    DATA_PREFIX="/scratch/datasets/processed/wiki_mistral_text_document"

    # =========================
    # DISTRIBUTED (CHANGED FOR 8 GPU)
    # =========================
    GPUS_PER_NODE=4
    NUM_NODES=2
    NODE_RANK=$SLURM_PROCID

    PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

    # =========================
    # PARALLEL CONFIG (KEEP FLAGS SAME)
    # =========================
    TP_SIZE=2
    DP_SIZE=2
    CP_SIZE=1
    PP_SIZE=2

    MICRO_BATCH_SIZE=1
    GLOBAL_BATCH_SIZE=32

    # =========================
    # MISTRAL-7B SHAPE (SAME)
    # =========================
    NUM_LAYERS=32
    HIDDEN_SIZE=4096
    FFN_HIDDEN_SIZE=14336
    NUM_HEADS=32
    NUM_QUERY_GROUPS=8
    KV_CHANNELS=128

    SEQ_LENGTH=1024
    MAX_POSITION_EMBEDDINGS=32768
    ROPE_THETA=1000000

    WINDOW_SIZE="4096,0"
    WINDOW_SKIP_FREQ="4"

    DATA_CACHE_PATH="${PWD}/data_cache_wikitext_mistral_8gpu"
    mkdir -p "$DATA_CACHE_PATH"

    echo "========================================="
    echo "Mistral-7B (mcore) SBATCH - 8 GPUs"
    echo "========================================="
    echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
    echo "Node rank: ${NODE_RANK} / ${NUM_NODES}"
    echo "TP=${TP_SIZE}  DP=${DP_SIZE}  CP=${CP_SIZE}  PP=${PP_SIZE}"
    echo "seq=${SEQ_LENGTH}  max_pos=${MAX_POSITION_EMBEDDINGS}"
    echo "window=${WINDOW_SIZE} skip_freq=${WINDOW_SKIP_FREQ}"
    echo "tokenizer=${TOKENIZER_PATH}"
    echo "data=${DATA_PREFIX}"
    echo "========================================="

    # ---- optional: confirm window flags exist
    python "$PRETRAIN_SCRIPT_PATH" --help | grep -i "window" || true

    torchrun \
      --nproc_per_node "${GPUS_PER_NODE}" \
      --nnodes "${NUM_NODES}" \
      --node_rank "${NODE_RANK}" \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      "$PRETRAIN_SCRIPT_PATH" \
      --use-mcore-models \
      --num-layers "${NUM_LAYERS}" \
      --hidden-size "${HIDDEN_SIZE}" \
      --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
      --num-attention-heads "${NUM_HEADS}" \
      --group-query-attention \
      --num-query-groups "${NUM_QUERY_GROUPS}" \
      --kv-channels "${KV_CHANNELS}" \
      --seq-length "${SEQ_LENGTH}" \
      --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}" \
      --normalization RMSNorm \
      --position-embedding-type rope \
      --rotary-base "${ROPE_THETA}" \
      --rotary-percent 1.0 \
      --attention-dropout 0.0 \
      --hidden-dropout 0.0 \
      --swiglu \
      --init-method-std 0.02 \
      --attention-backend fused \
      --apply-layernorm-1p \
      --untie-embeddings-and-output-weights \
      --disable-bias-linear \
      --window-size "${WINDOW_SIZE}" \
      --window-attn-skip-freq "${WINDOW_SKIP_FREQ}" \
      --micro-batch-size "${MICRO_BATCH_SIZE}" \
      --global-batch-size "${GLOBAL_BATCH_SIZE}" \
      --train-iters 50 \
      --lr 0.00015 \
      --min-lr 0.00001 \
      --lr-decay-style cosine \
      --lr-warmup-iters 10 \
      --clip-grad 1.0 \
      --weight-decay 0.1 \
      --adam-beta1 0.9 \
      --adam-beta2 0.95 \
      --bf16 \
      --recompute-granularity full \
      --recompute-method uniform \
      --recompute-num-layers 32 \
      --use-distributed-optimizer \
      --tensor-model-parallel-size "${TP_SIZE}" \
      --context-parallel-size "${CP_SIZE}" \
      --data-path "${DATA_PREFIX}" \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model "${TOKENIZER_PATH}" \
      --data-cache-path "${DATA_CACHE_PATH}" \
      --split "100,0,0" \
      --num-workers 2 \
      --log-interval 1 \
      --log-throughput \
      --eval-iters 0 \
  '

