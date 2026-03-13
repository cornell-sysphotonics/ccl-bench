#!/bin/bash
#SBATCH --job-name=deepseek-qwen2-16gpu-profile
#SBATCH --account=m4999
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=logs/deepseek_qwen2_profile_%j_%N.out
#SBATCH --error=logs/deepseek_qwen2_profile_%j_%N.err
#SBATCH --image=nvcr.io/nvidia/pytorch:25.03-py3

set -euo pipefail
mkdir -p logs

# Master node/port for torchrun
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=6000

# Launch one container per node, and run torchrun inside
srun --export=ALL shifter \
  --volume=/pscratch/sd/a/ab2352:/scratch \
  bash -c '
    set -euo pipefail

    # Install dependencies
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet transformers==4.36.0

    # Environment variables
    export PYTORCH_JIT=0
    export TORCH_COMPILE_DISABLE=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_DEBUG=INFO
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # =========================
    # PATHS
    # =========================
    cd /scratch/megatron-lm-new
    TOKENIZER_PATH="/scratch/tokenizers/deepseek-r1-distill-qwen-7b"
    DATA_PREFIX="/scratch/datasets/processed/wikitext_qwen2_text_document_text_document"

    # =========================
    # DISTRIBUTED (16 GPUs = 4 nodes × 4 GPUs)
    # =========================
    GPUS_PER_NODE=4
    NUM_NODES=4
    NODE_RANK=$SLURM_PROCID

    PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

    # =========================
    # PARALLEL CONFIG: DP=4, TP=4
    # =========================
    TP_SIZE=4
    PP_SIZE=1
    CP_SIZE=1
    EP_SIZE=1

    MICRO_BATCH_SIZE=2
    GLOBAL_BATCH_SIZE=64

    # =========================
    # DEEPSEEK-QWEN2-7B SHAPE
    # =========================
    NUM_LAYERS=28
    HIDDEN_SIZE=3584
    FFN_HIDDEN_SIZE=18944
    NUM_HEADS=28
    NUM_QUERY_GROUPS=4
    KV_CHANNELS=128
    VOCAB_SIZE=151665

    SEQ_LENGTH=4096
    MAX_POSITION_EMBEDDINGS=131072
    ROTARY_BASE=10000

    # Training settings - PROFILING ONLY
    TRAIN_ITERS=40

    DATA_CACHE_PATH="${PWD}/data_cache_wikitext_qwen2_16gpu"
    mkdir -p "$DATA_CACHE_PATH"

    echo "========================================="
    echo "DeepSeek-Qwen2-7B (mcore) - 16 GPUs PROFILE"
    echo "========================================="
    echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
    echo "Node rank: ${NODE_RANK} / ${NUM_NODES}"
    echo "TP=${TP_SIZE}  PP=${PP_SIZE}  CP=${CP_SIZE}"
    echo "seq=${SEQ_LENGTH}  max_pos=${MAX_POSITION_EMBEDDINGS}"
    echo "tokenizer=${TOKENIZER_PATH}"
    echo "data=${DATA_PREFIX}"
    echo "========================================="

    torchrun \
      --nproc_per_node "${GPUS_PER_NODE}" \
      --nnodes "${NUM_NODES}" \
      --node_rank "${NODE_RANK}" \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      "$PRETRAIN_SCRIPT_PATH" \
      --use-mcore-models \
      --vocab-size "${VOCAB_SIZE}" \
      --num-layers "${NUM_LAYERS}" \
      --hidden-size "${HIDDEN_SIZE}" \
      --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
      --num-attention-heads "${NUM_HEADS}" \
      --group-query-attention \
      --num-query-groups "${NUM_QUERY_GROUPS}" \
      --kv-channels "${KV_CHANNELS}" \
      --normalization RMSNorm \
      --norm-epsilon 1e-6 \
      --swiglu \
      --untie-embeddings-and-output-weights \
      --disable-bias-linear \
      --seq-length "${SEQ_LENGTH}" \
      --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}" \
      --position-embedding-type rope \
      --rotary-base "${ROTARY_BASE}" \
      --attention-dropout 0.0 \
      --hidden-dropout 0.0 \
      --attention-backend fused \
      --micro-batch-size "${MICRO_BATCH_SIZE}" \
      --global-batch-size "${GLOBAL_BATCH_SIZE}" \
      --train-iters "${TRAIN_ITERS}" \
      --lr 3e-4 \
      --min-lr 3e-5 \
      --lr-decay-style cosine \
      --lr-warmup-iters 10 \
      --clip-grad 1.0 \
      --weight-decay 0.1 \
      --adam-beta1 0.9 \
      --adam-beta2 0.95 \
      --bf16 \
      --recompute-granularity full \
      --recompute-method uniform \
      --recompute-num-layers 1 \
      --tensor-model-parallel-size "${TP_SIZE}" \
      --pipeline-model-parallel-size "${PP_SIZE}" \
      --expert-model-parallel-size "${EP_SIZE}" \
      --context-parallel-size "${CP_SIZE}" \
      --data-path "${DATA_PREFIX}" \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model "${TOKENIZER_PATH}" \
      --data-cache-path "${DATA_CACHE_PATH}" \
      --split "100,0,0" \
      --num-workers 2 \
      --log-interval 1 \
      --log-throughput \
      --eval-iters 0
  '
