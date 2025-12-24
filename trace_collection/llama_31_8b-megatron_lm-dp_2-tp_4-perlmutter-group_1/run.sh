#!/bin/bash
#SBATCH --nodes=2                  # 2 nodes (4 GPUs each = 8 total)
#SBATCH --ntasks-per-node=1        # 1 task per node
#SBATCH --gpus-per-node=4          # 4 GPUs per node
#SBATCH --cpus-per-task=64
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --account=m4999
#SBATCH --job-name=llama_8gpu
#SBATCH --output=logs/llama_8gpu_%j_%N.out
#SBATCH --error=logs/llama_8gpu_%j_%N.err
#SBATCH --image=nvcr.io/nvidia/pytorch:25.03-py3

# Create logs directory
mkdir -p logs

# Get list of nodes
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=6000

# Run training in Shifter container
srun --export=ALL shifter \
  --volume=/pscratch/sd/a/ab2352:/scratch \
  bash -c '
    # Install transformers in container
    pip install --quiet transformers || { echo "pip install failed"; exit 1; }
    
    # Change to megatron directory
    cd /scratch/megatron-lm
    
    # DISABLE JIT COMPILATION
    export PYTORCH_JIT=0
    export TORCH_COMPILE_DISABLE=1
    
    # Memory optimization
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_DEBUG=INFO
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    # PATHS
    TOKENIZER_PATH="/scratch/tokenizers/llama-3.1-8b"
    DATA_PREFIX="/scratch/datasets/processed/wikitext_llama_text_document_text_document"
    
    # Distributed training setup - 8 GPUs across 2 nodes
    GPUS_PER_NODE=4
    NUM_NODES=2
    MASTER_PORT=6000
    
    # Get node rank from SLURM
    NODE_RANK=$SLURM_PROCID
    WORLD_SIZE=8
    
    PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"
    
    # MODEL PARALLEL CONFIG - DP=2, TP=4
    TP_SIZE=4              # Tensor parallel = 4
    DP_SIZE=2              # Data parallel = 2 (derived: 8 GPUs / TP=4 = 2)
    CP_SIZE=1
    PP_SIZE=1
    MICRO_BATCH_SIZE=1
    GLOBAL_BATCH_SIZE=32    # Increased from 4 to 8 (2x for DP=2)
    NUM_LAYERS=32
    SEQ_LENGTH=1024
    MAX_POSITION_EMBEDDINGS=1024
    
    DATA_CACHE_PATH="${PWD}/data_cache_wikitext"
    mkdir -p "$DATA_CACHE_PATH"
    
    echo "========================================="
    echo "Llama-3.1-8B Training with WikiText - 8 GPUs"
    echo "========================================="
    echo "Node: $SLURM_NODEID / $NUM_NODES"
    echo "Master: $MASTER_ADDR:$MASTER_PORT"
    echo "Dataset: WikiText-103"
    echo "Tokenizer: Llama-3.1-8B"
    echo "Weights: Random initialization"
    echo "Tensor Parallel: $TP_SIZE"
    echo "Data Parallel: $DP_SIZE"
    echo "World Size: $WORLD_SIZE"
    echo "Seq Length: $SEQ_LENGTH"
    echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
    echo "Profiling: Iteration 25"
    echo "Traces: /scratch/profiling_traces/"
    echo "========================================="
    
    if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
        echo "Error: pretrain_gpt.py not found"
        exit 1
    fi
    
    # Run training - torchrun handles the multi-GPU distribution
    torchrun \
        --nproc_per_node $GPUS_PER_NODE \
        --nnodes $NUM_NODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        "$PRETRAIN_SCRIPT_PATH" \
        --use-mcore-models \
        --num-layers $NUM_LAYERS \
        --hidden-size 4096 \
        --ffn-hidden-size 14336 \
        --num-attention-heads 32 \
        --group-query-attention \
        --num-query-groups 8 \
        --kv-channels 128 \
        --seq-length $SEQ_LENGTH \
        --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
        --position-embedding-type rope \
        --rotary-base 1000000 \
        --rotary-percent 1.0 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --swiglu \
        --init-method-std 0.0134 \
        --attention-backend fused \
        --apply-layernorm-1p \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --micro-batch-size $MICRO_BATCH_SIZE \
        --global-batch-size $GLOBAL_BATCH_SIZE \
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
        --tensor-model-parallel-size $TP_SIZE \
        --context-parallel-size $CP_SIZE \
        --data-path "$DATA_PREFIX" \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model "$TOKENIZER_PATH" \
        --data-cache-path "${DATA_CACHE_PATH}" \
        --split "100,0,0" \
        --num-workers 2 \
        --log-interval 1 \
        --log-throughput
    '
