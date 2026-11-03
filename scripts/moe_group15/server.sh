CUDA_VISIBLE_DEVICES=0,1 \
VLLM_LOGGING_CONFIG_PATH=./logging_config.json \
vllm serve Qwen/Qwen3-30B-A3B \
    --tensor-parallel-size 2 \
    --max-model-len 16368 \
    --max-num-seqs 512 \
    --enforce-eager