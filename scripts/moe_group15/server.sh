export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_LOGGING_CONFIG_PATH=./logging_config.json \
vllm serve Qwen/Qwen3-30B-A3B \
    --tensor-parallel-size 4 \
    --max-model-len 16368 \
    --max-num-seqs 512 \
    --enforce-eager
