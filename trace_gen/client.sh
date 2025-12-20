python3 benchmark.py --backend vllm \
    --model Qwen/Qwen3-32B \
    --request-rate 1 \
    --num-prompts 10 \
    --dataset-name dummy \
    --long-prompts 0 \
    --long-prompt-len 32000