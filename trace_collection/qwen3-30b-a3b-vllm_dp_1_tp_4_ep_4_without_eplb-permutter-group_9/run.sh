
nsys profile \
  -t cuda \
  -s none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --stats=true \
  -o default_1k \
  vllm serve /pscratch/sd/j/jy2222/Qwen3-30B-A3B \
    --data-parallel-size 4 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --served-model-name Qwen \
    --gpu-memory-utilization 0.85