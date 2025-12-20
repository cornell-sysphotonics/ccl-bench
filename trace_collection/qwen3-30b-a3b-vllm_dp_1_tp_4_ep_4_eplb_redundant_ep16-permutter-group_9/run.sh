
nsys profile \
  -t cuda \
  -s none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --trace-fork-before-exec=true \
  --stats=true \
  -o default_eplb8 \
  vllm serve /pscratch/sd/j/jy2222/Qwen3-30B-A3B \
    --data-parallel-size 4 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --served-model-name Qwen \
    --gpu-memory-utilization 0.85 \
    --enable-eplb \ 
    --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":8,"log_balancedness":false}'