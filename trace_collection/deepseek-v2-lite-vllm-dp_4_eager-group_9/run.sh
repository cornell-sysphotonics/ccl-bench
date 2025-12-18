nsys profile \
  -t cuda \
  -s none \
  --cpuctxsw=none \
  --trace-fork-before-exec=true \
  --force-overwrite=true \
  --stats=true \
  -o moe_dp4tp1 \
  vllm serve /pscratch/sd/c/cp724/DeepSeek-V2-Lite \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --port 8000