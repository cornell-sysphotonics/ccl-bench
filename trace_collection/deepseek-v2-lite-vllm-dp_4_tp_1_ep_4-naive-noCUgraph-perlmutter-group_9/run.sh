export VLLM_ALL2ALL_BACKEND=naive

nsys profile \
  -t cuda \
  -s none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --stats=true \
  -o moe_dp4tp1 \
  vllm serve /pscratch/sd/c/cp724/DeepSeek-V2-Lite \
    --data-parallel-size 4 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --port 8000