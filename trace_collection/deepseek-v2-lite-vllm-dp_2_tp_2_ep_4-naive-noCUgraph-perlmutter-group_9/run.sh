export VLLM_ALL2ALL_BACKEND=naive

nsys profile \
  -t cuda \
  -s none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --stats=true \
  -o moe_dp2tp2 \
  vllm serve /pscratch/sd/c/cp724/DeepSeek-V2-Lite \
    --enforce-eager \
    --data-parallel-size 2 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --port 8000