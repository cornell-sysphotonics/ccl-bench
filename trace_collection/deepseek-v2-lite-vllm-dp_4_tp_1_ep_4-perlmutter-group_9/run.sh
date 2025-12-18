#export VLLM_ALL2ALL_BACKEND=pplx # or naive
nsys profile \
  -t cuda \
  -s none \
  --cpuctxsw=none \
  --trace-fork-before-exec=true \
  --force-overwrite=true \
  --stats=true \
  -o moe_dp2tp2 \
  vllm serve /pscratch/sd/c/cp724/DeepSeek-V2-Lite \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --port 8000

  #  default
  # vllm serve /pscratch/sd/c/cp724/DeepSeek-V2-Lite \
  # --tensor-parallel-size 1 \
  # --data-parallel-size 4\
  # --enable-expert-parallel \
  # --port 8000

  # disable CUDA graph
  # vllm serve /pscratch/sd/c/cp724/DeepSeek-V2-Lite \
  #   --enforce-eager \
  #   --tensor-parallel-size 1 \
  #   --data-parallel-size 4 \
  #   --enable-expert-parallel \
  #   --port 8000

  #diable chunked prefill
  #   vllm serve /pscratch/sd/c/cp724/DeepSeek-V2-Lite \
  # --tensor-parallel-size 1 \
  # --data-parallel-size 4 \
  # --enable-expert-parallel \
  # --max-num-batched-tokens 4096 \
  # --port 8000