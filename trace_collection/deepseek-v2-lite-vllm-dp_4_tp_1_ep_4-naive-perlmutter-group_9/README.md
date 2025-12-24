Online serving of DeepSeek-V2-Lite on 4 A100 GPUs with EP=4.
allgather_reducescatter(default), pplx and naive kernel is used as AlltoAll backend.

To change the all2all kernel, we need to set environment variable:
export VLLM_ALL2ALL_BACKEND=pplx  # or naive
we use this workload config for comparing different All-to-All backend kernels (allgether/reducescatter, pplx, naive).