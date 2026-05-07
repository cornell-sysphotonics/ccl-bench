# DeepSeek-V2-Lite TPU vLLM Trace

This row records `deepseek-ai/DeepSeek-V2-Lite` inference on one TPU v6e-4 VM with vLLM's TPU backend.

Collection settings:

- model: `deepseek-ai/DeepSeek-V2-Lite`
- tensor parallelism: `TP=4`
- batch size: `128`
- input length: `1024`
- output length: `128`
- max model length: `1152`
- dtype: `bfloat16`
- GPU/KV memory utilization setting: `0.90`
- Docker image: `vllm/vllm-tpu:v0.19.0`
- warmup iterations: `5`
- profiled iterations: `10`

The raw profiler artifacts live in:

```text
/data/ccl-bench_trace_collection/deepseek-v2-lite-vllm-tp4-batch128
```

The downloaded TPU profiler trace came from:

```text
tpu_vllm_inference/traces/MODEL_deepseek-ai_DeepSeek-V2-Lite,INPUT_1024,OUTPUT_128,BATCH_128,TP_4/plugins/profile/2026_05_07_05_45_51/
```

The run used the DeepSeek-V2 MLA dtype hotfix in `tpu_vllm_inference/run.sh`, casting `k_pe` to `kv_cache.dtype` before calling the TPU MLA attention kernel.
