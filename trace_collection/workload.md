# Standard Workload Set

9 reference configurations selected to maximize coverage of collected traces
while spanning the model size and batch regime space.

Sorted by phase (inference → training), then model size within each phase.
For inference workloads, batch size is treated as variable and not a distinguishing dimension.

| # | Model | Params | Phase | Batch Size | Seq Len | Coverage |
|---|-------|--------|-------|-----------|---------|----------|
| 1 | **Qwen3-4B** | ~4B | Inference | — | 1024 | `qwen3-4b-vllm-perlmutter[nccl/mscclpp]`, `Qwen3-4B-torchxla-vllm-tp1/2/4/8-tpu-group-4` |
| 2 | **Llama-3.1-8B** | ~8B | Inference | — | 1024 | `Llama-3.1-8B-torchxla-vllm-tp1/2/4/8-tpu-group-4` |
| 3 | **Llama-3.1-8B** | ~8B | Inference | — | 2048 | |
| 4 | **Llama-3.1-8B** | ~8B | Training | 4 | 512 | `llama-3.1-8b-torchxla_fsdp_v6e-4/8-tpu-group_21` |
| 5 | **Llama-3.1-8B** | ~8B | Training | 32 | 1024 | `llama_3.1_8b-megatron_lm-dp_2/4-perlmutter-group_1` |
| 6 | **Qwen3-8B** | ~8B | Training | 2 | 1024 | `qwen-3-8b-torchtitan-dp_4-perlmutter-group_5` |
| 7 | **DeepSeek-V3-16B** | 16B MoE | Training | 8 | 1024 | `deepseek-v3-16b-torchtitan-ep*-perlmutter` |
| 8 | **DeepSeek-V3-16B** | 16B MoE | Training | 4 | 5096 | |
| 9 | **DeepSeek-V3-236B** | 236B MoE | Training | 64 | 1024 | `deepseek-v3-236b-torchtitan-ep32-dp8-pp8-tp4-perlmutter` |
