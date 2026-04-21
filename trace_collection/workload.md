# Standard Workload Set

10 reference configurations selected to maximize coverage of collected traces
while spanning the model size and batch regime space.

Sorted by phase (inference → training), then model size within each phase.

| # | Model | Params | Phase | Batch Size | Seq Len | Batch Regime | Coverage |
|---|-------|--------|-------|-----------|---------|--------------|----------|
| 1 | **Qwen3-4B** | ~4B | Inference | 1 | 1024 | Single-request | `qwen3-4b-vllm-perlmutter[nccl/mscclpp]` (tp2, tp4) |
| 2 | **Qwen3-4B** | ~4B | Inference | 128 | 1024 | Large batch | `Qwen3-4B-torchxla-vllm-tp1/2/4/8-tpu-group-4` |
| 3 | **Llama-3.1-8B** | ~8B | Inference | 128 | 1024 | Large batch | `Llama-3.1-8B-torchxla-vllm-tp1/2/4/8-tpu-group-4` |
| 4 | **Llama-3.1-8B** | ~8B | Inference | 32 | 2048 | Medium batch, long ctx | |
| 5 | **Llama-3.1-8B** | ~8B | Training | 4 | 512 | Small batch | `llama-3.1-8b-torchxla_fsdp_v6e-4/8-tpu-group_21` |
| 6 | **Llama-3.1-8B** | ~8B | Training | 32 | 1024 | Medium batch | `llama_3.1_8b-megatron_lm-dp_2/4-perlmutter-group_1` |
| 7 | **Qwen3-8B** | ~8B | Training | 2 | 1024 | Small batch | `qwen-3-8b-torchtitan-dp_4-perlmutter-group_5` |
| 8 | **DeepSeek-V3-16B** | 16B MoE | Training | 8 | 1024 | Small batch | `deepseek-v3-16b-torchtitan-ep*-perlmutter` |
| 9 | **DeepSeek-V3-16B** | 16B MoE | Training | 4 | 5096 | Small batch, long ctx | |
| 10 | **DeepSeek-V3-236B** | 236B MoE | Training | 64 | 1024 | Large batch | `deepseek-v3-236b-torchtitan-ep32-dp8-pp8-tp4-perlmutter` |
