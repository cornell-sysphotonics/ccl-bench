# Trace collection
Trace collection: Eric

On github this folder should only store the metadata, i.e. workload card.

You are going to upload your traces to Canvas.

Sample traces are stored in the [Google drive](https://drive.google.com/drive/u/0/folders/1shHsa3WvlGh9YRaX7iqYBYTLnwdDfLX6)

## Current list of traces
- llama3.1-8b-torchtitan-perlmutter (legacy sample)
- deepseekv2lite-vllm-lambda1 (legacy sample)
- deepseek-v2-lite-torchtitan-4d-perlmutter-group-16 (TorchTitan 4D parallel baseline)
- deepseek-v2-lite-torchtitan-fsdp-perlmutter-group-16 (FSDP-only)
- deepseek-v2-lite-torchtitan-fsdp+tp-perlmutter-group-16 (FSDP + tensor parallel)
- deepseek-v2-lite-torchtitan-fsdp+ep-perlmutter-group-16 (FSDP + expert parallel)
- deepseek-v2-lite-torchtitan-fsdp+pp+ep-perlmutter-group-16 (FSDP + pipeline + expert parallel)
- deepseek-v2-lite-torchtitan-fsdp+tp+ep-perlmutter-group-16 (FSDP + tensor + expert parallel)
- deepseek-v2-lite-torchtitan-pp-perlmutter-group-16 (pipeline parallel only)
- deepseek-v2-lite-torchtitan-tp-perlmutter-group-16 (tensor parallel only)
- deepseek-v2-lite-torchtitan-tp+ep-perlmutter-group-16 (tensor + expert parallel)
- deepseek-v2-lite-torchtitan-tp+pp+ep-perlmutter-group-16 (tensor + pipeline + expert parallel)
- llama3-8b-torchtitan-fsdp-perlmutter-group-16 (Llama3 8B, FSDP)
- llama3-8b-torchtitan-fsdp+tp-perlmutter-group-16 (Llama3 8B, FSDP + tensor parallel)
- llama3-8b-torchtitan-fsdp+tp+pp-perlmutter-group-16 (Llama3 8B, FSDP + tensor + pipeline)
- llama3-8b-torchtitan-pp-perlmutter-group-16 (Llama3 8B, pipeline only)
- llama3-8b-torchtitan-tp-perlmutter-group-16 (Llama3 8B, tensor only)
- llama3-8b-torchtitan-tp+pp-perlmutter-group-16 (Llama3 8B, tensor + pipeline)
- qwen3-32b-torchtitan-fsdp-perlmutter-group-16 (Qwen3 32B, FSDP)
- qwen3-32b-torchtitan-fsdp+tp-perlmutter-group-16 (Qwen3 32B, FSDP + tensor parallel)
- qwen3-32b-torchtitan-fsdp+cp-perlmutter-group-16 (Qwen3 32B, FSDP + context parallel)
- qwen3-32b-torchtitan-tp-perlmutter-group-16 (Qwen3 32B, tensor only)
