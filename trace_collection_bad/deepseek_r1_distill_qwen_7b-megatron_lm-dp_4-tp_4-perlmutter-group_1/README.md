# Torch Trace Collection for DeepSeek-R1 Distilled Qwen-7B

## Generate trace from PyTorch model

This directory contains the trace collection setup for the DeepSeek-R1 distilled Qwen-7B model using Megatron-LM with the following parallelization:

- Data Parallelism (DP): 4
- Tensor Parallelism (TP): 4
- Pipeline Parallelism (PP): 1
- Context Parallelism (CP): 1
- Expert Parallelism (EP): 1

The workload is run on Perlmutter with 16 GPUs (4 nodes × 4 GPUs each).

The workload configuration is located in `run.sh`