# Torch Trace Collection for Llama 3.1 8B

## Generate trace from PyTorch model

This directory contains the trace collection setup for the Llama-3.1-8B model using Megatron-LM with the following parallelization:

- Data Parallelism (DP): 2
- Tensor Parallelism (TP): 4
- Pipeline Parallelism (PP): 1
- Context Parallelism (CP): 1
- Expert Parallelism (EP): 1

The workload is run on Perlmutter with 8 GPUs (1 nodes × 4 GPUs each).

The workload configuration is located in `run.sh`