# Torch Trace Collection for Mistral 7B Instruct v0.2

## Generate trace from PyTorch model

This directory contains the trace collection setup for the Mistral-7B-Instruct-v0.2 model using Megatron-LM with the following parallelization:

- Data Parallelism (DP): 2
- Tensor Parallelism (TP): 2
- Pipeline Parallelism (PP): 2
- Context Parallelism (CP): 1
- Expert Parallelism (EP): 1

The workload is run on Perlmutter with 8 GPUs (1 nodes × 4 GPUs each).

The workload configuration is located in `run.sh`