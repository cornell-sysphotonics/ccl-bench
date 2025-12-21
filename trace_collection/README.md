# Trace Collection

This directory contains performance benchmark traces and results for Mixtral-8x7B-v0.1 model inference experiments conducted on NERSC Perlmutter.

## Overview

Each subdirectory represents a single experiment configuration that benchmarks the Mixtral-8x7B-v0.1 model under different settings. The experiments compare two MoE (Mixture-of-Experts) kernel implementations:

- **Baseline**: Uses Triton MoE kernel with Tensor Parallelism (TP=4)
- **DefaultAll2All**: Uses FlashInfer MoE backend with Expert Parallelism (EP=4) and All2All communication

## Directory Naming Convention

Each experiment directory follows the format:
```
<model>-<framework+parallelism>-<platform[config]>-<group>
```

For example:
- `Mixtral8x7B-vllmTP4-Perlmutter[baseline_64_8192]-group12`
- `Mixtral8x7B-vllmEP4-Perlmutter[defaultall2all_32_8192_eplbon]-group12`

### Components:

- **Model**: `Mixtral8x7B` (Mixtral-8x7B-v0.1 model)
- **Framework+Parallelism**: 
  - `vllmTP4`: vLLM with Tensor Parallel 4 (Baseline)
  - `vllmEP4`: vLLM with Expert Parallel 4 (DefaultAll2All)
- **Platform**: `Perlmutter[<config>]` where config includes:
  - Model type: `baseline` or `defaultall2all`
  - `MAX_SEQS`: Maximum number of sequences in a batch
  - `BATCH_TOKENS`: Maximum number of batched tokens
  - Optional EPLB settings: `_eplbon` or `_eplboff` (Expert Parallel Load Balancing)


## Controlled Variables

The experiments systematically vary the following parameters:

1. **Model Type**: Baseline (Triton MoE) vs. DefaultAll2All (FlashInfer MoE)
2. **MAX_SEQS**: Maximum number of sequences (8, 16, 32, 64)
3. **BATCH_TOKENS**: Maximum number of batched tokens (2048, 4096, 8192, 16384)
4. **EPLB**: Expert Parallel Load Balancing (enabled/disabled for DefaultAll2All experiments)

All experiments use:
- **Hardware**: 4x NVIDIA A100 GPUs on NERSC Perlmutter
- **Model**: Mixtral-8x7B-v0.1
- **MAX_MODEL_LEN**: 4096
- **Framework**: vLLM

## Experiment Structure

Each experiment directory contains:

- `server.sh`: Script to launch the vLLM inference server
- `client.sh`: Script to run benchmark requests and collect metrics
- `README.md`: Detailed configuration and instructions for that specific experiment
- `*.yaml`: Workload card documenting the experiment configuration
- `results_json/`: Performance metrics (TTFT, TPOT, throughput, etc.)
- `logs/`: Server logs and profiling traces (nsys)

## Results

Performance metrics are collected for each experiment, including:

- **TTFT** (Time to First Token): Latency until the first token is generated
- **TPOT** (Time Per Output Token): Average latency per generated token
- **Request Throughput**: Requests processed per second
- **Output Throughput**: Tokens generated per second

Results are stored in JSON format in the `results_json/` directory of each experiment.

## Running Experiments

For detailed instructions on running a specific experiment, see the `README.md` file in that experiment's directory. In general:

1. Navigate to the experiment directory
2. load environment
3. Start the server: `./server.sh`
4. In another terminal, run the client: `./client.sh`

Both server and client should be run on the same compute node.

## Related Tools (under tools/)

Performance metric extraction and plotting scripts are available in the `tools/` directory:
- `TTFT-group_12/`: Extract and plot Time to First Token metrics
- `TPOT-group_12/`: Extract and plot Time Per Output Token metrics
- `request_throughput-group_12/`: Extract request throughput metrics
- `requestThroughput-group_12/`: Extract output throughput metrics
- `Throughput-group_12/`: Plot throughput metrics

