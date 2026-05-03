# Trace Collection

This directory contains performance benchmark traces and results for Mixtral-8x7B-v0.1 model inference experiments conducted on NERSC Perlmutter.

You are going to upload your traces to Canvas.

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

## Current list of traces
1. `llama3.1-8b-torchtitan-perlmutter`
2. `deepseekv2lite-vllm-lambda1`
3. `deepseek-v2-lite-sglang-ep_1-perlmutter-group_6`: Online serving of Deepseek-V2-Lite on 4 A100 GPUs with TP=4, EP=1.
4. `deepseek-v2-lite-sglang-ep_2-perlmutter-group_6`: Online serving of Deepseek-V2-Lite on 4 A100 GPUs with TP=4, EP=2.
5. `deepseek-v2-lite-sglang-ep_4-perlmutter-group_6`: Online serving of Deepseek-V2-Lite on 4 A100 GPUs with TP=4, EP=4.
6. `llama-3.1-8b-sglang-tp_1-pp_4-perlmutter-group_6`: Online serving of Llama-3.1-8B on 4 A100 GPUs with TP=1, PP=4.
7. `llama-3.1-8b-sglang-tp_2-pp_2-perlmutter-group_6`: Online serving of Llama-3.1-8B on 4 A100 GPUs with TP=2, PP=2.
8. `llama-3.1-8b-sglang-tp_4-pp_1-perlmutter-group_6`: Online serving of Llama-3.1-8B on 4 A100 GPUs with TP=4, PP=1.
9. `qwen-32b-sglang-pp_1-perlmutter-group_6`: Online serving of Qwen3-32b on 4 A100 GPUs with TP=4, PP=1.
10. `qwen-32b-sglang-pp_2-perlmutter-group_6`: Online serving of Qwen3-32b on 8 A100 GPUs on 2 nodes with TP=4, PP=2.
3. `[Qwen3-4B-torchxla-vllm-tp8-tpu-group-4](./Qwen3-4B-torchxla-vllm-tp8-tpu-group-4)`
4. `[Qwen3-4B-torchxla-vllm-tp4-tpu-group-4](./Qwen3-4B-torchxla-vllm-tp4-tpu-group-4)`
5. `[Qwen3-4B-torchxla-vllm-tp2-tpu-group-4](./Qwen3-4B-torchxla-vllm-tp2-tpu-group-4)`
6. `[Qwen3-4B-torchxla-vllm-tp1-tpu-group-4](./Qwen3-4B-torchxla-vllm-tp1-tpu-group-4)`
7. `[Qwen3-32B-torchxla-vllm-tp8-tpu-group-4](./Qwen3-32B-torchxla-vllm-tp8-tpu-group-4)`
8. `[Qwen3-32B-torchxla-vllm-tp4-tpu-group-4](./Qwen3-32B-torchxla-vllm-tp4-tpu-group-4)`
9. `[Qwen3-32B-torchxla-vllm-tp2-tpu-group-4](./Qwen3-32B-torchxla-vllm-tp2-tpu-group-4)`
10. `[Qwen3-32B-torchxla-vllm-tp1-tpu-group-4](./Qwen3-32B-torchxla-vllm-tp1-tpu-group-4)`
11. `[Llama-3.3-70B-Instruct-torchxla-vllm-tp8-tpu-group-4](./Llama-3.3-70B-Instruct-torchxla-vllm-tp8-tpu-group-4)`
12. `[Llama-3.3-70B-Instruct-torchxla-vllm-tp4-tpu-group-4](./Llama-3.3-70B-Instruct-torchxla-vllm-tp4-tpu-group-4)`
13. `[Llama-3.3-70B-Instruct-torchxla-vllm-tp2-tpu-group-4](./Llama-3.3-70B-Instruct-torchxla-vllm-tp2-tpu-group-4)`
14. `[Llama-3.3-70B-Instruct-torchxla-vllm-tp1-tpu-group-4](./Llama-3.3-70B-Instruct-torchxla-vllm-tp1-tpu-group-4)`
15. `[Llama-3.1-8B-torchxla-vllm-tp8-tpu-group-4](./Llama-3.1-8B-torchxla-vllm-tp8-tpu-group-4)`
16. `[Llama-3.1-8B-torchxla-vllm-tp4-tpu-group-4](./Llama-3.1-8B-torchxla-vllm-tp4-tpu-group-4)`
17. `[Llama-3.1-8B-torchxla-vllm-tp2-tpu-group-4](./Llama-3.1-8B-torchxla-vllm-tp2-tpu-group-4)`
18. `[Llama-3.1-8B-torchxla-vllm-tp1-tpu-group-4](./Llama-3.1-8B-torchxla-vllm-tp1-tpu-group-4)`
3. `deepseek_r1_distill_qwen_7b-megatron_lm-dp_2-tp_4-pp_2-perlmutter-group_1`
4. `deepseek_r1_distill_qwen_7b-megatron_lm-dp_4-tp_4-perlmutter-group_1`
5. `llama_31_8b-megatron_lm-dp_2-tp_4-perlmutter-group_1`
6. `llama_31_8b-megatron_lm-dp_4-tp_2-perlmutter-group_1`
7. `mistral_7b_instruct_v02-megatron_lm-dp_2-tp_2-pp_2-perlmutter-group_1`
8. `mistral_7b_instruct_v02-megatron_lm-dp_2-tp_4-perlmutter-group_1`
19. `[llama-3.1-8b-torchxla-train-tp4-dp1-fsdp2-tpu](./llama-3.1-8b-torchxla-train-tp4-dp1-fsdp2-tpu)`: Training of Llama-3.1-8B on 8 TPU chips with TP=4, DP=1, FSDP (dp_shard)=2.
20. `[deepseek-v2-16b-maxtext-train-tp2-ep4-dp1-tpu](./deepseek-v2-16b-maxtext-train-tp2-ep4-dp1-tpu)`: Training of DeepSeek-V2 16B MoE on 8 TPU v6e chips with TP=2, EP=4, DP=1 using MaxText.
