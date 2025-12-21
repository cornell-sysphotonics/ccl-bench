# Trace collection
Trace collection: Eric

On github this folder should only store the metadata, i.e. workload card.

You are going to upload your traces to Canvas. 

Sample traces are stored in the [Google drive](https://drive.google.com/drive/u/0/folders/1k6xokFE2MAnt39YR8f-DTKK86IXitV0x?usp=drive_link)

## Current list of traces
1. `llama3.1-8b-torchtitan-perlmutter`
2. `deepseekv2lite-vllm-lambda1`
3. `qwen3-32b-tp_2-batch_1-sglang-group_11`
4. `qwen3-32b-tp_2-batch_4-sglang-group_11`
5. `qwen3-32b-tp_4-batch_1-sglang-group_11`
6. `qwen3-32b-tp_4-batch_4-sglang-group_11`
7. `qwen3-32b-tp_4-batch_4-burst-sglang-group_11`
8. `qwen3-32b-tp_2-batch_1-vllm-group_11`
9. `qwen3-32b-tp_2-batch_4-vllm-group_11`
10. `qwen3-32b-tp_4-batch_1-vllm-group_11`
11. `qwen3-32b-tp_4-batch_4-vllm-group_11`
12. `qwen3-32b-tp_4-batch_4-burst-vllm-group_11`
13. `qwen3-30b-a3b-tp_2-batch_1-vllm-group_11`
14. `qwen3-30b-a3b-tp_2-batch_4-vllm-group_11`
15. `qwen3-30b-a3b-tp_4-batch_1-vllm-group_11`
16. `qwen3-30b-a3b-tp_4-batch_4-vllm-group_11`

## Workload Descriptions

Our experiments are categorized into three primary workload families designed to stress different communication regimes within the distributed serving environment:

### 1. Baseline & Scaling (Static Workload)
*   **Model:** Qwen3-32B (Dense)
*   **Dataset:** Dummy prompts (Repeated tokens)
*   **Purpose:** Establishes a performance baseline for dense models sharded via Tensor Parallelism. By varying TP size (2, 4) and Batch size (1, 4) under a fixed "infinite" request rate, we measure the scalability of NVLink throughput as the model sharding and concurrency increase.

### 2. Temporal Dynamics (Bursty Workload)
*   **Model:** Qwen3-32B (Dense)
*   **Dataset:** BurstGPT (Temporal traces)
*   **Purpose:** To reproduce realistic, non-deterministic arrival patterns. We pair the arrival timestamps and token counts from the BurstGPT dataset with dummy prompts of matching lengths. This stresses the serving framework's (vLLM vs. SGLang) ability to schedule communication and batching effectively during idle gaps and sudden traffic spikes.

### 3. Architectural Impact (MoE Workload)
*   **Model:** Qwen3-30B-A3B (Mixture-of-Experts)
*   **Dataset:** Dummy prompts
*   **Purpose:** Investigates the communication "duty cycle" of conditional computation. Unlike dense models that use All-Reduce after every layer, MoE models utilize Expert Parallelism and All-to-All primitives. This workload quantifies the hardware-level communication savings inherent to MoE architectures compared to their dense counterparts.

### 4. Framework Characterization (vLLM vs. SGLang)
*   **Purpose:** A direct comparison between vLLM’s PagedAttention and SGLang’s Radix-tree prefix sharing. While both optimize KV-cache management, this workload analyzes if their scheduling differences result in different instantaneous NVLink bursts, particularly during the prefill phase.