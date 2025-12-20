# Trace collection
Trace collection: Eric

On github this folder should only store the metadata, i.e. workload card.

You are going to upload your traces to Canvas. 

Sample traces are stored in the [Google drive](https://drive.google.com/drive/u/0/folders/1shHsa3WvlGh9YRaX7iqYBYTLnwdDfLX6)

## Current list of traces
1. `llama3.1-8b-torchtitan-perlmutter`
2. `deepseekv2lite-vllm-lambda1`
3. `qwen3-32b-tp_2-batch_1-sglang-group_11`
4. `qwen3-32b-tp_2-batch_4-sglang-group_11`
5. `qwen3-32b-tp_4-batch_1-sglang-group_11`
6. `qwen3-32b-tp_4-batch_4-sglang-group_11`
7. `qwen3-32b-tp_4-batch_4-burst-sglang-group_11`
*(vLLM counterparts for Qwen3-32B collected externally)*

## Key Metrics Framework
The experiments focus on characterizing the temporal dynamics of NVLink utilization through high-resolution polling (1ms).

### 1. System Performance (User Experience)
*   **TTFT (Time to First Token):** Latency of the prefill phase (Mean/P99).
*   **TPOT (Time per Output Token):** Latency of the decode phase (Mean/P99).
*   **Throughput (tok/s):** Aggregate generation rate.

### 2. Hardware Intensity & Burst Dynamics
*   **Peak NVLink Bandwidth (GB/s):** Maximum observed throughput saturation point.
*   **Burst Intensity ($Peak / Average$):** Quantifies the communication "spikiness."
*   **Burst Duration (s):** Cumulative time spent in high-utilization states (>20 GB/s).
*   **Aggregate Traffic (GB):** Total volume of data transferred over the workload.

### 3. Communication Efficiency
*   **NCCL Contribution (%):** Fraction of GPU active time spent on communication kernels.
*   **Bandwidth Usage Efficiency (%):** Ratio of average throughput to the A100 theoretical maximum (600 GB/s aggregate).

### 4. Phase-Specific Characterization
*   **Prefill vs. Decode Intensity:** Average aggregate bandwidth reported separately for activation-heavy prefill and frequency-heavy decode.

## Analysis Tools
Metrics are generated using the consolidated analysis pipeline:
*   `tools/nvlink/generate_report_metrics.py`: Automatically correlates NVML binaries, `nsys` SQLite exports, and benchmark JSONs to produce the `experiment_metrics.csv` and throughput dynamics plots.
