# Qwen3-32B TP4 Batch 4 Bursty (SGLang)

## Implementation Details
This workload profiles the NVLink interconnect utilization for the Qwen3-32B model using the SGLang serving engine under a bursty traffic pattern. 

- **Framework:** SGLang
- **Model:** Qwen3-32B
- **Parallelism:** TP=4
- **Batch Size:** 4
- **Dataset:** BurstGPT

## Execution Method
To replicate this experiment, ensure you are in a compute node (e.g., Slurm allocation on Perlmutter) and run:

```bash
./run.sh
```

The script performs the following actions:
1.  **Server Launch:** Starts the SGLang server with TP=4 and Qwen3-32B.
2.  **Profiling Orchestration:** Launches `nsys profile` and our custom NVLink throughput poller in the background.
3.  **Client Workload:** Executes `benchmark.py` with the BurstGPT dataset.
4.  **Post-Processing:** Exports the `nsys` report to SQLite for metric correlation.

## Artifacts Generated
- `experiment_metadata.json`: Configuration and completion status.
- `nvlink_trace.bin`: Raw high-resolution NVLink utilization data.
- `sglang_profile.nsys-rep`: Nsight Systems profile.
- `sglang_profile.sqlite`: Exported SQLite database for kernel correlation.

