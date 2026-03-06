# GPU Trace Analysis Knowledge Base

## Run 1 — 2025-02-13 — LLaMA 3.1 8B (torchtitan, Perlmutter)

### Trace Format (Kineto / PyTorch Profiler)
- Top-level keys: `schemaVersion`, `deviceProperties`, `cupti_version`, `cuda_runtime_version`, `cuda_driver_version`, `distributedInfo`, `trace_id`, `displayTimeUnit`, `baseTimeNanoseconds`, `traceEvents`, `traceName`
- `traceEvents` is the main array of events (can be 100K+ events for a 65MB file)
- Event categories (`cat` field): `kernel`, `cpu_op`, `user_annotation`, `gpu_user_annotation`, `ac2g`, `cuda_runtime`
- Kernel events have `cat="kernel"`, NCCL kernels have "nccl" in the name (case-insensitive)
- `ts` is in microseconds, `dur` is in microseconds

### NCCL Kernel Args (Rich Metadata!)
NCCL kernel events on newer PyTorch versions (seen with torchtitan) have very rich args:
- `Collective name`: e.g., "reduce_scatter_tensor_coalesced", "allgather_into_tensor_coalesced"
- `In msg nelems`, `Out msg nelems`: number of elements in/out
- `Group size`: number of ranks in the collective
- `dtype`: e.g., "BFloat16"
- `In split size`, `Out split size`: for variable-size collectives
- `Process Group Name`: the pg number (e.g., "17")
- `Process Group Description`: human-readable (e.g., "mesh_tp")
- `Process Group Ranks`: list of ranks in the group

### Key Observations — LLaMA 3.1 8B on Perlmutter
- GPU: NVIDIA A100-SXM4-40GB, 108 SMs
- World size: 16, Rank 0
- Parallelism: PP=2 (ranks 0,8), TP=4 (ranks 0-3), DP_shard=2 (ranks 0,4)
- 724 NCCL kernel calls per iteration
- Communication dominates: 92.2% of total kernel time
  - SendRecv (PP): 32.4% — 5 calls but very long duration (blocking pipeline sends)
  - AllGather (TP+FSDP): 31.5% — 405 calls
  - ReduceScatter (TP+FSDP): 26.3% — 277 calls
  - AllReduce: 2.1% — 37 calls (mostly small gradient syncs)
- Compute is only ~7.8% of kernel time (GEMM 4.2%, attention 1.1%, elementwise 1.8%)
- Very low comm-compute overlap: 0.6% — almost all communication is serialized
- GPU utilization ~99.6% (GPU is always busy, but mostly with NCCL)
- Iteration wall clock: ~16.8s (ProfilerStep#3)
- SendRecv kernels don't have `In/Out msg nelems` — their bytes show as 0
- The "unknown" process group for SendRecv is because PP send/recv doesn't always annotate the PG

### Metric Computation Notes
- `total_kernel_time`: Use sum of all kernel durations. Also compute merged (non-overlapping) version.
- `communication_ratio`: NCCL dur sum / total kernel dur sum * 100
- `comm_comp_overlap`: Use merged intervals for NCCL and non-NCCL kernels, compute intersection
- `aggregate_gpu_utilization`: merged kernel intervals / iteration wall clock
- `mean_sm_coverage`: Use grid dimensions vs num_sms; `est. achieved occupancy %` is often 0 in traces
- `estimated_bandwidth`: total NCCL bytes / total NCCL time. For individual collectives, use appropriate bytes (out for AllGather, in for ReduceScatter, in*2 for AllReduce)
- `bandwidth_utilization`: Compare observed BW vs theoretical peak. A100 SXM4 NVLink: 600 GB/s bidirectional
- `traffic_window`: Sort NCCL events by PG and timestamp, compute inter-call gaps
- `break_down_steps`: Classify kernels by name patterns into categories

### Metrics That Cannot Be Computed from Single-Rank Kineto Trace
- `mfu`: Needs model config (hidden_size, num_layers, seq_len, vocab_size) and token count
- `throughput_tokens_sec`: Needs token count
- `load_imbalance_ratio`: Needs multi-rank data
- `straggler`: Needs multi-rank data
- `TTFT`/`TPOT`: Training trace, not inference
- `average_memory_bandwidth`: Needs NSight Compute memory counters
- `token_to_expert_assignment`: Only for MoE models
- Group 6 metrics: Require sglang benchmark JSONL or nsys sqlite files

### Gotchas
- Some events have `dur=None` (e.g., `ac2g` events, instant events)
- `est. achieved occupancy %` is often 0 in Kineto traces (not reliably populated)
- SendRecv NCCL kernels may not have `Process Group Name/Description` in args
- The trace can have multiple ProfilerStep events (user_annotation and gpu_user_annotation); the gpu_user_annotation version represents the GPU-side view
- Kernel events across different streams can overlap in time — must use interval merging for true utilization
- The file is 65MB; loads fine with json.load() in ~2-3 seconds on modern hardware
