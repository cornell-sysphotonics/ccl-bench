# CCL-Bench Simulation Pipeline

What-if hardware simulation for CCL-Bench traces using [AstraSim](https://astra-sim.github.io/) and [Chakra](https://github.com/mlcommons/chakra) execution traces.

## Overview

The pipeline replays real distributed training/inference communication traces under hypothetical hardware configurations — different network bandwidth, topology, latency, or collective algorithm — without re-running the actual workload.

```
rankN_trace.json  ──►  gen_chakra_et.py  ──►  chakra_trace.*.et
                                          ──►  comm_group.json
                                                    │
                        network.yml  ──────────────►│
                        system.json  ──────────────►│
                                                    ▼
                                            AstraSim (Docker)
                                                    │
                                                    ▼
                                          simulation.log (per-rank stats)
```

**Two modes:**

| Mode | Input | Fidelity |
|------|-------|----------|
| `comm-only` | `rankN_trace.json` (kineto Chrome JSON) | Compute time replayed as measured; comm time simulated |
| `full` | `pytorch_et_N.json` + `kineto_trace_N.json` | Full op-graph fidelity via `chakra_trace_link` |

## Prerequisites

Docker must be running with the `astra-sim:latest` image available. To build it:

```bash
cd agent/experiments/tools/astra-sim-hybrid-parallelism
docker build -t astra-sim:latest .
```

The image is ~14 GB and takes 20–40 minutes to build. It includes AstraSim, Chakra, and all dependencies.

No host Python packages are required beyond the standard library — all heavy lifting happens inside Docker.

## Quick Start

```bash
# Baseline: deepseek-v3-16b on A100 Slingshot (200 GB/s Switch)
python simulation/pipeline.py --mode comm-only \
    --trace-dir /data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep4-dp2-pp2-tp4-perlmutter

# What-if: 2× network bandwidth
python simulation/pipeline.py --mode comm-only \
    --trace-dir /data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep4-dp2-pp2-tp4-perlmutter \
    --bandwidth 400

# What-if: different collective algorithm
python simulation/pipeline.py --mode comm-only \
    --trace-dir /data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep4-dp2-pp2-tp4-perlmutter \
    --collective-algo halving_doubling

# What-if: H100 hardware (900 TFLOPS BF16, 3.35 TB/s HBM, 900 GB/s NVLink)
python simulation/pipeline.py --mode full \
    --trace-dir trace_collection_backlog/llama-3.1-8b-torchtitan-perlmutter \
    --peak-perf 900 --mem-bw 3350 --bandwidth 900
```

Run from the repo root (`/home/dd687/ccl-bench/`).

## Arguments

```
--trace-dir PATH        Trace directory (required)
--mode {full,comm-only} Simulation mode (default: full)
--output-dir PATH       Output directory (default: /var/tmp/ccl_bench_sim_*)
```

**Network parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--topology` | `Switch` | `Ring`, `Switch`, or `FullyConnected` |
| `--bandwidth` | `200.0` | Per-link bandwidth in GB/s |
| `--latency` | `500.0` | Per-link latency in ns |

**System parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--collective-algo` | `ring` | `ring`, `halving_doubling`, `direct_point2point`, `doubleBinaryTree` |
| `--peak-perf` | `312.0` | Peak compute TFLOPS (A100 BF16) |
| `--mem-bw` | `2000.0` | HBM bandwidth GB/s (A100) |
| `--scheduling-policy` | `LIFO` | `LIFO` or `FIFO` |

In `comm-only` mode, `--peak-perf` and `--mem-bw` have no effect (roofline is disabled; compute time comes directly from measured trace durations).

## Output

The output directory contains:

```
simulation.log        Full AstraSim output with per-rank statistics
comm_group.json       Process group → NPU mapping (if process group info found in traces)
chakra_trace.*.et     Chakra Execution Trace files (one per rank)
network.yml           Network configuration used
system.json           System configuration used
remote_memory.json    Remote memory configuration (NO_MEMORY_EXPANSION)
run_pipeline.sh       The exact Docker script that was executed
```

The pipeline prints a summary table at the end:

```
[pipeline] AstraSim simulation results (ns):
   sys         Wall time         Comm time          GPU time   Comm %
     0     8,556,424,400       840,785,397     7,715,639,003     9.8%
     ...
   avg     8,555,932,650     2,300,241,772                      26.9%

  Simulated step time: 8555.9 ms  |  Comm fraction: 26.9%
```

## How It Works

### `comm-only` mode

1. **Trace parsing** (`gen_chakra_et.py`, runs inside Docker):
   - Reads each `rankN_trace.json` and extracts NCCL collective kernel events (`ncclDevKernel_AllGather_*`, `ncclDevKernel_ReduceScatter_*`, etc.)
   - P2P send/recv (`ncclDevKernel_SendRecv`) are excluded — they represent pipeline-parallel point-to-point transfers that don't fit the flat-topology collective model
   - Each collective op is preceded by a `COMP_NODE` whose `duration_micros` captures the measured compute gap since the previous collective

2. **Process group extraction** (automatic for torchtitan traces):
   - NCCL kernel events in torchtitan traces carry `"Process Group Ranks"` in their args (e.g., `[0, 1, 2, 3]` for a TP group)
   - Each unique rank-set gets a stable integer ID; the `pg_name` attribute on each `COMM_COLL_NODE` is set to this ID
   - `comm_group.json` maps those IDs to NPU lists, enabling AstraSim's hybrid-parallelism-aware routing (TP/DP/EP groups use separate ring topologies)
   - Traces without this info (older kineto captures) fall back to the flat all-NPU model

3. **AstraSim run** (inside Docker):
   - Uses the `AstraSim_Analytical_Congestion_Unaware` analytical network backend
   - If `comm_group.json` is present it is passed via `--comm-group-configuration`
   - Compute nodes replay at measured duration; communication nodes are re-simulated under the specified network parameters

### `full` mode

Uses the official Chakra toolchain (`chakra_trace_link` + `chakra_converter`) to merge PyTorch execution traces with kineto device traces, producing fully attributed Chakra ET files with compute roofline enabled.

## Available Traces

Torchtitan traces (support process-group-aware simulation):

```
/data/ccl-bench_trace_collection/
  deepseek-v3-16b-torchtitan-ep4-dp2-pp2-tp4-perlmutter/   # 16 GPUs
  deepseek-v3-16b-torchtitan-ep4-dp2-tp4-perlmutter/
  deepseek-v3-16b-torchtitan-ep4-dp4-tp2-perlmutter/
  deepseek-v3-16b-torchtitan-ep8-dp2-pp2-tp4-perlmutter/
  deepseek-v3-16b-torchtitan-ep32-dp8-pp4-tp4-perlmutter/  # 128 GPUs
```

Full-fidelity traces (for `--mode full`):

```
trace_collection_backlog/llama-3.1-8b-torchtitan-perlmutter/
```

## Known Limitations

- **PP send/recv not simulated**: Pipeline-parallel point-to-point transfers (`ncclDevKernel_SendRecv`) are filtered out. The simulated step time excludes these communication events, so comm fraction will be understated for PP-heavy configurations.
- **Flat topology**: All communication groups share the same bandwidth and latency parameters. Intra-node (NVLink) and inter-node (InfiniBand/Slingshot) links are not modeled separately.
- **No overlap modeling**: AstraSim's congestion-unaware analytical backend reports zero compute-communication overlap; actual overlap is visible in the `simulation.log` per-rank `exposed communication` cycle counts.
