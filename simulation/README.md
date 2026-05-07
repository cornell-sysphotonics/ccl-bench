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
| `tpu-xla` | single TPU `*.trace.json` (JAX/XLA Chrome JSON) | One XLA iteration replayed as HLO/device kernels plus XLA collectives |

## Quick Start (no Docker required)

`mock_trace_gen.py` + `mock_pipeline.py` provide a full what-if simulation
without Docker, AstraSim, or real traces.  They use the same trace format and
network parameters as the full pipeline, so results can later be validated
against AstraSim.

```bash
# Generate 8-rank synthetic traces for Llama-3.1-8B (tp=4, dp=2)
python simulation/mock_trace_gen.py \
    --output-dir /tmp/mock_trace \
    --tp 4 --dp 2 --layers 32

# Baseline: A100 NVLink (400 GB/s) + Slingshot (25 GB/s)
python simulation/mock_pipeline.py \
    --trace-dir /tmp/mock_trace \
    --intra-bandwidth 400 --bandwidth 25

# What-if: 2× scale-out bandwidth
python simulation/mock_pipeline.py \
    --trace-dir /tmp/mock_trace \
    --intra-bandwidth 400 --bandwidth 50

# Run the full demo (all three scenarios):
bash simulation/examples/00_mock.sh
```

The mock traces are also compatible with `pipeline.py --mode comm-only` once
Docker is available, so they serve as test fixtures for the full pipeline.

## Prerequisites (full AstraSim pipeline)

Docker must be running with the `astra-sim:latest` image available. To build it:

```bash
cd plug-ins/astra-sim-hybrid-parallelism
docker build -t astra-sim:latest .
```

The image is ~14 GB and takes 20–40 minutes to build. It includes AstraSim, Chakra, and all dependencies.

No host Python packages are required beyond the standard library — all heavy lifting happens inside Docker.

## Quick Start (full pipeline)

```bash
# Baseline: deepseek-v3-16b on A100 Slingshot
python simulation/pipeline.py --mode comm-only \
    --trace-dir /data/ccl-bench_trace_collection/deepseek-v3-16b-torchtitan-ep4-dp2-pp2-tp4-perlmutter

# What-if: 2× scale-out network bandwidth
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
    --peak-perf 900 --mem-bw 3350 --gpus-per-node 8 \
    --intra-bandwidth 900 --bandwidth 50

# TPU/XLA: 8-chip TPU torus bandwidth what-if
python simulation/pipeline.py --mode tpu-xla \
    --trace-dir /data/ccl-bench_trace_collection/deepseek-v2-16b-maxtext-train-tp2-ep4-dp1-tpu \
    --compute-model kernels --tpu-ranks 8 --tpu-torus-dims 2,4 \
    --tpu-iteration-index 1 --bandwidth 3200 --latency 100
```

Run from the repo root (`/home/dd687/ccl-bench/`).

## Examples

Ready-to-run scripts are in `simulation/examples/`:

| Script | What it shows |
|--------|---------------|
| `00_mock.sh` | **No Docker** — synthetic traces + analytical pipeline; baseline vs 2× inter-BW vs H100 NVLink |
| `01_baseline.sh` | Single baseline run, ep4-dp2-tp4, A100 defaults |
| `02_bandwidth_sweep.sh` | Step time vs scale-out bandwidth (50→800 GB/s) |
| `03_algo_compare.sh` | `ring` vs `halving_doubling` vs `doubleBinaryTree` |
| `04_parallelism_compare.sh` | ep4-dp2-tp4 vs ep4-dp4-tp2 vs ep8-dp2-tp4 vs ep8-dp8 |
| `05_topology_compare.sh` | Scale-out Switch vs Ring vs FullyConnected with fixed scale-up |
| `06_hardware_generations.sh` | Two-tier A100 → H100 → next-gen network presets |
| `12_tpu_bandwidth_sweep_deepseekv2_maxtext.sh` | TPU v6e torus bandwidth sweep from a MaxText/XLA trace |

Run any example from the repo root:

```bash
bash simulation/examples/02_bandwidth_sweep.sh
```

## Arguments

```
--trace-dir PATH        Trace directory (required)
--mode {full,comm-only,tpu-xla}
                        Simulation mode (default: full)
--compute-model {gap,kernels}
                        comm-only compute model (default: gap)
--output-dir PATH       Output directory (default: /var/tmp/ccl_bench_sim_*)
```

**Network parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--topology` | `Switch` | Inter-node topology: `Ring`, `Switch`, or `FullyConnected` |
| `--bandwidth` | `200.0` | Inter-node per-link bandwidth in GB/s |
| `--latency` | `500.0` | Inter-node per-link latency in ns |
| `--gpus-per-node` | `4` | Ranks/GPUs sharing the intra-node fabric |
| `--intra-topology` | `FullyConnected` | Intra-node topology: `Ring`, `Switch`, or `FullyConnected` |
| `--intra-bandwidth` | `400.0` | Intra-node per-link bandwidth in GB/s |
| `--intra-latency` | `50.0` | Intra-node per-link latency in ns |

For multi-node traces, the pipeline writes a two-dimensional ASTRA network:

```yaml
topology: [ FullyConnected, Switch ]
npus_count: [ <gpus-per-node>, <num_nodes> ]
bandwidth: [ <intra-bandwidth>, <bandwidth> ]
latency: [ <intra-latency>, <latency> ]
```

Ranks are assumed to be node-contiguous: ranks `0..gpus_per_node-1` are on node 0,
the next block is on node 1, and so on. Single-node traces use the intra-node
topology and keep a one-dimensional network because there is no inter-node tier.

**System parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--compute-model` | `gap` | `gap` inserts replayed compute gaps between collectives; `kernels` emits non-NCCL GPU kernels plus measured idle gaps as replayed `COMP_NODE`s |
| `--kernel-dependency-mode` | `rank` | `kernels` mode only: `rank` adds rank-local chronological dependencies and rank-local gaps between compute and communication kernels when they do not invert global collective order; `stream` preserves CUDA stream ordering and per-stream gaps |
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
   - `ncclDevKernel_SendRecv` `all_to_allv` payload events are simulated as `ALL_TO_ALL` nodes with per-instance sizes normalized across ranks; tiny SendRecv control/barrier exchanges are excluded
   - With `--compute-model gap`, each collective op is preceded by a `COMP_NODE` whose `duration_micros` captures the measured compute gap since the previous collective in that process group
   - With `--compute-model kernels`, non-NCCL GPU kernels are emitted as replayed `COMP_NODE`s with their measured kernel durations. Positive idle intervals between emitted kernels are also emitted as gap `COMP_NODE`s. Dependencies preserve CUDA stream order and NCCL process-group ordering. By default, `--kernel-dependency-mode rank` adds rank-local compute↔communication dependencies and gaps when they do not invert the global collective order; use `--kernel-dependency-mode stream` to allow cross-stream overlap with per-stream gaps.

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

### TPU MaxText/XLA example

`simulation/examples/12_tpu_bandwidth_sweep_deepseekv2_maxtext.sh` uses `pipeline.py --mode tpu-xla` for:

```
/data/ccl-bench_trace_collection/deepseek-v2-16b-maxtext-train-tp2-ep4-dp1-tpu/
```

That trace is a single JAX/XLA TPU Chrome trace (`*.trace.json`), not the `rankN_trace.json` format consumed by `pipeline.py --mode comm-only`. TPU/XLA mode calls `gen_tpu_xla_chakra_et.py` inside the AstraSim Docker image, selects one clustered `jit_train_step` iteration, emits `chakra_trace.0.et` ... `chakra_trace.7.et`, and runs AstraSim with an 8-chip TPU torus represented as:

```yaml
topology: [ Ring, Ring ]
npus_count: [ 2, 4 ]
```

In TPU `--compute-model kernels`, non-collective HLO/device events with TPU device timing (`device_duration_ps`, `hlo_category`) are emitted as Chakra `COMP_NODE`s, while XLA collective HLO events such as `all-reduce`, `all-gather`, `reduce-scatter`, `all-to-all`, and `collective-permute` are emitted as `COMM_COLL_NODE`s. Positive idle intervals between rank-local emitted events are represented as gap `COMP_NODE`s, so the generated trace is closer to the GPU `--compute-model kernels --kernel-dependency-mode rank` path than the older collective-gap-only path.

This TPU path is still approximate:

- The MaxText trace may not expose exact collective payload sizes, so the example uses `FALLBACK_COMM_SIZE` when no byte count is present.
- If a collective event is not associated with a specific TPU rank, the generator mirrors it to all ranks so AstraSim can model synchronization.
- The model serializes emitted events per TPU rank; it does not reconstruct full XLA dependency DAG overlap or TPU compiler scheduling.
- XLA `broadcast` is treated as local compute, not a distributed collective.

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

- **PP/control SendRecv ops not simulated**: `ncclDevKernel_SendRecv` `all_to_allv` payload events are simulated, but tiny PP barrier/control exchanges are excluded. Variable-size `all_to_allv` payloads are normalized to the largest rank-local payload for each logical instance because AstraSim collective nodes require a single shared `comm_size` across participating ranks.
- **Node-contiguous rank assumption**: Two-tier topology assumes contiguous rank blocks per node. If a trace uses a different rank placement, adjust `--gpus-per-node` or reorder/remap the trace ranks before simulation.
- **Overlap is approximate**: `--compute-model gap` serializes compute gaps with communication inside one rank-local collective chain. `--compute-model kernels --kernel-dependency-mode rank` adds rank-local compute↔communication edges and positive rank-local idle gaps unless doing so would invert the global collective order, but it still does not reconstruct the full PyTorch dependency graph. `--kernel-dependency-mode stream` preserves per-stream event ordering and per-stream idle gaps, so compute and communication can overlap when they are on independent streams.
- **TPU/XLA simulation is experimental**: The TPU MaxText example converts one XLA step to Chakra directly and models non-collective HLO/device events as compute kernels, but payload sizes, rank mapping, and overlap are limited by what the TPU Chrome trace exposes.
