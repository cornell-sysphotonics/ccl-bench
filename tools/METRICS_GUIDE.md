# Metrics Extraction Guide

This document outlines the various metrics that can be extracted from torch profile traces and memory snapshots.

## Available Data Sources

### 1. Profile Traces (JSON format)
- **Location**: `profile_traces/iteration_*/rank*_trace.json`
- **Format**: Chrome trace format (JSON)
- **Contains**:
  - CPU operations (`cpu_op` category)
  - CUDA kernels (`kernel` category)
  - GPU memory operations (`gpu_memcpy`, `gpu_memset`)
  - Communication operations (FSDP, NCCL)
  - Forward/backward pass markers
  - Operation durations and timestamps

### 2. Memory Snapshots (Pickle format)
- **Location**: `memory_snapshot/iteration_*/rank*_memory_snapshot.pickle`
- **Format**: Python pickle
- **Contains**:
  - Memory segments (allocated, active, inactive)
  - Device traces
  - Allocator settings
  - Memory fragmentation data

## Extractable Metrics

### Performance Metrics

#### 1. **Throughput** ✅ (Already implemented)
- Tokens per second
- Iteration time
- Per-rank throughput

#### 2. **Communication Metrics**
- **Communication volume**: Total bytes transferred in all_gather, all_reduce, reduce_scatter
- **Communication time**: Total time spent in collective operations
- **Communication frequency**: Number of collective calls per iteration
- **Communication breakdown**: Time per operation type (all_gather, all_reduce, etc.)
- **Communication efficiency**: Bytes transferred per second

#### 3. **Compute Metrics**
- **GPU utilization**: Percentage of time GPU is executing kernels
- **Kernel time**: Total time spent in CUDA kernels
- **CPU overhead**: Time spent in CPU operations
- **Compute efficiency**: Useful compute time vs total time
- **Kernel breakdown**: Time per kernel type/name

#### 4. **Memory Metrics** (from memory snapshots)
- **Peak memory usage**: Maximum memory allocated during iteration
- **Memory efficiency**: Active memory / Total allocated memory
- **Memory fragmentation**: Fragmentation ratio
- **Memory allocation patterns**: Allocation/deallocation frequency
- **Memory per device**: Memory usage per GPU

#### 5. **Training Phase Metrics**
- **Forward pass time**: Time for forward pass
- **Backward pass time**: Time for backward pass
- **Optimizer step time**: Time for optimizer updates
- **Phase breakdown**: Percentage of time in each phase

#### 6. **FSDP-Specific Metrics**
- **FSDP overhead**: Time spent in FSDP operations (all_gather, reduce_scatter, reshard)
- **FSDP prefetch efficiency**: How well prefetching works
- **FSDP communication time**: Time in FSDP collectives
- **FSDP copy time**: Time in copy_in/copy_out operations

#### 7. **Overlap Metrics**
- **Compute-communication overlap**: How much compute and communication overlap
- **Overlap efficiency**: Percentage of communication that overlaps with compute
- **Bubble time**: Idle time waiting for communication

#### 8. **Variability Metrics**
- **Straggler detection**: Which ranks are slower
- **Time variance**: Standard deviation of iteration times across ranks
- **Synchronization overhead**: Time spent waiting for slowest rank

#### 9. **Hardware Saturation**
- **Memory bandwidth utilization**: How well memory bandwidth is used
- **Compute intensity**: Operations per byte transferred
- **Pipeline efficiency**: How well the pipeline is utilized

## Implemented Metric Tools

| Tool | Description | Source Data |
|------|-------------|-------------|
| `throughput/` | Tokens per second, iteration time | Profile traces |
| `communication/` | Communication time, volume, frequency by operation type | Profile traces |
| `memory/` | Peak memory, fragmentation, efficiency | Memory snapshots |
| `overlap/` | Compute-communication overlap, bubble time | Profile traces |
| `training_phases/` | Forward/backward/optimizer time breakdown | Profile traces |
| `fsdp/` | FSDP-specific metrics (all_gather, reduce_scatter, reshard, etc.) | Profile traces |
| `straggler/` | Straggler detection, load imbalance, sync overhead | Profile traces |
| `kernel/` | Kernel analysis by type (GEMM, attention, elementwise, etc.) | Profile traces |
| `gpu_utilization/` | GPU busy time, idle time, utilization percentage | Profile traces |

## Usage

Each metric tool follows the same interface:

```python
def metric_cal(
    trace_dir: str,
    workload_card_path: str | None = None,
    profile_mode: str = "auto",
) -> dict[str, Any]:
    """
    Calculate metric from profile traces.

    Returns:
        Dictionary with metric results
    """
    pass
```

### Run Individual Tool

```python
from tools.communication import metric_cal

result = metric_cal("/path/to/profile_traces/iteration_10/")
print(result)
```

### Run All Tools via main.py

```bash
python -m tools.main --workload-dir <path> --tools <metric_name>
```

### Run Directly on Trace Directory

```python
# Example: Run all tools on a trace directory
from pathlib import Path
import json

trace_dir = "/path/to/profile_traces/iteration_10/"

# Import and run each tool
from tools.throughput import metric_cal as throughput_cal
from tools.communication import metric_cal as communication_cal
from tools.memory import metric_cal as memory_cal
from tools.overlap import metric_cal as overlap_cal
from tools.training_phases import metric_cal as phases_cal
from tools.fsdp import metric_cal as fsdp_cal
from tools.straggler import metric_cal as straggler_cal
from tools.kernel import metric_cal as kernel_cal
from tools.gpu_utilization import metric_cal as gpu_cal

results = {
    "throughput": throughput_cal(trace_dir),
    "communication": communication_cal(trace_dir),
    "overlap": overlap_cal(trace_dir),
    "training_phases": phases_cal(trace_dir),
    "fsdp": fsdp_cal(trace_dir),
    "straggler": straggler_cal(trace_dir),
    "kernel": kernel_cal(trace_dir),
    "gpu_utilization": gpu_cal(trace_dir),
}

# Memory needs to find snapshots relative to trace directory
# results["memory"] = memory_cal(trace_dir)

print(json.dumps(results, indent=2))
```
