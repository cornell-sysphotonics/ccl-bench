# Qwen3-32B Full 3D Parallelism (DP=2, TP=2, PP=2)

**Workload ID:** `qwen3-32b-torchtitan-3d-perlmutter-16`

## Overview

This workload runs Qwen3-32B training with full 3D Parallelism (DP + TP + PP) on two Perlmutter GPU nodes (8 GPUs total).

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-32B |
| Framework | TorchTitan |
| Parallelism | DP=2, TP=2, PP=2 |
| Total GPUs | 8 |
| Nodes | 2 |
| Batch Size | 4 |
| Sequence Length | 2048 |
| Training Steps | 80 |

## Files

- `run.sh` - Simple wrapper to submit the SLURM job
- `run.sbatch` - SLURM batch script
- `train_config.toml` - TorchTitan training configuration
- `workload_card.yaml` - Workload metadata for CCL-Bench

## Running

```bash
# From this directory
./run.sh

# Or with dry-run to preview
./run.sh --dry-run

# Set profiling mode before running
export PROFILE_MODE=nsys  # Options: both, nsys, torch
./run.sh
```

## Traces Output

After job completion, traces will be in:
```
$SCRATCH/ccl-bench-traces/qwen3-32b-torchtitan-3d-perlmutter-16/
```
