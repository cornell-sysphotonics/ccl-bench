# DeepSeek-V2-Lite DP+TP Parallelism (DP=2, TP=2)

**Workload ID:** `deepseek-v2-lite-torchtitan-dp+tp-perlmutter-16`

## Overview

This workload runs DeepSeek-V2-Lite (16B MoE) training with Data Parallel + Tensor Parallelism on a single Perlmutter GPU node.

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | DeepSeek-V2-Lite (16B MoE) |
| Framework | TorchTitan |
| Parallelism | DP=2, TP=2 |
| Total GPUs | 4 |
| Nodes | 1 |
| Batch Size | 2 |
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
$SCRATCH/ccl-bench-traces/deepseek-v2-lite-torchtitan-dp+tp-perlmutter-16/
```
