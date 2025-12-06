# LLaMA-3.1-8B Pure Pipeline Parallelism (PP=4)

**Workload ID:** `llama3.1-8b-torchtitan-pp-perlmutter-16`

## Overview

This workload runs LLaMA-3.1-8B training with Pure Pipeline Parallelism (PP=4) on a single Perlmutter GPU node.

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | LLaMA-3.1-8B |
| Framework | TorchTitan |
| Parallelism | PP=4 |
| Total GPUs | 4 |
| Nodes | 1 |
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
$SCRATCH/ccl-bench-traces/llama3.1-8b-torchtitan-pp-perlmutter-16/
```
