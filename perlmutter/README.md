# Perlmutter Slurm Scripts

This directory contains Slurm batch scripts for running TorchTitan training jobs on NERSC Perlmutter with trace collection for CCL-Bench.

## Directory Structure

```
perlmutter/
├── README.md                           # This file
├── setup_env.sh                        # Environment setup script (run once)
├── activate.sh                         # Environment activation (created by setup)
├── common.sh                           # Shared configuration and functions
├── submit_all.sh                       # Submit all workloads at once
├── run_llama3_8b_tp.sbatch            # LLaMA-3.1-8B with Tensor Parallelism
├── run_llama3_8b_pp.sbatch            # LLaMA-3.1-8B with Pipeline Parallelism
├── run_deepseek_v2_lite_dp_tp.sbatch  # DeepSeek-V2-Lite with DP+TP
├── run_deepseek_v2_lite_dp_pp.sbatch  # DeepSeek-V2-Lite with DP+PP
├── run_qwen3_32b_3d.sbatch            # Qwen3-32B with 3D parallelism
├── run_qwen3_32b_dp_tp.sbatch         # Qwen3-32B with DP+TP
└── run_qwen3_32b_dp_pp.sbatch         # Qwen3-32B with DP+PP
```

## Prerequisites

1. **NERSC Account** with GPU allocation on Perlmutter
2. **Python environment** (uv preferred, or standard venv)

## Quick Start

### 1. Initial Setup (Run Once on Login Node)

```bash
# On Perlmutter login node
cd $SCRATCH
git clone https://github.com/IanHollow/ccl-torchtitan-train.git ccl-bench
cd ccl-bench

# Run the setup script
./perlmutter/setup_env.sh
```

This will:
- Create a Python virtual environment using uv (or fallback to venv)
- Install PyTorch with CUDA 12.9 support
- Clone and set up TorchTitan
- Create an activation script at `perlmutter/activate.sh`

### 2. Configure Your Allocation

Edit `perlmutter/common.sh` and set your NERSC allocation:

```bash
export NERSC_ALLOCATION="m1234"  # Replace with your actual allocation
```

You can find your allocation with:
```bash
sacctmgr show assoc user=$USER
```

### 3. Submit a Job

```bash
# Single node job (LLaMA-8B with TP)
sbatch perlmutter/run_llama3_8b_tp.sbatch

# Multi-node job (Qwen-32B 3D parallelism)
sbatch perlmutter/run_qwen3_32b_3d.sbatch

# Submit all workloads
./perlmutter/submit_all.sh
```

### 4. Check Job Status

```bash
squeue -u $USER
```

### 5. View Results

After job completion, traces will be organized in:
```
trace_collection/<workload_name>/
├── <workload>_<timestamp>.nsys-rep    # NSight Systems trace
├── profile_trace/                      # TorchTitan profiler output
│   ├── *rank0*.json                   # Kineto trace (rank 0)
│   ├── *rank1*.json                   # Kineto trace (rank 1)
│   └── ...
├── kineto_trace_0.json                # Symlink to rank-0 trace (for tools)
└── *.log                              # Training logs
```

## Workload Configuration Summary

| Config | Model | Parallelism | GPUs | Nodes |
|--------|-------|-------------|------|-------|
| `llama3_8b_tp` | LLaMA-3.1-8B | TP=4 | 4 | 1 |
| `llama3_8b_pp` | LLaMA-3.1-8B | PP=4 | 4 | 1 |
| `deepseek_v2_lite_dp_tp` | DeepSeek-V2-Lite | DP=2, FSDP=2, TP=2 | 8 | 2 |
| `deepseek_v2_lite_dp_pp` | DeepSeek-V2-Lite | DP=2, FSDP=2, PP=2 | 8 | 2 |
| `qwen3_32b_3d` | Qwen3-32B | DP=2, TP=4, PP=2 | 16 | 4 |
| `qwen3_32b_dp_tp` | Qwen3-32B | DP=4, TP=2 | 8 | 2 |
| `qwen3_32b_dp_pp` | Qwen3-32B | DP=4, PP=2 | 8 | 2 |

## Trace Collection Details

Each job collects three types of traces:

### 1. NSight Systems (`nsys`)
- **Files**: `<workload>_<timestamp>.nsys-rep`
- **Contents**: GPU/CPU timeline, CUDA kernels, NVTX annotations, memory usage
- **Analysis**: Use `nsys stats` CLI or NSight Systems GUI

### 2. PyTorch Profiler / Kineto Traces
- **Location**: `profile_trace/` subdirectory
- **Files**: `*trace*.json` (one per rank)
- **Contents**: PyTorch operator traces, NCCL collective details
- **Format**: Chrome trace format (open in `chrome://tracing`)

### 3. Convenience Symlinks
- **Files**: `kineto_trace_0.json` (symlink)
- **Purpose**: Provides consistent location for metric analysis tools

## Analyzing Traces

After collecting traces, use the CCL-Bench metric tools:

```bash
# Activate environment
source perlmutter/activate.sh

# Count NCCL collective calls
ccl-metrics --trace trace_collection/llama3_8b_tp --metric coll_call_num

# Measure throughput
ccl-metrics --trace trace_collection/llama3_8b_tp --metric throughput_tokens

# Available metrics:
#   coll_call_num       - Count of NCCL collective operations
#   throughput_tokens   - Training throughput (tokens/sec)
#   iter_time           - Per-iteration wall-clock time
#   pipeline_bubble     - Pipeline parallelism bubble time
#   comm_comp_overlap   - Communication/computation overlap ratio
#   straggler_lag       - Slowest rank lag time
#   traffic_distribution - Communication traffic per collective type
```

### Extracting NSight Statistics

```bash
# Generate summary report
nsys stats trace_collection/llama3_8b_tp/llama3_8b_tp_*.nsys-rep

# Export GPU kernel summary to JSON
nsys stats --report gpu-kern-summary --format json \
    trace_collection/llama3_8b_tp/llama3_8b_tp_*.nsys-rep > gpu_stats.json
```

## Test Run (Recommended First Step)

Before running full workloads, do a quick validation:

1. Edit `train_configs/llama3_8b_tp.toml` and set:
   ```toml
   [training]
   steps = 20  # Quick test
   ```

2. Submit and verify:
   ```bash
   sbatch perlmutter/run_llama3_8b_tp.sbatch
   # Wait for completion...
   ls trace_collection/llama3_8b_tp/
   ls trace_collection/llama3_8b_tp/profile_trace/
   ```

3. You should see:
   - `.nsys-rep` file(s)
   - `profile_trace/` directory with JSON traces

## Troubleshooting

### Job fails immediately
- Check allocation: `sacctmgr show assoc user=$USER`
- Verify GPU availability: `sinfo -p gpu`
- Check `NERSC_ALLOCATION` in `common.sh` is not `CHANGE_ME`

### NCCL errors
- Add `export NCCL_DEBUG=INFO` to the sbatch script for details
- Ensure `module load cudatoolkit/12.9` completes successfully

### Out of memory
- Reduce `local_batch_size` in the TOML config
- Enable activation checkpointing: `mode = "full"`

### Traces not generated
- Check `enable_profiling = true` in the TOML config
- Verify trace directory is writable
- Check job stderr for profiler errors

### Metric tools can't find traces
- Traces should be in `trace_collection/<workload>/profile_trace/`
- A symlink `kineto_trace_0.json` is created automatically
- If missing, manually find and link: `ln -s profile_trace/*rank0*.json kineto_trace_0.json`
