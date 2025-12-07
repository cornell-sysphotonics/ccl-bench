# Perlmutter Slurm Scripts

This directory contains shared configuration and utility scripts for running TorchTitan training jobs on NERSC Perlmutter with trace collection for CCL-Bench.

## Directory Structure

```
perlmutter/
├── README.md           # This file
├── setup_env.sh        # Environment setup script (run once)
├── activate.sh         # Environment activation (source this)
├── common.sh           # Shared configuration and functions
└── submit_all.sh       # Submit all workloads at once
```

Each workload's execution scripts are now located in their respective folders under `trace_collection/`:

```
trace_collection/
├── llama3.1-8b-torchtitan-tp-perlmutter-16/
│   ├── run.sh              # Simple wrapper to submit job
│   ├── run.sbatch          # SLURM batch script
│   ├── train_config.toml   # TorchTitan configuration
│   ├── workload_card.yaml  # Workload metadata
│   └── pyproject.toml      # uv workspace member
├── llama3.1-8b-torchtitan-pp-perlmutter-16/
│   └── ... (same structure)
├── deepseek-v2-lite-torchtitan-dp+tp-perlmutter-16/
│   └── ...
└── ... (other workloads)
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
# Option 1: Use run.sh from the workload folder
cd trace_collection/llama3.1-8b-torchtitan-tp-perlmutter-16
./run.sh

# Option 2: Submit directly using sbatch
sbatch trace_collection/llama3.1-8b-torchtitan-tp-perlmutter-16/run.sbatch

# Option 3: Submit all workloads at once
./perlmutter/submit_all.sh
```

### 4. Check Job Status

```bash
squeue -u $USER
```

### 5. View Results

After job completion, traces will be organized in:
```
$SCRATCH/ccl-bench-traces/<workload_folder>/
├── <workload>_<timestamp>.nsys-rep    # NSight Systems trace
├── profile_trace/                      # TorchTitan profiler output
│   ├── *rank0*.json                   # Kineto trace (rank 0)
│   ├── *rank1*.json                   # Kineto trace (rank 1)
│   └── ...
├── kineto_trace_0.json                # Symlink to rank-0 trace (for tools)
└── *.log                              # Training logs
```

## Workload Configuration

Training configurations are stored in `trace_collection/<workload_folder>/train_config.toml`.

| Workload Folder | Model | Parallelism | GPUs | Nodes |
|-----------------|-------|-------------|------|-------|
| `llama3.1-8b-torchtitan-tp-perlmutter-16` | LLaMA-3.1-8B | TP=4 | 4 | 1 |
| `llama3.1-8b-torchtitan-pp-perlmutter-16` | LLaMA-3.1-8B | PP=4 | 4 | 1 |
| `deepseek-v2-lite-torchtitan-dp+tp-perlmutter-16` | DeepSeek-V2-Lite | DP=2, TP=2 | 4 | 1 |
| `deepseek-v2-lite-torchtitan-dp+pp-perlmutter-16` | DeepSeek-V2-Lite | DP=2, PP=2 | 4 | 1 |
| `qwen3-32b-torchtitan-3d-perlmutter-16` | Qwen3-32B | DP=2, TP=2, PP=2 | 8 | 2 |
| `qwen3-32b-torchtitan-dp+tp-perlmutter-16` | Qwen3-32B | DP=2, TP=2 | 4 | 1 |
| `qwen3-32b-torchtitan-dp+pp-perlmutter-16` | Qwen3-32B | DP=2, PP=2 | 4 | 1 |

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

### 4. PyTorch Execution Trace
- **Files**: `torch_et_0.json` (symlink to `torch_et_<rank>.json`)
- **Purpose**: ExecutionTraceObserver output collected alongside Kineto for tool compatibility
- **Check**: After a run, verify both Kineto and torch_et traces exist:
  ```bash
  ls $SCRATCH/ccl-bench-traces/<workload>/kineto_trace_0.json
  ls $SCRATCH/ccl-bench-traces/<workload>/torch_et_0.json
  ```
  If missing, rerun with `PROFILE_MODE=both` (or `torch`) to capture torch_et.

## Analyzing Traces

After collecting traces, use the CCL-Bench metric tools:

```bash
# Activate environment
source perlmutter/activate.sh

# Count NCCL collective calls
ccl-metrics --trace $SCRATCH/ccl-bench-traces/llama3.1-8b-torchtitan-tp-perlmutter-16 --metric coll_call_num_16

# Measure throughput
ccl-metrics --trace $SCRATCH/ccl-bench-traces/llama3.1-8b-torchtitan-tp-perlmutter-16 --metric throughput_tokens_16

# Available metrics (all use _16 suffix for group 16):
#   coll_call_num_16       - Count of NCCL collective operations
#   throughput_tokens_16   - Training throughput (tokens/sec)
#   iter_time_16           - Per-iteration wall-clock time
#   pipeline_bubble_16     - Pipeline parallelism bubble time
#   comm_comp_overlap_16   - Communication/computation overlap ratio
#   straggler_lag_16       - Slowest rank lag time
#   traffic_distribution_16 - Communication traffic per collective type

# List all available metrics
ccl-metrics --list-metrics
```

### Extracting NSight Statistics

```bash
# Generate summary report
nsys stats $SCRATCH/ccl-bench-traces/llama3.1-8b-torchtitan-tp-perlmutter-16/*.nsys-rep

# Export GPU kernel summary to JSON
nsys stats --report gpu-kern-summary --format json \
    $SCRATCH/ccl-bench-traces/llama3.1-8b-torchtitan-tp-perlmutter-16/*.nsys-rep > gpu_stats.json
```

## Test Run (Recommended First Step)

Before running full workloads, do a quick validation:

1. Edit `trace_collection/llama3.1-8b-torchtitan-tp-perlmutter-16/train_config.toml` and set:
   ```toml
   [training]
   steps = 20  # Quick test
   ```

2. Submit and verify:
   ```bash
   sbatch perlmutter/run_llama3_8b_tp.sbatch
   # Wait for completion...
   ls $SCRATCH/ccl-bench-traces/llama3.1-8b-torchtitan-tp-perlmutter-16/
   ls $SCRATCH/ccl-bench-traces/llama3.1-8b-torchtitan-tp-perlmutter-16/profile_trace/
   ```

3. You should see:
   - `.nsys-rep` file(s)
   - `profile_trace/` directory with JSON traces

## Troubleshooting

### Job fails immediately
- Check allocation: `sacctmgr show assoc user=$USER`
- Verify GPU availability: `sinfo -p gpu`
- Check `NERSC_ALLOCATION` in `common.sh` or set it via environment variable

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
- Traces should be in `$SCRATCH/ccl-bench-traces/<workload_folder>/profile_trace/`
- A symlink `kineto_trace_0.json` is created automatically
- If missing, manually find and link: `ln -s profile_trace/*rank0*.json kineto_trace_0.json`
