# Trace Collection

This folder contains workload configurations and execution scripts for CCL-Bench trace collection on Perlmutter. On GitHub, this folder stores the metadata (workload cards), TorchTitan configuration files, and execution scripts.

You will upload your actual traces to Cornell Box in a single zip file named `group-16`.

## Folder Structure

Each workload has its own folder following the naming convention: `<model>-<framework+parallelism>-<platform>-<group_number>`

Each folder contains:
- `run.sh` - Simple wrapper script to submit the job
- `run.sbatch` - SLURM batch script for Perlmutter
- `train_config.toml` - TorchTitan training configuration
- `workload_card.yaml` - Metadata describing the workload
- `pyproject.toml` - uv workspace member configuration

## Running Workloads

### Option 1: Run from workload folder
```bash
cd trace_collection/llama3.1-8b-torchtitan-tp-perlmutter-16
./run.sh              # Submit job
./run.sh --dry-run    # Preview without submitting
```

### Option 2: Submit directly with sbatch
```bash
sbatch trace_collection/llama3.1-8b-torchtitan-tp-perlmutter-16/run.sbatch
```

### Option 3: Submit all workloads
```bash
./perlmutter/submit_all.sh
```

## Current List of Workloads

### LLaMA-3.1-8B (Group 16)
| Folder | Parallelism | GPUs | Nodes |
|--------|-------------|------|-------|
| `llama3.1-8b-torchtitan-pp-perlmutter-16` | Pure Pipeline (PP=4) | 4 | 1 |
| `llama3.1-8b-torchtitan-tp-perlmutter-16` | Pure Tensor (TP=4) | 4 | 1 |

### DeepSeek-V2-Lite (Group 16)
| Folder | Parallelism | GPUs | Nodes |
|--------|-------------|------|-------|
| `deepseek-v2-lite-torchtitan-dp+pp-perlmutter-16` | DP=2, PP=2 | 4 | 1 |
| `deepseek-v2-lite-torchtitan-dp+tp-perlmutter-16` | DP=2, TP=2 | 4 | 1 |

### Qwen3-32B (Group 16)
| Folder | Parallelism | GPUs | Nodes |
|--------|-------------|------|-------|
| `qwen3-32b-torchtitan-3d-perlmutter-16` | 3D (DP=2, TP=2, PP=2) | 8 | 2 |
| `qwen3-32b-torchtitan-dp+pp-perlmutter-16` | DP=2, PP=2 | 4 | 1 |
| `qwen3-32b-torchtitan-dp+tp-perlmutter-16` | DP=2, TP=2 | 4 | 1 |

### Other Groups (Examples)
- `deepseek-v2-lite-vllm-lambda1` - DeepSeek-V2-Lite inference with vLLM

## Trace Upload Instructions

1. Collect traces using `run.sh` in each workload folder
2. Traces will be saved to `$SCRATCH/ccl-bench-traces/<workload_folder>/`
3. Zip the traces: `zip -r group-16.zip <trace_folders>`
4. Upload to Cornell Box

## Sample Traces

Sample traces are stored in the [Google Drive](https://drive.google.com/drive/u/0/folders/1shHsa3WvlGh9YRaX7iqYBYTLnwdDfLX6)
