# qwen3-32b-torchtitan-fsdp+tp-perlmutter-group-16

Quick steps to set up, submit, and find outputs for this workload on Perlmutter.

## Setup (once)
1. Ensure you have Perlmutter access and a Hugging Face token; defaults for HF_HOME, HF_ASSETS_ROOT, VENV_DIR, and caches are set in env.sh.
2. From this directory run `bash install.sh` to create/update the virtualenv and install dependencies (safe to rerun).
3. Edit train_config.toml if you need to tweak batch size, steps, or knobs before launching.

## Run
- Submit the job with `./run.sh` (wraps `sbatch run.sbatch`). Use `./run.sh --dry-run` to inspect the Slurm header without launching.
- You can also run `sbatch run.sbatch` directly from a login node if you prefer.

## Outputs
- Slurm logs land in logs/*.out and logs/*.err.
- Profiler traces, dumps, and checkpoints go under traces/ with timestamped subdirectories (nsys or torch profiler outputs when enabled).
- Analysis reports are written to analysis_output/ by tools/main.py at the end of the job; regenerate manually with `python ../../tools/main.py --workload-dir .` if needed.

## Notes
- env.sh sets scratch/cache locations and NCCL defaults; common.sh provides the trace setup and torchrun helpers used by run.sbatch.
