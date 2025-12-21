# llama3-8b-torchtitan

This workload collects traces for **Llama 3.x 8B** training using **TorchTitan** on a multi-node GPU cluster (e.g., Perlmutter).

## What’s in here
- `configs/`
  - `llama-3.1-8b-torchtitan-perlmutter.toml`: main run config (64×A100 preset)
  - `debug_model.toml`: smaller / debug-friendly config
- `scripts/`
  - `multinode_trainer.slurm`: Slurm job entrypoint
  - `multinode1.sh`, `multinode2.sh`: helper scripts used by the Slurm job
- `src/`
  - `profiling.py`: profiler toggles + export helpers
  - `logging.py`: lightweight logging utilities
  - `utils.py`: shared utilities used by the scripts
- `tools/`
  - `parse_nccl_trace_to_json.py`: converts NCCL trace logs into JSON

## How to run (typical Slurm flow)
1. **Edit config paths** inside the toml if needed (assets/output dirs).
2. Submit the job:
   ```bash
   sbatch scripts/multinode_trainer.slurm
   ```
3. After completion, collect exported traces (example):
   - `rank0_trace.json`, `rank1_trace.json`, ... (per rank)
   - upload to your trace storage (Cornell Box / S3 / etc.)

## Notes
- GPU/node counts are controlled primarily in `scripts/multinode_trainer.slurm` (e.g., nodes, tasks, GPUs per task).
- Profiler frequency and trace output folders are controlled in the toml (`[profiling]`) and/or `src/profiling.py`.
