# scripts

Launch scripts for the workload.

Typical responsibilities:
- Slurm resource requests (nodes, tasks, GPUs, CPUs, partition)
- Environment setup (modules / conda / venv / paths)
- Launching distributed training (torchrun / mpirun / srun, etc.)
- Passing config paths and enabling profiling

Files:
- `multinode_trainer.slurm`: primary Slurm submission script
- `multinode1.sh`, `multinode2.sh`: helper scripts invoked by the Slurm job

Tip: Keep Slurm directives at the top of the `.slurm` script and avoid “hidden” resource
assumptions inside the helper scripts.
