# Collective Communication Profiler

Profile AllReduce, AllGather, and ReduceScatter operations across multiple GPUs and nodes.
Supports NCCL, NCCLX (via torchcomms), and GLOO backends. Note that there is no workload card
because this microprofiling is run on tensor sizes of 1M, 4M, 16M.

## Requirements

### Basic Setup (NCCL and GLOO)
```bash
conda create -n micro_profiling python=3.10
conda activate micro_profiling
pip install torch
```

### For NCCLX Support
```bash
pip install --pre torch torchcomms --index-url https://download.pytorch.org/whl/nightly/cu126
```

Note: Replace `cu126` with your CUDA version (cu126, cu128, cu129, cu130)

## Profiling

### Multi-Node (8 GPUs across 2 nodes)
```bash
# Allocate resources
salloc --nodes 2 --qos interactive --time 01:00:00 --constraint gpu --gpus 8 --account YOUR_ACCOUNT

# Load environment
module load conda
conda activate micro_profiling
cd <this-directory>

# Set master address (use current node hostname)
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Run profiling
srun --ntasks=8 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=closest python3 micro_profiling.py --backend ncclx
```

### Optimized for Perlmutter Slingshot Network

For best multi-node performance, set these environment variables before running:
```bash
export NCCL_NET=AWS Libfabric
export NCCL_NET_GDR_LEVEL=PHB
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
```

## Output Format

Results are saved to timestamped files: `profile_{backend}_world{N}_{timestamp}.txt`

Example output:
```
================================================================================
Configuration: World Size=4
Backend: nccl (torch.distributed)
================================================================================

Tensor Size: 1,048,576 elements (4.00 MB)
  AllReduce:     0.090 ms | AlgBW: 64.78 GB/s | BusBW: 43.19 GB/s
  AllGather:     0.179 ms | AlgBW: 16.33 GB/s | BusBW: 16.33 GB/s
  ReduceScatter: 0.174 ms | AlgBW: 16.84 GB/s | BusBW: 16.84 GB/s
```

### Metrics Explanation

- **Time (ms)**: Average operation latency
- **AlgBW (GB/s)**: Algorithm bandwidth - theoretical data movement
- **BusBW (GB/s)**: Bus bandwidth - actual data transmitted over network

## Troubleshooting

### "MASTER_ADDR not set" error
Ensure you've exported MASTER_ADDR before running srun:
```bash
export MASTER_ADDR=$(hostname)
```

### Low multi-node bandwidth
Set Perlmutter-specific NCCL environment variables (see "Optimized for Perlmutter" section above).

### "ncclx backend not available"
Install torchcomms or use `--backend nccl` instead.