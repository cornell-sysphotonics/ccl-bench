#!/bin/bash
#SBATCH -A m4999
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -J llama_mscclpp
#SBATCH -o logs/bench-%j.out

module load pytorch/2.3.1 
export OMP_NUM_THREADS=1 

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes allocated: $SLURM_JOB_NUM_NODES"

# NCCL Baseline
echo "[Phase 1] Running NCCL Baseline..."

# 修改点：删除了 --gpus-per-node=4
# 因为 #SBATCH 已经分配了 GPU，srun 会自动让这些 GPU 对任务可见
# 保留 --gpu-bind=none 是为了让每个进程都能看到该节点上的所有4张卡
srun -u -n 8 --ntasks-per-node=4 --gpu-bind=none --export=ALL \
    nsys profile \
    --trace=cuda,nvtx,osrt,cublas-verbose \
    --output=nsys_nccl_rank%q{SLURM_PROCID} \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    python train_llama_nccl.py

# MSCCL++ Experiment
echo "[Phase 2] Running MSCCL++ Experiment..."

# 修改点：删除了 --gpus-per-node=4
srun -u -n 8 --ntasks-per-node=4 --gpu-bind=none --export=ALL \
    nsys profile \
    --trace=cuda,nvtx,osrt,cublas-verbose \
    --output=nsys_mscclpp_rank%q{SLURM_PROCID} \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    python train_llama.py