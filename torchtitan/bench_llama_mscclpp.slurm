#!/bin/bash
#SBATCH -A m4999             # 请替换为你的 Allocation ID
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:02:00
#SBATCH -N 1                 # 2个节点 (Project 要求多机通信)
#SBATCH --ntasks-per-node=4  # Perlmutter 每个节点 4 个 A100
#SBATCH --gpus-per-node=4
#SBATCH -J llama_mscclpp
#SBATCH -o logs/%x-%j.out

# 1. 环境准备
source ~/.bashrc
conda activate ccl-bench

# 2. 自动定位关键库路径
MSCCLPP_LIB=$(find /pscratch/sd/x/xz987/CS5470/final_project/mscclpp -name "libmscclpp_nccl.so" | head -n 1)
CONDA_CUDA_LIB=$(find $CONDA_PREFIX/lib -name "libcudart.so.12*" | head -n 1)

# 构建 LD_PRELOAD
export MY_PRELOAD="${CONDA_CUDA_LIB}:${MSCCLPP_LIB}"

# 3. 网络配置
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=hsn

# export MSCCLPP_XML_FILE=/pscratch/sd/x/xz987/CS5470/final_project/mscclpp/allgather.xml

# 开启 Debug
export MSCCLPP_DEBUG=INFO
export NCCL_DEBUG=INFO 

# 4. 配置文件路径 (请确认此绝对路径正确)
CONFIG=/pscratch/sd/x/xz987/CS5470/final_project/torchtitan/torchtitan/models/llama3/train_configs/llama3_8b_dp_bench.toml

# 5. 运行 TorchTitan (关键修复：使用 bash -c 注入环境变量)
echo "Starting Training with MSCCL++ (Job ID: $SLURM_JOB_ID)..."

srun --label --export=ALL,LD_PRELOAD=$MY_PRELOAD \
    bash -c "
    export LOCAL_RANK=\$SLURM_LOCALID
    export RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    python -m torchtitan.train --job.config_file $CONFIG
    "