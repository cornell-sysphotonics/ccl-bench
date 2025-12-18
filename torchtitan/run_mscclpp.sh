#!/bin/bash

# 1. 自动定位关键库路径
# MSCCL++ 插件 (你的加速库)
MSCCLPP_LIB=$(find /pscratch/sd/x/xz987/CS5470/final_project/mscclpp -name "libmscclpp_nccl.so" | head -n 1)
# CUDA 运行时 (解决 undefined symbol)
CONDA_CUDA_LIB=$(find $CONDA_PREFIX/lib -name "libcudart.so.12*" | head -n 1)
# === [新增] 原生 NCCL 库 (用于 Fallback 回退) ===
# 通常在 Conda 环境的 lib 目录下，名字可能是 libnccl.so.2 或 libnccl.so
REAL_NCCL_LIB=$(find $CONDA_PREFIX/lib -name "libnccl.so.2" | head -n 1)

# 检查是否都找到了
if [ -z "$MSCCLPP_LIB" ] || [ -z "$REAL_NCCL_LIB" ]; then
    echo "❌ Error: Libraries not found!"
    echo "MSCCL++: $MSCCLPP_LIB"
    echo "NCCL: $REAL_NCCL_LIB"
    exit 1
fi

# 2. 设置注入环境变量 (LD_PRELOAD)
export LD_PRELOAD="${CONDA_CUDA_LIB}:${MSCCLPP_LIB}"

# 3. 指定算法配置文件
# 这里你指向了 reducescatter.json。这意味着：
# - ReduceScatter (Backward): 尝试用 MSCCL++ 加速
# - AllGather (Forward): 必须回退到 NCCL (因为你没有提供 AllGather 的 JSON)
export MSCCLPP_EXECUTION_PLAN_DIR=$(pwd)

# 4. === [关键修改] 开启 NCCL Fallback ===
# 允许 MSCCL++ 在遇到不懂的算子(如 Broadcast, AllGather)时，自动调用原生 NCCL 
export MSCCLPP_ENABLE_NCCL_FALLBACK=TRUE
export MSCCLPP_NCCL_LIB_PATH=$REAL_NCCL_LIB

# (可选) 强制指定回退哪些操作。不写这行默认是p "all" (除了 JSON 里有的) 
# export MSCCLPP_FORCE_NCCL_FALLBACK_OPERATION="allgather,broadcast,reducescatter,allreduce"
export MSCCLPP_FORCE_NCCL_FALLBACK_OPERATION="allreduce"

# 开启 Debug 以便观察
export MSCCLPP_DEBUG=INFO
export NCCL_DEBUG=INFO

echo "=== Running Experimental (MSCCL++) with Fallback ==="
echo "Plugin: $MSCCLPP_LIB"
echo "Fallback NCCL: $REAL_NCCL_LIB"
echo "Plan: $MSCCLPP_XML_FILE"

CONFIG=/pscratch/sd/x/xz987/CS5470/final_project/torchtitan/torchtitan/models/llama3/train_configs/debug_model.toml

# 5. 启动训练
# nsys profile \
#     --trace=cuda,nvtx,osrt \
#     --output=nsyslog/check_reducescatter_%p \
#     --delay=20 \
#     --duration=10 \
#     --force-overwrite=true \
#     torchrun --nproc_per_node=4 --standalone \
#         -m torchtitan.train --job.config_file $CONFIG

torchrun --nproc_per_node=4 --standalone \
    -m torchtitan.train --job.config_file $CONFIG