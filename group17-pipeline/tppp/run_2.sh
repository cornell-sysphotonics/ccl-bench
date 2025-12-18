#!/bin/bash

# 1. 检查环境
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: Conda environment not activated! Please run 'conda activate ccl-bench' first."
    exit 1
fi

# 2. 准备路径
INC_PATH="${CONDA_PREFIX}/include"
LIB_PATH="${CONDA_PREFIX}/lib"

# 3. 构造远程执行命令
# 注意：我们将 IP 的获取逻辑移到了这里面
CMD="
export CPATH=${INC_PATH}:\$CPATH
export C_INCLUDE_PATH=${INC_PATH}:\$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=${INC_PATH}:\$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=${LIB_PATH}:\$LIBRARY_PATH
export LD_LIBRARY_PATH=${LIB_PATH}:\$LD_LIBRARY_PATH

# === 关键修改 ===
# 强制使用本地回环 IP，避开 hsn0 网络握手问题
# 因为在 TP=2, PP=2 模式下，AllReduce 不需要跨节点
export MSCCLPP_COMM_ID_IP=127.0.0.1

echo \"[Rank \${SLURM_PROCID}] Running on \$(hostname), Forcing IP: \$MSCCLPP_COMM_ID_IP\"

python tppp_train_llama_mscclpp.py --tp_size 2 --pp_size 2
"

# 4. 提交任务
# 去掉了 --export 里的 IP 设置，避免语法错误

srun -u -N 2 -n 8 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
    --export=ALL \
    bash -c "$CMD"
    
srun -u -N 2 -n 8 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --export=ALL \
    python tppp_train_llama_nccl.py --tp_size 2 --pp_size 2