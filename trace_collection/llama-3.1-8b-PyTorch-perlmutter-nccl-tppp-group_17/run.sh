if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# 2. 设置 CPATH，让 nvcc 能找到 infiniband/verbs.h
export CPATH=$CONDA_PREFIX/include:$CPATH
export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH

# 3. 设置库路径 (防止运行时报错)
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# mpirun -np 4 \
#     -x LD_LIBRARY_PATH \
#     -x MASTER_ADDR=localhost \
#     -x MASTER_PORT=12355 \
#     python train_llama.py

srun -u -n 4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --export=ALL \
    python tppp_train_llama_nccl.py --tp_size 1 --pp_size 2

# srun -u -n 4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --export=ALL,MSCCLPP_COMM_ID_IP=$(hostname -i) \
#     python tppp_train_llama_mscclpp.py --tp_size 1 --pp_size 2