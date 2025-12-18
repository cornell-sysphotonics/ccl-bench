if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi
# mpirun -np 4 \
#     -x LD_LIBRARY_PATH \
#     -x MASTER_ADDR=localhost \
#     -x MASTER_PORT=12355 \
#     python train_llama.py

# srun -u -n 4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --export=ALL \
#     python train_llama_nccl.py

srun -u -n 4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --export=ALL \
    python train_llama_mscclpp.py