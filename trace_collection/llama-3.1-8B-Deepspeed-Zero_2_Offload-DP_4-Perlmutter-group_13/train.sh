#!/bin/bash

module load conda
conda activate final_project

export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/pscratch/sd/c/cy566/nsight-system/bin:$PATH # Update this path as necessary

NUM_GPUS=4
CONFIG_FILE="ds_config_zero2.json"
SCRIPT_FILE="train.py"
OUTPUT_NAME="/pscratch/sd/c/cy566/output/llama_zero2_comm_trace"

echo "Starting Nsight Systems profiling..."

nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=$OUTPUT_NAME \
    --force-overwrite=true \
    --stats=true \
    deepspeed --num_gpus=4 train.py \
    --deepspeed ds_config_zero2.json

echo "Profiling finished. Output saved to ${OUTPUT_NAME}.nsys-rep"