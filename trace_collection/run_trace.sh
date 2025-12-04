#!/bin/bash
module load conda
conda activate /pscratch/sd/q/qiaox226/ccl-bench/envs/final_project

export HF_HOME=$PSCRATCH/huggingface
export HF_DATASETS_CACHE=$PSCRATCH/huggingface_datasets
export HF_TOKEN="local_model"  # 使用本地模型，不需要真实 token
export PATH=/global/homes/q/qiaox226/nsight-systems-2025.5.1/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting training with Nsys profiling..."

nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=llama_3.1_8b_trace \
    --force-overwrite=true \
    --stats=true \
    deepspeed --num_gpus=4 train_local.py

echo "Trace collection finished."
