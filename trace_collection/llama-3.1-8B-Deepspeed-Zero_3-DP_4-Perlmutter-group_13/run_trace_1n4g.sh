module load conda
conda activate final_project

export HF_HOME=$PSCRATCH/huggingface
export HF_DATASETS_CACHE=$PSCRATCH/huggingface_datasets
export HF_TOKEN="hf_xxx"
export PATH=/global/homes/r/rb945/nsight-systems-2025.5.1/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

echo "Starting training with Nsys profiling..."

nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=llama_3.1_8b_nsys_comms_logger \
    --force-overwrite=true \
    --stats=true \
    deepspeed --num_gpus=4 train_1n4g.py

echo "Trace collection finished."