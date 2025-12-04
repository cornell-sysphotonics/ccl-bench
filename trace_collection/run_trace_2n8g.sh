#!/bin/bash
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:40:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH -A m4999
#SBATCH -J llama_2n8g
#SBATCH -o logs_2nodes_%j.out

module load conda
module load cudatoolkit
conda activate /pscratch/sd/q/qiaox226/ccl-bench/envs/final_project

export HF_TOKEN="你的HF_TOKEN"  # 如果模型是本地的，可以不需要
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/global/homes/q/qiaox226/nsight-systems-2025.5.1/bin:$PATH
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

if [ -z "$SLURM_NTASKS" ]; then
    export WORLD_SIZE=8
else
    export WORLD_SIZE=$SLURM_NTASKS
fi

echo "======================================================"
echo "Starting 2-Node (8-GPU) Training with Trace Collection"
echo "Script      : train_2n8g.py"
echo "Master Node : $MASTER_ADDR"
echo "World Size  : $WORLD_SIZE"
echo "======================================================"

srun -n 8 -u --cpu-bind=cores ./nsys_wrapper_2n.sh train_2n8g.py --deepspeed ds_config.json

echo "Trace collection finished."