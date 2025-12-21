#!/bin/bash
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:20:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH -A m4999
#SBATCH -J deepseek_2n8g
#SBATCH -o logs_2nodes_%j.out

module load conda
module load cudatoolkit
conda activate final_project

export HF_TOKEN="hf_xxx"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/global/homes/r/rb945/nsight-systems-2025.5.1/bin:$PATH
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO

if [ -z "$SLURM_NTASKS" ]; then
    export WORLD_SIZE=8
else
    export WORLD_SIZE=$SLURM_NTASKS
fi

echo "======================================================"
echo "Starting 2-Node (8-GPU) Training with Trace Collection"
echo "Script      : train_deepseek_2n8g.py"
echo "Master Node : $MASTER_ADDR"
echo "World Size  : $WORLD_SIZE"
echo "======================================================"

srun -n 8 -u --cpu-bind=cores ./nsys_wrapper_deepseek_2n.sh train_deepseek_2n8g.py --deepspeed ds_config.json

echo "Trace collection finished."