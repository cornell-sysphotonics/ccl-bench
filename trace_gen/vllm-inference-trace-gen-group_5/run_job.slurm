#!/bin/bash
#SBATCH -J mixed-parallel-sweep
#SBATCH -A m4999                     
#SBATCH -q regular                     
#SBATCH -C gpu                         
#SBATCH -t 02:00:00                    
#SBATCH -N 1                           
#SBATCH -G 4                           
#SBATCH --ntasks-per-node=1            

echo "Running on: $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

# activate virtual environment
module load conda
conda activate straggler_infer

# run your script
~/ccl-bench-straggler-scale-up/trace_gen/vllm-inference-trace-gen-group_5/run_experiment_sweep.sh