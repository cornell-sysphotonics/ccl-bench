## Setup
Copy the config into torchtitan:
```
cp <path-to-this-dir>/qwen3_32b.toml <path-to-torchtitan>/torchtitan/models/qwen3/train_configs/qwen3_32b.toml
```
Copy the launch script into torchtitan:
```
cp <path-to-this-dir>/multinode_trainer.slurm <path-to-torchtitan>/multinode_trainer.slurm
```

Allocate the correct number of nodes and GPUs:
```
salloc --nodes 4 --qos interactive --time 01:00:00 --constraint gpu --gpus 16 --account m4999
module load conda
conda activate <path-to-conda-env>
export HF_HOME=$PSCRATCH/huggingface
cd <path-to-torchtitan>
```

## Run
Run training from the torchtitan root:
```
bash multinode_trainer.slurm
```

## Output
Profile traces will be written to:
```
<path-to-torchtitan>/outputs/profile_trace
```