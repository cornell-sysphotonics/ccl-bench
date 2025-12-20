## Setup
Copy the config into torchtitan:
```
cp <path-to-this-dir>/llama3_8b.toml <path-to-torchtitan>/torchtitan/models/llama3/train_configs/llama3_8b.toml
```
Copy the launch script into torchtitan:
```
cp <path-to-this-dir>/run_train.sh <path-to-torchtitan>/run_train.sh
```
Allocate the correct number of nodes and GPUs:
```
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account m4999
module load conda
conda activate <path-to-conda-env>
export HF_HOME=$PSCRATCH/huggingface
cd <path-to-torchtitan>
```

## Run
Run training from the torchtitan root:
```
./run_train.sh
```

## Output
Profile traces will be written to:
```
<path-to-torchtitan>/outputs/profile_trace
```
