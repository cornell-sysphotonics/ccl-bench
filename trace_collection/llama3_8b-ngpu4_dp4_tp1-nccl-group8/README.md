Copy the config into torchtitan:
```
cp <path-to-this-dir>/llama3_8b.toml <path-to-torchtitan>/torchtitan/models/llama3/train_configs/llama3_8b.toml
```

Copy the launch script into torchtitan:
```
cp <path-to-this-dir>/run_train.sh <path-to-torchtitan>/run_train.sh
```

Set Hugging Face cache location (adjust as needed):
```
export HF_HOME=$PSCRATCH/huggingface
```

Run training from the torchtitan root:
```
cd <path-to-torchtitan> && ./run_train.sh
```
