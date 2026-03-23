# Fully Sharded Data Parallel (FSDP) on TPU v6e-4 (4 chips)

## 1. Setup

Assume:

- You are on a v6e VM with 4 TPU chips.
- TorchPrime is already cloned and installed in a virtual environment (2.8.0 worked best for us):

```bash
cd ~/torchprime
pip install -U setuptools==69.5.1
pip install -e '.[dev]'
```

## 2. BaseTrainer (/torchprime/torch_xla_models/trainer/base_trainer.py)

Does not require changes to base_trainer.py

## 3. Config (/torchprime/torch_xla_models/configs/default.yaml)

```yaml
ici_mesh:
  data: 1
  fsdp: 4
  tensor: 1 
  expert: 1
  context: 1
```

## 4. Run Command

```bash
export HF_TOKEN=...

python3 torchprime/torch_xla_models/train.py \
  model=llama-3.1-8b \
  task=train \
  task.max_steps=20 \
  profile_start_step=3 \
  profile_end_step=13
```

## 5. Trace Collection

Zip directory generated in `/torchprime/profile/plugins/profile/`

