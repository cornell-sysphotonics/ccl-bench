# Tensor Parallel (TP) on TPU v6e-8 (8 chips)

## 1. Setup

Assume:

- You are on a v6e VM with 8 TPU chips.
- TorchPrime is already cloned and installed in a virtual environment (2.8.0 worked best for us):

```bash
cd ~/torchprime
pip install -U setuptools==69.5.1
pip install -e '.[dev]'
```

## 2. BaseTrainer (/torchprime/torch_xla_models/trainer/base_trainer.py)

Use modified base_trainer.py file in directory (minibatch sharding is disabled)

## 3. Config (/torchprime/torch_xla_models/configs/default.yaml)

```yaml
ici_mesh:
  data: 1
  fsdp: 1
  tensor: 8 
  expert: 1
  context: 1
```

## 4. Run Command

```bash
export HF_TOKEN=...

python3 torchprime/torch_xla_models/train.py \
  model=llama-3.1-8b \
  task=train \
  task.global_batch_size=1 \
  task.max_steps=20 \
  profile_start_step=3 \
  profile_end_step=13
```

## 5. Trace Collection

Zip directory generated in `/torchprime/profile/plugins/profile/`

