# Llama 3.1 8B Training Configuration

This directory contains the training configuration for Llama 3.1 8B model with a specific parallelism setup.

## Configuration Overview

This configuration file (`llama3_8b.toml`) defines the training parameters for distributed training of the Llama 3.1 8B model using **torchtitan**, a PyTorch native platform for large-scale LLM training.

### Parallelism Configuration

The training uses the following parallelism strategy:

- **Data Parallel Replicate Degree**: 1
- **Data Parallel Shard Degree**: -1
- **Tensor Parallel Degree**: 1
- **Pipeline Parallel Degree**: 1
- **Context Parallel Degree**: 1

**Total GPU Count**: 4 GPUs

**Effective Parallelism Strategy**:
- {PARALLELISM_STRATEGY_DESCRIPTION}

### Model Configuration

- **Model**: Llama 3.1 8B
- **HuggingFace Assets Path**: `./assets/hf/Llama-3.1-8B`
- **Sequence Length**: `512`
- **Local Batch Size**: `2`
- **Training Steps**: `200`
- **Dataset**: C4

### Training Hyperparameters

- **Optimizer**: AdamW
- **Learning Rate**: 3e-4
- **Learning Rate Scheduler**: Warmup with `50` warmup steps
- **Gradient Clipping**: Max norm = 1.0
- **Data Type**: `float32` 

### Additional Features

- **Activation Checkpointing**: {ACTIVATION_CHECKPOINT_MODE} mode
- **Compilation**: {COMPILE_STATUS}
- **Profiling**: {PROFILING_STATUS} (saves traces to `profile_trace/`)
- **TensorBoard Logging**: Enabled (saves to `tb/`)
- **Checkpointing**: {CHECKPOINT_STATUS}

## Running the Training

### Prerequisites

1. Ensure you have the required dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the HuggingFace model assets:
   ```bash
   python scripts/download_hf_assets.py
   ```

3. Set the communication backend (if using NVSHMEM):
   ```bash
   export COMM_BACKEND=nvshmem  # or "nccl" for default
   ```

### Basic Training Command

Run training using the provided script:

```bash
CONFIG_FILE="./path/to/llama3_8b.toml" NGPU={TOTAL_GPUS} ./run_train.sh
```

Or directly with `torchrun`:

```bash
torchrun --nproc_per_node={TOTAL_GPUS} \
    torchtitan/train.py \
    --job.config_file ./path/to/llama3_8b.toml
```

### Environment Variables

You can customize the run with the following environment variables:

- `NGPU`: Number of GPUs to use (default: 4)
- `COMM_BACKEND`: Communication backend - `"nccl"` or `"nvshmem"` (default: `"nccl"`)
- `LOG_RANK`: Comma-separated ranks to log (e.g., `"0,1"`)
- `DRY_RUN`: Set to `1` for configuration validation without GPU

Example with custom settings:

```bash
COMM_BACKEND=nvshmem \
LOG_RANK=0,1 \
NGPU={TOTAL_GPUS} \
CONFIG_FILE="./path/to/llama3_8b.toml" \
./run_train.sh
```

## Output Directories

After training, the following directories will be created:

- **Output Folder**: `{DUMP_FOLDER}` (as specified in `[job]` section)
- **Profile Traces**: `{DUMP_FOLDER}/profile_trace/` (if profiling enabled)
- **TensorBoard Logs**: `{DUMP_FOLDER}/tb/` (if TensorBoard enabled)
- **Checkpoints**: `{DUMP_FOLDER}/checkpoint/` (if checkpointing enabled)

## Monitoring Training

### TensorBoard

View training metrics in TensorBoard:

```bash
tensorboard --logdir {DUMP_FOLDER}/tb
```

### Log Files

Training logs will be displayed in the console. For distributed training, logs from the specified `LOG_RANK` ranks will be shown.

## Configuration Details

### Parallelism Strategy Explanation

- **Data Parallel (DP)**: Replicates the model across multiple GPUs, with each GPU processing different data batches
- **FSDP (Fully Sharded Data Parallel)**: Shards model parameters, gradients, and optimizer states across GPUs
- **HSDP (Hybrid Sharded Data Parallel)**: Combines replication and sharding for optimal memory and communication trade-offs
- **Tensor Parallel (TP)**: Splits tensor operations across GPUs, enabling training of larger models
- **Pipeline Parallel (PP)**: Splits the model layers across GPUs, processing different layers on different GPUs
- **Context Parallel (CP)**: Parallelizes attention computation across sequence length for long-context training

### Key Configuration Sections

- `[job]`: Job metadata and output directory
- `[model]`: Model architecture and checkpoint paths
- `[training]`: Training hyperparameters and dataset configuration
- `[parallelism]`: Multi-dimensional parallelism configuration
- `[optimizer]`: Optimizer settings
- `[lr_scheduler]`: Learning rate scheduling
- `[profiling]`: Profiling and tracing options
- `[metrics]`: Logging and monitoring configuration
- `[checkpoint]`: Checkpoint saving/loading settings
- `[activation_checkpoint]`: Memory optimization via activation checkpointing
- `[compile]`: PyTorch compilation settings

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `local_batch_size` or enable activation checkpointing
2. **Communication Backend Errors**: Ensure `COMM_BACKEND` matches your setup (NCCL for standard, NVSHMEM if configured)
3. **Model Assets Not Found**: Run `scripts/download_hf_assets.py` to download required model files

### Validation

Validate your configuration without running training:

```bash
DRY_RUN=1 CONFIG_FILE="./path/to/llama3_8b.toml" ./run_train.sh
```

## References

- [torchtitan Documentation](https://github.com/pytorch/torchtitan)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)

## Notes

- This configuration is optimized for {HARDWARE_INFO} GPUs
- The parallelism degrees must satisfy: `DP_REPLICATE × DP_SHARD × TP × PP × CP = Total GPUs`
- For best performance, ensure all GPUs are on the same node or connected via high-speed interconnect
