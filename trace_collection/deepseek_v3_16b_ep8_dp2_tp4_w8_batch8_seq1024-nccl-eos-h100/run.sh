#!/usr/bin/env bash
# One task per node; run.sbatch launches two copies with SLURM_PROCID 0 and 1.
# This is the Hopper rerun of the A100 WL5 EP8/DP2/TP4 training row.
set -euo pipefail

NUM_NODES=2
GPUS_PER_NODE=4
NODE_RANK="${SLURM_PROCID:?run.sh must be launched by srun}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR must be set by run.sbatch}"
MASTER_PORT="${MASTER_PORT:?MASTER_PORT must be set by run.sbatch}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set by run.sbatch}"
REPO_ROOT="${REPO_ROOT:?REPO_ROOT must be set by run.sbatch}"
TORCHTITAN_ROOT="${TORCHTITAN_ROOT:?TORCHTITAN_ROOT must be set by run.sbatch}"
RUN_DIR="$REPO_ROOT/trace_collection/deepseek_v3_16b_ep8_dp2_tp4_w8_batch8_seq1024-nccl-eos-h100"

# Eos does not register GPUs as Slurm GRES, so the allocation reserves both
# nodes exclusively and this mask preserves the A100 row's 4-GPU/node shape.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export PYTHONPATH="${PYTHON_DEPS:+$PYTHON_DEPS:}$TORCHTITAN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR/profile_traces" "$OUTPUT_DIR/logs"

mapfile -t gpu_names < <(
  nvidia-smi --query-gpu=name --format=csv,noheader | sed -n "1,${GPUS_PER_NODE}p"
)
if (( ${#gpu_names[@]} != GPUS_PER_NODE )); then
  echo "Expected $GPUS_PER_NODE GPUs on rank $NODE_RANK; found ${#gpu_names[@]}" >&2
  exit 1
fi
for gpu_name in "${gpu_names[@]}"; do
  if [[ "$gpu_name" != *H100* ]]; then
    echo "Architecture guard failed: expected H100, found '$gpu_name'" >&2
    exit 1
  fi
done

{
  echo "hostname=$(hostname)"
  echo "node_rank=$NODE_RANK"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "torchtitan_commit=4a57715f34338416d467dfc5fb54a9b69c938db5"
  python3 - <<'PY'
import torch
print(f"pytorch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"nccl={torch.cuda.nccl.version()}")
PY
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
    --format=csv,noheader
} >"$OUTPUT_DIR/environment-rank${NODE_RANK}.txt"

torch_version="$(python3 -c 'import torch; print(torch.__version__)')"
if [[ "$torch_version" != 2.12.0a0+0291f960b6.nv26.04.48445190 ]]; then
  echo "Refusing unmatched run: expected container PyTorch 2.12.0a0+0291f960b6.nv26.04.48445190, found $torch_version" >&2
  exit 1
fi

train_args=(
  --module deepseek_v3 \
  --config deepseek_v3_16b \
  --hf_assets_path "$TORCHTITAN_ROOT/tests/assets/tokenizer" \
  --dump_folder "$OUTPUT_DIR" \
  --profiling.enable_profiling \
  --profiling.save_traces_folder "$OUTPUT_DIR/profile_traces" \
  --profiling.profile_freq 4 \
  --metrics.log_freq 1 \
  --metrics.disable_color_printing \
  --dataloader.dataset c4_test \
  --optimizer.name AdamW \
  --optimizer.lr 8e-4 \
  --optimizer.eps 1e-8 \
  --lr_scheduler.warmup_steps 2 \
  --lr_scheduler.decay_ratio 0.8 \
  --lr_scheduler.decay_type linear \
  --lr_scheduler.min_lr_factor 0.0 \
  --training.local_batch_size 4 \
  --training.seq_len 1024 \
  --training.max_norm 1.0 \
  --training.steps 6 \
  --training.dtype bfloat16 \
  --parallelism.tensor_parallel_degree 4 \
  --parallelism.data_parallel_shard_degree 2 \
  --parallelism.data_parallel_replicate_degree 1 \
  --parallelism.fsdp_reshard_after_forward always \
  --parallelism.pipeline_parallel_degree 1 \
  --parallelism.pipeline_parallel_microbatch_size 1 \
  --parallelism.context_parallel_degree 1 \
  --parallelism.expert_parallel_degree 8 \
  --parallelism.expert_tensor_parallel_degree 1 \
  --checkpoint.no-enable \
  --checkpoint.interval 10 \
  --checkpoint.no-last-save-model-only \
  --activation_checkpoint.mode none \
  --compile.no-enable \
)

# Parse and assert the complete configuration before spawning eight workers.
python3 "$RUN_DIR/validate_config.py" "${train_args[@]}"

torchrun \
  --nproc_per_node="$GPUS_PER_NODE" \
  --nnodes="$NUM_NODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$TORCHTITAN_ROOT/torchtitan/train.py" \
  "${train_args[@]}" \
  2>&1 | tee "$OUTPUT_DIR/logs/node-rank-${NODE_RANK}.log"
