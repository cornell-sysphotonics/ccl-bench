#!/usr/bin/env bash
# Submit from the CCL-Bench repository root:
#   sbatch trace_collection/deepseek_v3_16b_ep8_dp2_tp4_w8_batch8_seq1024-nccl-eos-h100/run.sbatch

#SBATCH --job-name=nvr_nrg_misc-cclbench.ds16b-ep8-h100
#SBATCH --partition=batch
#SBATCH --account=nvr_nrg_misc
#SBATCH --constraint=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsw/nvr_nrg_misc/erding/ccl-bench/slurm-%x-%j.out
#SBATCH --error=/lustre/fsw/nvr_nrg_misc/erding/ccl-bench/slurm-%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit this file with sbatch." >&2
  exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_DIR="$REPO_ROOT/trace_collection/deepseek_v3_16b_ep8_dp2_tp4_w8_batch8_seq1024-nccl-eos-h100"
TORCHTITAN_ROOT="$REPO_ROOT/third_party/torchtitan"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/nvr_nrg_misc/erding/ccl-bench}"
OUTPUT_DIR="${OUTPUT_DIR:-$ARTIFACT_ROOT/deepseek_v3_16b_ep8_dp2_tp4_w8_batch8_seq1024-nccl-eos-h100-$SLURM_JOB_ID}"
CACHE_ROOT="${CACHE_ROOT:-$ARTIFACT_ROOT/.cache}"
PYTHON_DEPS="${PYTHON_DEPS:-$CACHE_ROOT/torchtitan-4a57715-nodeps}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:26.04-py3}"

mapfile -t nodes < <(scontrol show hostnames "$SLURM_NODELIST")
if (( ${#nodes[@]} != 2 )); then
  echo "Expected exactly two nodes; got ${#nodes[@]}" >&2
  exit 1
fi
MASTER_ADDR="${nodes[0]}"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

mkdir -p "$ARTIFACT_ROOT" "$OUTPUT_DIR" "$CACHE_ROOT" "$PYTHON_DEPS"
cp "$RUN_DIR/deepseek_v3_16b_ep8_dp2_tp4_w8_batch8_seq1024-nccl-eos-h100.yaml" \
  "$OUTPUT_DIR/"

# Install only TorchTitan's top-level Python dependencies into a persistent
# cache. --no-deps is essential: PyTorch/CUDA/NCCL must remain those supplied
# by the date-matched NVIDIA PyTorch container, not a newer PyPI torch wheel.
if [[ ! -f "$PYTHON_DEPS/.complete" ]]; then
  srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$REPO_ROOT:$REPO_ROOT,$ARTIFACT_ROOT:$ARTIFACT_ROOT" \
    --container-workdir="$TORCHTITAN_ROOT" \
    bash -lc \
      "python3 -m pip install --no-deps --target '$PYTHON_DEPS' -r '$TORCHTITAN_ROOT/requirements.txt' && touch '$PYTHON_DEPS/.complete'"
fi

export MASTER_ADDR MASTER_PORT OUTPUT_DIR PYTHON_DEPS REPO_ROOT TORCHTITAN_ROOT

echo "job=$SLURM_JOB_ID nodes=${nodes[*]} output=$OUTPUT_DIR"
echo "container=$CONTAINER_IMAGE master=$MASTER_ADDR:$MASTER_PORT"

srun \
  --nodes=2 \
  --ntasks=2 \
  --ntasks-per-node=1 \
  --cpus-per-task=64 \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$REPO_ROOT:$REPO_ROOT,$ARTIFACT_ROOT:$ARTIFACT_ROOT" \
  --container-workdir="$TORCHTITAN_ROOT" \
  --container-env=MASTER_ADDR,MASTER_PORT,OUTPUT_DIR,PYTHON_DEPS,REPO_ROOT,TORCHTITAN_ROOT \
  bash "$RUN_DIR/run.sh"

python3 "$RUN_DIR/compare_metrics.py" \
  --repo "$REPO_ROOT" \
  --hopper-trace "$OUTPUT_DIR" \
  --output "$OUTPUT_DIR/a100-vs-h100-metrics.md"

echo "comparison=$OUTPUT_DIR/a100-vs-h100-metrics.md"
