#!/bin/bash
set -euo pipefail

# ==========================================
# Interactive Nsight Systems runner (NO module load, NO conda activate)
# Assumes:
#   1) You already did salloc and are on a GPU node
#   2) You already activated the correct conda env (e.g., vllm-prof)
#
# Usage:
#   ./run_with_nsys.sh E1.1_llama8b_baseline.yaml
# ==========================================

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <CONFIG_YAML_NAME>"
  exit 1
fi

CONFIG="$1"
EXP_NAME="$(basename "$CONFIG" .yaml)"

echo "[INFO] Running Nsys profiling (interactive)"
echo "[INFO] CONFIG = ${CONFIG}"
echo "[INFO] EXP_NAME = ${EXP_NAME}"

# ---- Use the CURRENT python from your activated env ----
PY="$(python -c 'import sys; print(sys.executable)')"
echo "[INFO] Using python: ${PY}"
"${PY}" --version

# ---- Nsight Systems absolute path (your local install) ----
NSYS="/pscratch/sd/z/zc574/nsight-systems-2025.5.1/pkg/bin/nsys"
if [[ ! -x "${NSYS}" ]]; then
  echo "[ERROR] nsys not found or not executable at: ${NSYS}"
  exit 2
fi
"${NSYS}" --version

# ---- Quick sanity: confirm your file is valid under THIS python ----
"${PY}" -m py_compile vllm_profiler.py

# ---- Runtime env (optional) ----
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,COLL

mkdir -p logs
mkdir -p "trace_collection/${EXP_NAME}"

# ---- Profile (NO nccl trace flag; not supported in your nsys build) ----
srun -n 1 "${NSYS}" profile \
  --force-overwrite=true \
  -o "trace_collection/${EXP_NAME}/nsys_report" \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=none \
  --cpuctxsw=none \
  "${PY}" vllm_profiler.py \
    --config "experiments/configs/${CONFIG}" \
  2>&1 | tee "logs/nsys_${EXP_NAME}.log"

# ---- Locate report ----
NSYS_REP="$(ls trace_collection/${EXP_NAME}/nsys_report.*rep 2>/dev/null | head -n 1 || true)"
if [[ -z "${NSYS_REP}" ]]; then
  echo "[ERROR] Nsys report not found under trace_collection/${EXP_NAME}/ (expected nsys_report.*rep)"
  echo "[HINT] Check: logs/nsys_${EXP_NAME}.log"
  exit 3
fi
echo "[INFO] Found Nsys report: ${NSYS_REP}"

# ---- Export CSVs ----
"${NSYS}" stats --force-overwrite=true --report cuda_gpu_trace --format csv \
  -o "trace_collection/${EXP_NAME}/cuda_gpu_trace" \
  "${NSYS_REP}"

"${NSYS}" stats --force-overwrite=true --report gpukernsum --format csv \
  -o "trace_collection/${EXP_NAME}/gpukernsum" \
  "${NSYS_REP}"

echo "[INFO] Done."
echo "[INFO] Outputs:"
echo "  - ${NSYS_REP}"
echo "  - trace_collection/${EXP_NAME}/cuda_gpu_trace.csv"
echo "  - trace_collection/${EXP_NAME}/gpukernsum.csv"
echo "  - logs/nsys_${EXP_NAME}.log"
