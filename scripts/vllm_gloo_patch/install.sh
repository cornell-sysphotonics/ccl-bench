#!/usr/bin/env bash
# Install the ccl-bench vLLM Gloo patch into a target venv.
#
# Usage:
#   scripts/vllm_gloo_patch/install.sh                  # uses VLLM_VENV env or default
#   VLLM_VENV=/path/to/venv scripts/vllm_gloo_patch/install.sh
#
# Effect:
#   - Applies scripts/vllm_gloo_patch/gpu_worker.patch to
#     <venv>/lib/python*/site-packages/vllm/v1/worker/gpu_worker.py
#   - Applies scripts/vllm_gloo_patch/cuda_communicator.patch to
#     <venv>/lib/python*/site-packages/vllm/distributed/device_communicators/cuda_communicator.py
#   - Saves each pre-patch file as <file>.cclbench-orig so the patch is
#     reversible via `cp <file>.cclbench-orig <file>`.
#
# The patch is env-var gated. With it installed but
# CCL_BENCH_FORCE_TORCH_DISTRIBUTED unset, vLLM behavior is unchanged.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_VENV="${VLLM_VENV:-/pscratch/sd/k/kg597/session1/cvllm}"

if [[ ! -d "$VLLM_VENV" ]]; then
  echo "ERROR: VLLM_VENV not found: $VLLM_VENV" >&2
  exit 1
fi

pkg_dir=$(find "$VLLM_VENV/lib" -maxdepth 3 -type d -name vllm 2>/dev/null | head -1)
if [[ -z "$pkg_dir" ]]; then
  echo "ERROR: could not locate site-packages/vllm under $VLLM_VENV/lib" >&2
  exit 1
fi

apply_one() {
  local target=$1
  local patch=$2
  if [[ ! -f "$target" ]]; then
    echo "ERROR: target file missing: $target" >&2
    exit 1
  fi
  if grep -q "CCL_BENCH_FORCE_TORCH_DISTRIBUTED\|CCL_BENCH_DIST_BACKEND" "$target"; then
    echo "  already patched: $target"
    return 0
  fi
  if [[ ! -f "${target}.cclbench-orig" ]]; then
    cp -p "$target" "${target}.cclbench-orig"
  fi
  if ! patch -p0 --batch "$target" < "$patch"; then
    # On failure, restore from backup so the venv isn't half-patched.
    cp -p "${target}.cclbench-orig" "$target"
    echo "ERROR: patch failed to apply: $patch -> $target (restored)" >&2
    exit 1
  fi
  echo "  patched: $target"
}

echo "VLLM venv: $VLLM_VENV"
echo "vllm pkg : $pkg_dir"

apply_one "$pkg_dir/v1/worker/gpu_worker.py" \
          "$SCRIPT_DIR/gpu_worker.patch"
apply_one "$pkg_dir/distributed/device_communicators/cuda_communicator.py" \
          "$SCRIPT_DIR/cuda_communicator.patch"

# Basic sanity: both files should now compile.
python -m py_compile \
  "$pkg_dir/v1/worker/gpu_worker.py" \
  "$pkg_dir/distributed/device_communicators/cuda_communicator.py"

echo "ccl-bench Gloo patch installed."
