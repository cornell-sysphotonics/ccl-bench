# SGLang Debugging and Integration Summary on Perlmutter

This document summarizes the steps taken, issues encountered, and the current state of integrating SGLang into the CCL-Bench framework on Perlmutter for standard NCCL, Gloo, and MSCCL++ communication backends.

## 1. Environment Setup & Container Bypass
Initially, attempting to run SGLang via Shifter containers resulted in hangs. We bypassed this by installing SGLang directly into a bare-metal Conda environment on Perlmutter:
`/pscratch/sd/k/kg597/session1/csglang`

## 2. Server Startup Issues
### Piecewise CUDA Graph Capture
By default, SGLang attempts to capture piecewise CUDA graphs during initialization. This process was taking an excessive amount of time or hanging.
**Fix:** Added the `--disable-piecewise-cuda-graph` flag to the server launch arguments (in addition to `--disable-cuda-graph`) inside `scripts/run_sglang_inference_perlmutter.sh`.

### FlashInfer `flock` Hang on NFS
Even after the server reported "The server is fired up and ready to roll!", sending any inference request (via `bench_serving` or a manual `curl`) would hang indefinitely with 0% GPU utilization.
**Diagnosis:** Strace revealed that the SGLang worker processes were stuck in an infinite loop trying to acquire a file lock (`flock`) for the FlashInfer cache in `~/.cache/flashinfer/`. Because the home directory on Perlmutter is mounted on a clustered filesystem (Lustre/NFS) that does not support `flock`, the kernel returned `ENOTSUPP` (Unknown error 524), causing FlashInfer to endlessly retry.
**Fix:** Added the following environment variable to `scripts/run_sglang_inference_perlmutter.sh` to force FlashInfer to use the local `/tmp` filesystem:
```bash
export FLASHINFER_WORKSPACE_BASE="/tmp/flashinfer_cache"
```

### TVM FFI / SGL Kernel JIT `flock` Hang on NFS
After the FlashInfer cache was moved, SGLang still hung on the first request. A minimal TP=1 direct `/generate` test showed the same behavior, so the issue was not TP, NCCL, or `bench_serving`.

**Diagnosis:** `strace` on the scheduler process showed repeated failed locks under:

```text
/global/homes/k/kg597/.cache/tvm-ffi/sgl_kernel_jit_*/lock
flock(..., LOCK_EX|LOCK_NB) = -1 ENOTSUPP
```

The relevant installed code is `tvm_ffi/cpp/extension.py`, which defaults to `~/.cache/tvm-ffi` unless `TVM_FFI_CACHE_DIR` is set.

**Fix:** The SGLang scripts now create a job-local node-local cache root and export:

```bash
JOB_CACHE_BASE="${JOB_CACHE_BASE:-/tmp/sglang_${SLURM_JOB_ID:-manual}_$$}"
export XDG_CACHE_HOME="$JOB_CACHE_BASE/xdg"
export FLASHINFER_WORKSPACE_BASE="$JOB_CACHE_BASE/flashinfer"
export TVM_FFI_CACHE_DIR="$JOB_CACHE_BASE/tvm-ffi"
export TRITON_CACHE_DIR="$JOB_CACHE_BASE/triton"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE_BASE/torchinductor"
export CUDA_CACHE_PATH="$JOB_CACHE_BASE/cuda"
```

This unblocked first-request kernel compilation. Verified:

- TP=1 direct `/generate`, Qwen3-4B, 1024 input / 4 output tokens: HTTP 200.
- TP=4 direct `/generate`, Qwen3-4B, 1024 input / 4 output tokens: HTTP 200.

## 3. Client Benchmark Issues
### ShareGPT Tokenization Stall
The default dataset used by `sglang.bench_serving` is `random`. However, if the local ShareGPT dataset is missing, the script attempts to download and tokenize it. On Perlmutter compute nodes, this process was stalling.
**Fix:** Changed the client dataset arguments to bypass ShareGPT entirely and use exactly sized random token arrays:
`--dataset-name random-ids --tokenize-prompt` (with `--random-input-len 1024 --random-output-len 128`).

## 4. Communication Backend Scripts Developed
To mirror the benchmarking approach used for vLLM, the `scripts/run_sglang_inference_perlmutter.sh` script was refactored to accept a `COMM_BACKEND` environment variable. Two new top-level runner scripts were created:

### Gloo Baseline
Created `scripts/run_sglang_gloo_batch128_perlmutter.sh`.
This script forces SGLang to use PyTorch's Gloo backend instead of NCCL by:
1. Setting `COMM_BACKEND=gloo`.
2. Inside the runner, this exports `CCL_BENCH_DIST_BACKEND=gloo` and `CCL_BENCH_FORCE_TORCH_DISTRIBUTED=1` (to utilize the vLLM Gloo patch logic if applicable).
3. Passing `--disable-custom-all-reduce` to SGLang so it falls back to the standard `torch.distributed` collective calls instead of using its custom CUDA kernels or `pynccl`.

### Pure MSCCL++
Created `scripts/run_sglang_pure_mscclpp_batch128_perlmutter.sh`.
This script injects Microsoft's MSCCL++ shim over standard NCCL by:
1. Setting `COMM_BACKEND=pure_mscclpp`.
2. Inside the runner, this sets `LD_PRELOAD` to `libmscclpp_nccl.so`.
3. Exports `SGLANG_NCCL_SO_PATH="$MSCCLPP_PRELOAD"` to ensure SGLang's `pynccl_wrapper` loads the shim instead of the system NCCL library.
4. Unsets `MSCCLPP_NCCL_LIB_PATH` to prevent fallback to system NCCL.
5. Unsets `PYTORCH_CUDA_ALLOC_CONF` because MSCCL++ rejects expandable segments on A100.
6. Passes `--disable-custom-all-reduce` to ensure collectives are routed through the intercepted `torch.distributed` / `pynccl` paths.

### Bundling
Updated `scripts/make_sglang_inference_bundles_perlmutter.sh` to dynamically read `$COMM_BACKEND` and output the correct communication library metadata to the resulting YAML files.

## 5. Current State and Remaining Blockers
All scripts are written, patched, and synchronized to `~/ccl-bench/scripts/` on Perlmutter.

**Resolved:** SGLang no longer hangs on the first request after moving both FlashInfer and TVM FFI/SGL kernel JIT caches to `/tmp`.

NSYS capture initially failed because `stop_server()` only waited about 60 seconds after sending SIGINT to the `nsys profile` wrapper. The server log showed NSYS starting export and then being killed before `.sqlite` completed. The runner now:

- uses `NSYS_TRACE_TYPES="${NSYS_TRACE_TYPES:-cuda,nvtx}"` to avoid excessive OS runtime tracing;
- uses `NSYS_STOP_WAIT_SECONDS="${NSYS_STOP_WAIT_SECONDS:-420}"` so NSYS has time to export `.nsys-rep` and `.sqlite`.

Verified successful full artifact row:

```text
/pscratch/sd/k/kg597/ccl-bench-traces/sglang-inference-qwen-nsysfix/
  qwen3-4b-sglang-tp4-batch128-perlmutter/
    bench_results.jsonl
    qwen3-4b-sglang-tp4-batch128-perlmutter.nsys-rep
    qwen3-4b-sglang-tp4-batch128-perlmutter.sqlite
```

Final NCCL SGLang collection:

```text
Slurm job: 51905959 (completed)
Node: nid001000
Trace root: /pscratch/sd/k/kg597/ccl-bench-traces/sglang-inference-nccl-full
Bundle root: /pscratch/sd/k/kg597/ccl-bench-traces/sglang-inference-nccl-full-bundles
```

The final bundle root contains six SGLang NCCL rows, each with:

- `bench_results.jsonl`
- `.nsys-rep`
- `.sqlite`
- YAML workload card
- README

- `qwen3-4b-sglang-tp4-batch8-perlmutter`
- `qwen3-4b-sglang-tp4-batch128-perlmutter`
- `llama-3.1-8b-sglang-tp4-batch8-perlmutter`
- `llama-3.1-8b-sglang-tp4-batch128-perlmutter`
- `deepseek-moe-16b-sglang-tp4-ep4-batch8-perlmutter`
- `deepseek-moe-16b-sglang-tp4-ep4-batch128-perlmutter`

`qwen3-4b-sglang-tp4-batch128-perlmutter` was collected first as the NSYS export proof row under:

```text
/pscratch/sd/k/kg597/ccl-bench-traces/sglang-inference-qwen-nsysfix/
  qwen3-4b-sglang-tp4-batch128-perlmutter/
```

It was copied into both the final raw trace root and the final bundle root after job 51905959 completed, so the six SGLang NCCL rows now live together.

There are no active Perlmutter Slurm jobs for this SGLang collection after completion.

Observed SGLang `bench_serving` summary values from the six NCCL rows:

| Row | Request Throughput | Output Tok/s | Mean E2E Latency ms | Mean TTFT ms | Mean TPOT ms |
|---|---:|---:|---:|---:|---:|
| qwen3-4b-sglang-tp4-batch8-perlmutter | 0.1443 | 18.47 | 55420.14 | 55420.18 | -0.0003 |
| qwen3-4b-sglang-tp4-batch128-perlmutter | 2.2489 | 287.86 | 56849.01 | 56849.03 | -0.0002 |
| llama-3.1-8b-sglang-tp4-batch8-perlmutter | 0.8695 | 111.29 | 9183.64 | 9183.67 | -0.0002 |
| llama-3.1-8b-sglang-tp4-batch128-perlmutter | 10.8547 | 1389.41 | 11749.61 | 11749.63 | -0.0002 |
| deepseek-moe-16b-sglang-tp4-ep4-batch8-perlmutter | 0.3250 | 41.60 | 24600.60 | 24600.64 | -0.0003 |
| deepseek-moe-16b-sglang-tp4-ep4-batch128-perlmutter | 9.9122 | 1268.77 | 12875.23 | 12875.25 | -0.0001 |

Note: for these fixed random-id SGLang runs, `bench_serving` reports TTFT essentially equal to E2E latency and near-zero/negative TPOT. Treat these fields carefully when wiring website metrics; the NSYS SQLite traces are still valid for kernel/communication analysis.
