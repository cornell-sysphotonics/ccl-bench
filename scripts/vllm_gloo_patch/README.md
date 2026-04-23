# vLLM Gloo patch

A minimal, env-gated patch that lets vLLM's CUDA communicator fall back to
`torch.distributed` collectives instead of pynccl. This is what enables the
Gloo baseline row on the ccl-bench website.

## Why this exists

Out of the box, vLLM's `CudaCommunicator` hard-constructs a `PyNcclCommunicator`
and asserts on it inside `all_gatherv` / `broadcast` / `reduce_scatterv`.
Setting `backend="gloo"` on `torch.distributed` alone is not enough — pynccl
still gets initialized and every collective path goes through it, which needs
NCCL.

The patch makes three code paths optional:

1. `gpu_worker.py` — `init_worker_distributed_environment` now honors
   `CCL_BENCH_DIST_BACKEND` for the ProcessGroup backend string.
2. `cuda_communicator.py` — gated by `CCL_BENCH_FORCE_TORCH_DISTRIBUTED=1`:
   - skips `PyNcclCommunicator` construction
   - disables custom allreduce, symm-mem allreduce, flashinfer allreduce
   - replaces the pynccl branches in `reduce_scatter`, `reduce_scatterv`,
     `all_gatherv`, `broadcast` with `torch.distributed` equivalents

Neither env var set → zero behavior change.

## Install

```
# On Perlmutter (against the cvllm env):
bash scripts/vllm_gloo_patch/install.sh

# Or for a custom venv:
VLLM_VENV=/path/to/venv bash scripts/vllm_gloo_patch/install.sh
```

The installer writes `<file>.cclbench-orig` before modifying each file and
exits cleanly if the files are already patched. To revert:

```
for f in \
    "$VLLM_VENV"/lib/python*/site-packages/vllm/v1/worker/gpu_worker.py \
    "$VLLM_VENV"/lib/python*/site-packages/vllm/distributed/device_communicators/cuda_communicator.py; do
  cp -p "${f}.cclbench-orig" "$f"
done
```

## Use

Set both env vars before invoking `vllm bench latency` / `serve`:

```
export CCL_BENCH_DIST_BACKEND=gloo
export CCL_BENCH_FORCE_TORCH_DISTRIBUTED=1
```

Pair with `--disable-custom-all-reduce` on the CLI as an extra guardrail.

## Verify Gloo is actually engaged

1. In the vLLM log, grep for `backend=gloo` in the
   `parallel_state.py:1400] world_size=...` init line. If you see
   `backend=nccl` or the line is missing, the run is not Gloo.
2. Grep the same log for `pynccl` or `ncclCommInit` — neither should appear.
3. In the NSYS sqlite (`CUPTI_ACTIVITY_KIND_KERNEL`) there should be **no**
   `ncclDevKernel_*` kernels. Collectives will show up as CPU-side Gloo
   send/recv work plus host→device copies for the tensor payloads.

The `run_qwen3_4b_gloo_batch128_perlmutter.sh` runner calls
`verify_gloo_in_log` for checks (1)+(2) and refuses to produce a bundle if
either fails.

## Scope

The patch covers every collective vLLM's `CudaCommunicator` actually calls in
the TP inference path we benchmark:

| vLLM method        | NCCL path                     | Gloo path                                |
|--------------------|-------------------------------|------------------------------------------|
| `all_reduce`       | custom AR / pynccl / symm_mem | upstream `torch.distributed.all_reduce`  |
| `all_gather`       | pynccl                        | `torch.distributed.all_gather`           |
| `all_gatherv`      | pynccl                        | pad-to-max + `all_gather` + slice        |
| `reduce_scatter`   | pynccl                        | `reduce_scatter_tensor`                  |
| `reduce_scatterv`  | pynccl                        | pad-to-max + `reduce_scatter_tensor`     |
| `broadcast`        | pynccl                        | `torch.distributed.broadcast`            |

Not covered (would need extra work if ever touched): `send` / `recv` pairs
used by pipeline parallelism. Our TP-only Perlmutter runs don't hit that
path.
