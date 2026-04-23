# Gloo and SGLang Perlmutter Collection Status

This note records the Gloo and SGLang follow-up collection for the
Perlmutter A100 inference rows.

## Summary

The collected artifacts were copied to the website host:

```text
lambda7:/data/ccl-bench_trace_collection
```

The dashboard config in `~/ccl-bench-main` was updated and regenerated to 79
rows. The new rows are:

### vLLM + Gloo

Collected with the same model family, TP=4, input length 1024, output length
128, and batch sizes 8/128 used by the existing NCCL / MSCCL++ comparisons.

```text
qwen3-4b-vllm-tp4-batch8-gloo-perlmutter
qwen3-4b-vllm-tp4-batch128-gloo-perlmutter
llama-3.1-8b-vllm-tp4-batch8-gloo-perlmutter
llama-3.1-8b-vllm-tp4-batch128-gloo-perlmutter
deepseek-moe-16b-vllm-tp4-ep4-batch8-gloo-perlmutter
deepseek-moe-16b-vllm-tp4-ep4-batch128-gloo-perlmutter
```

Gloo was forced by patching vLLM to route collectives through
`torch.distributed` instead of PyNCCL/custom CUDA collectives. The run scripts
verify `backend=gloo` appears in the vLLM worker logs and fail if NCCL/PyNCCL
initialization is detected.

Website metric snapshot after regeneration:

| Row | Step Time | MFU | Communication Fraction | Total Communication Time |
|---|---:|---:|---:|---:|
| qwen3-4b-vllm-tp4-batch8-gloo-perlmutter | 0.128875 | 0.4628 | 0.0 | 0.0 |
| qwen3-4b-vllm-tp4-batch128-gloo-perlmutter | 0.455548 | 1.4541 | 0.0 | 0.0 |
| llama-3.1-8b-vllm-tp4-batch8-gloo-perlmutter | 0.120760 | 0.8403 | 0.0 | 0.0 |
| llama-3.1-8b-vllm-tp4-batch128-gloo-perlmutter | 0.592039 | 2.2729 | 0.0 | 0.0 |
| deepseek-moe-16b-vllm-tp4-ep4-batch8-gloo-perlmutter | 0.135681 | 0.2772 | 0.0 | 0.0 |
| deepseek-moe-16b-vllm-tp4-ep4-batch128-gloo-perlmutter | 0.347328 | 1.4226 | 0.0 | 0.0 |

Note: Gloo communication appears as CPU/TCP collectives rather than NCCL GPU
kernels. The current communication metrics are kernel-name based, so the Gloo
communication fraction/time fields show `0.0`. Treat Step Time and MFU as the
primary comparable values for these rows unless the communication metric is
extended to recognize Gloo events.

### SGLang + NCCL

Collected with SGLang bare-metal Conda environment on Perlmutter:

```text
/pscratch/sd/k/kg597/session1/csglang
```

Rows:

```text
qwen3-4b-sglang-tp4-batch8-perlmutter
qwen3-4b-sglang-tp4-batch128-perlmutter
llama-3.1-8b-sglang-tp4-batch8-perlmutter
llama-3.1-8b-sglang-tp4-batch128-perlmutter
deepseek-moe-16b-sglang-tp4-ep4-batch8-perlmutter
deepseek-moe-16b-sglang-tp4-ep4-batch128-perlmutter
```

The first-request hang was caused by SGLang/TVM FFI cache locks on the home
filesystem. The runner now moves all SGLang/FlashInfer/Triton/TorchInductor
cache paths to node-local `/tmp`.

Website metric snapshot after regeneration:

| Row | TTFT ms | TPOT ms | Throughput tok/s | Communication Fraction | Total Communication Time |
|---|---:|---:|---:|---:|---:|
| qwen3-4b-sglang-tp4-batch8-perlmutter | 55420.3 | -0.000306 | 18.4674 | 81.0588 | None |
| qwen3-4b-sglang-tp4-batch128-perlmutter | 56850.8 | -0.000153 | 287.859 | 55.3831 | None |
| llama-3.1-8b-sglang-tp4-batch8-perlmutter | 9183.62 | -0.000197 | 111.29 | 76.7777 | None |
| llama-3.1-8b-sglang-tp4-batch128-perlmutter | 11749.8 | -0.000146 | 1389.41 | 37.2806 | None |
| deepseek-moe-16b-sglang-tp4-ep4-batch8-perlmutter | 24600.7 | -0.000181 | 41.5959 | 79.3671 | None |
| deepseek-moe-16b-sglang-tp4-ep4-batch128-perlmutter | 12875.6 | -0.000144 | 1268.77 | 45.8575 | None |

Note: SGLang `bench_serving` reports TTFT nearly equal to E2E latency and
near-zero/negative TPOT for these fixed random-id requests. Throughput is
usable, and the NSYS SQLite traces are available for kernel metrics. Total
communication time currently returns `None` for SGLang rows because the NSYS
helper path does not handle these SGLang traces cleanly.

## Files Updated

- `website/benchmark_config.json`
- `website/benchmark_data.json`
- `website/data.js`
- `tools/main.py`
- `tools/ttft_group_6/ttft_group_6.py`
- `tools/tpot_group_6/tpot_group_6.py`
- Gloo/SGLang collection scripts under `scripts/`
- YAML workload cards under `trace_collection/`

The large `.sqlite`, `.nsys-rep`, and runtime logs are stored under
`/data/ccl-bench_trace_collection` and should not be committed.

## SGLang + MSCCL++ Follow-up

After the SGLang NCCL rows were working, the same six SGLang workloads were
collected with the MSCCL++ NCCL shim loaded directly through SGLang:

```text
SGLANG_NCCL_SO_PATH=/global/homes/k/kg597/mscclpp/build/lib/libmscclpp_nccl.so
LD_PRELOAD=/global/homes/k/kg597/mscclpp/build/lib/libmscclpp_nccl.so
COMM_BACKEND=pure_mscclpp
```

Rows:

```text
qwen3-4b-sglang-tp4-batch8-puremscclpp-perlmutter
qwen3-4b-sglang-tp4-batch128-puremscclpp-perlmutter
llama-3.1-8b-sglang-tp4-batch8-puremscclpp-perlmutter
llama-3.1-8b-sglang-tp4-batch128-puremscclpp-perlmutter
deepseek-moe-16b-sglang-tp4-ep4-batch8-puremscclpp-perlmutter
deepseek-moe-16b-sglang-tp4-ep4-batch128-puremscclpp-perlmutter
```

The server log verified that SGLang loaded the MSCCL++ shim:

```text
Found nccl from environment variable SGLANG_NCCL_SO_PATH=/global/homes/k/kg597/mscclpp/build/lib/libmscclpp_nccl.so
```

The first completed SQLite was checked directly and contained MSCCL++ kernels,
including:

```text
mscclpp::collective::allreducePacket
mscclpp::collective::allreduceFullmesh
mscclpp::collective::allgatherFullmesh2
```

The bundles were copied to:

```text
lambda7:/data/ccl-bench_trace_collection
```

The website was regenerated to 85 rows.

Metric snapshot after regeneration:

| Row | TTFT ms | TPOT ms | Throughput tok/s | Comm Fraction | Total Comm Time |
|---|---:|---:|---:|---:|---:|
| qwen3-4b-sglang-tp4-batch8-puremscclpp-perlmutter | 59947.1 | -0.000303 | 17.078 | 77.5756 | None |
| qwen3-4b-sglang-tp4-batch128-puremscclpp-perlmutter | 12307.8 | -0.000228 | 1326.63 | 37.6991 | None |
| llama-3.1-8b-sglang-tp4-batch8-puremscclpp-perlmutter | 9547.66 | -0.000174 | 107.095 | 67.8116 | None |
| llama-3.1-8b-sglang-tp4-batch128-puremscclpp-perlmutter | 11950.2 | -0.000147 | 1365.57 | 30.4693 | None |
| deepseek-moe-16b-sglang-tp4-ep4-batch8-puremscclpp-perlmutter | 24062.3 | -0.000172 | 42.5243 | 64.8761 | None |
| deepseek-moe-16b-sglang-tp4-ep4-batch128-puremscclpp-perlmutter | 12694.9 | -0.000145 | 1286.67 | 30.0816 | None |

The same SGLang caveat applies: TPOT is near zero/negative because of how
`bench_serving` reports fixed random-id requests, and total communication time
is still `None` for SGLang traces because the current NSYS helper path does not
handle these traces cleanly.
