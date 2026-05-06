# PyTorch profiler overhead on TP2 LLM training

This experiment measures PyTorch profiler overhead on a synthetic LLM training
workload using two local GPUs with tensor parallelism. It does not download a
model or dataset; token batches are generated on device so the timing is focused
on model compute, collectives, optimizer work, and profiler cost.

The trainer implements 2-way tensor parallel transformer blocks:

- attention QKV and MLP gate/up projections are column-sharded
- attention output and MLP down projections are row-sharded
- partial outputs are summed with NCCL `all_reduce`
- embeddings, layer norms, and the LM head are replicated

## Run

```bash
cd trace_gen/pytorch-profiler-overhead-tp2
./run_experiment.sh
```

By default the launcher runs four modes with `torch.distributed.run` on GPUs
`0,1`:

- `none`: baseline training without profiler
- `kineto`: `torch.profiler.profile` with CPU/CUDA activities
- `execution_trace`: PyTorch `ExecutionTraceObserver`
- `both`: Kineto plus execution trace collection

The default model is intentionally sized for two 40 GB GPUs:

```text
layers=8, d_model=4096, n_heads=32, ffn_dim=11008, seq_len=512, batch_size=2
```

For a quicker smoke test:

```bash
./run_experiment.sh --modes none kineto --steps 6 --warmup-steps 1 \
  --d-model 1024 --n-heads 16 --ffn-dim 2816 --n-layers 2
```

## Outputs

Results are written under `results/` by default.

- `overhead_summary.csv`: rank 0 mean/median step time, tokens/s, peak memory,
  and percent overhead versus `none`
- `<mode>/summary_rank{0,1}.json`: per-rank step timings and memory
- `<mode>/kineto_trace_rank{0,1}.json`: Kineto traces for profiler modes
- `<mode>/torch_et_rank{0,1}.json`: execution traces for execution-trace modes
- `<mode>/stdout.log`: full distributed run log

You can change the output location with `--out-dir`.
