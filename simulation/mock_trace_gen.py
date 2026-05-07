#!/usr/bin/env python3
"""
Synthetic rankN_trace.json generator for CCL-Bench simulation dry runs.

Simulates one Transformer training step's communication pattern on a
configurable GPU setup.  Output traces use the same kineto Chrome JSON
format that pipeline.py --mode comm-only expects, so they work with both
mock_pipeline.py (no Docker) and the real pipeline (with Docker + AstraSim).

Communication pattern generated:
  - Per layer: 2 × TP AllReduce (post-attention + post-MLP)
  - After all layers: DP ReduceScatter + DP AllGather (gradient sync)

Exit codes:
  0 — success, traces written to --output-dir
  1 — invalid configuration

Usage:
  python simulation/mock_trace_gen.py --output-dir /tmp/mock_trace [options]
"""

import argparse
import json
import random
from pathlib import Path


DTYPE_BYTES = {"BFloat16": 2, "Float16": 2, "Float32": 4}

NCCL_KERNEL_NAMES = {
    "all_reduce":      "ncclDevKernel_AllReduceRing_BFloat16_Sum_RING_LL",
    "all_gather":      "ncclDevKernel_AllGatherRing_RING_LL",
    "reduce_scatter":  "ncclDevKernel_ReduceScatterRing_BFloat16_Sum_RING_LL",
    "all_to_all":      "ncclDevKernel_SendRecv",
}

# Bandwidth assumptions used for generating realistic event durations.
# The mock event duration is meant to represent measured GPU kernel time so that
# compute gaps in the trace look realistic — not the simulated time.
_REFERENCE_INTRA_BW_GBps = 400.0   # NVLink-class


def _collective_duration_us(coll_type: str, size_bytes: int, n: int,
                             bw_gbps: float) -> int:
    """Approximate measured NCCL kernel duration in microseconds (for trace realism)."""
    bw_bytes_per_us = bw_gbps * 1e3   # GB/s → MB/us → bytes/us × 1e6 / 1e6
    if coll_type in ("all_reduce",):
        t = 2.0 * (n - 1) / n * size_bytes / bw_bytes_per_us
    else:
        t = (n - 1) / n * size_bytes / bw_bytes_per_us
    return max(10, int(t))


def generate_rank_trace(
    rank: int,
    tp_ranks: list[int],
    dp_ranks: list[int],
    num_layers: int,
    hidden_size: int,
    seq_len: int,
    batch_size: int,
    num_params: int,
    dtype: str,
    rng: random.Random,
) -> list[dict]:
    """Build kineto Chrome JSON events for one rank."""
    dtype_bytes = DTYPE_BYTES[dtype]
    events: list[dict] = []
    ts = 0  # current timestamp in microseconds

    pg_tp = str(sorted(tp_ranks))
    pg_dp = str(sorted(dp_ranks))

    # TP AllReduce size: activations passed through TP split
    tp_allreduce_bytes = hidden_size * seq_len * batch_size * dtype_bytes
    # DP gradient size split across TP (model is sharded along TP)
    dp_tp_degree = len(tp_ranks)
    dp_grad_bytes = num_params * dtype_bytes // max(1, dp_tp_degree)

    def _add_compute(duration_us: int):
        nonlocal ts
        events.append({
            "ph": "X", "cat": "kernel",
            "name": "aten::mm",
            "pid": rank, "tid": 7,
            "ts": ts, "dur": duration_us,
            "args": {},
        })
        ts += duration_us

    def _add_collective(coll_type: str, size_bytes: int, pg_ranks: str,
                        pg_bw: float):
        nonlocal ts
        dur = _collective_duration_us(coll_type, size_bytes, len(eval(pg_ranks)), pg_bw)
        events.append({
            "ph": "X", "cat": "kernel",
            "name": NCCL_KERNEL_NAMES.get(coll_type, f"ncclDevKernel_{coll_type}"),
            "pid": rank, "tid": 7,
            "ts": ts, "dur": dur,
            "args": {
                "Collective name": coll_type,
                "In msg nelems": size_bytes // dtype_bytes,
                "Out msg nelems": size_bytes // dtype_bytes,
                "Data Type": dtype,
                "Process Group Ranks": pg_ranks,
            },
        })
        ts += dur

    # Wrap everything in a ProfilerStep so avg_step_time can parse the trace too
    step_start = ts

    for layer_idx in range(num_layers):
        # Attention compute
        attn_compute_us = rng.randint(800, 1200)
        _add_compute(attn_compute_us)

        # TP AllReduce — post-attention
        _add_collective("all_reduce", tp_allreduce_bytes, pg_tp, _REFERENCE_INTRA_BW_GBps)

        # MLP compute
        mlp_compute_us = rng.randint(1200, 1800)
        _add_compute(mlp_compute_us)

        # TP AllReduce — post-MLP
        _add_collective("all_reduce", tp_allreduce_bytes, pg_tp, _REFERENCE_INTRA_BW_GBps)

    # Gradient sync (DP group)
    dp_bw = _REFERENCE_INTRA_BW_GBps if len(dp_ranks) == 1 else 25.0
    _add_collective("reduce_scatter", dp_grad_bytes, pg_dp, dp_bw)
    _add_collective("all_gather",     dp_grad_bytes, pg_dp, dp_bw)

    # Wrap in ProfilerStep user annotation
    step_dur = ts - step_start
    events.append({
        "ph": "X", "cat": "user_annotation",
        "name": "ProfilerStep#0",
        "pid": rank, "tid": 0,
        "ts": step_start, "dur": step_dur,
        "args": {},
    })

    return events


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic rankN_trace.json for CCL-Bench simulation dry runs",
    )
    parser.add_argument("--output-dir", default="/tmp/mock_trace",
                        help="Directory for output rankN_trace.json files")
    parser.add_argument("--tp", type=int, default=4,
                        help="Tensor parallelism degree (default: 4)")
    parser.add_argument("--dp", type=int, default=2,
                        help="Data parallelism degree (default: 2)")
    parser.add_argument("--layers", type=int, default=32,
                        help="Number of Transformer layers (default: 32)")
    parser.add_argument("--hidden-size", type=int, default=4096,
                        help="Transformer hidden dimension (default: 4096, Llama-3.1-8B)")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Sequence length (default: 1024)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Micro-batch size (default: 4)")
    parser.add_argument("--num-params", type=int, default=8_030_261_248,
                        help="Total model parameters (default: 8.03B, Llama-3.1-8B)")
    parser.add_argument("--dtype", default="BFloat16",
                        choices=list(DTYPE_BYTES),
                        help="Activation/gradient dtype (default: BFloat16)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for compute gap jitter (default: 42)")
    args = parser.parse_args()

    total_ranks = args.tp * args.dp
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    print(f"[mock_trace_gen] Generating {total_ranks}-rank trace "
          f"(tp={args.tp}, dp={args.dp}, layers={args.layers}, "
          f"hidden={args.hidden_size}, seq={args.seq_len}, batch={args.batch_size})")

    for rank in range(total_ranks):
        # TP group: ranks sharing the same DP replica
        tp_group_idx = rank // args.tp
        tp_ranks = list(range(tp_group_idx * args.tp, (tp_group_idx + 1) * args.tp))
        # DP group: same TP-local position across DP replicas
        tp_local = rank % args.tp
        dp_ranks = [tp_local + g * args.tp for g in range(args.dp)]

        events = generate_rank_trace(
            rank=rank,
            tp_ranks=tp_ranks,
            dp_ranks=dp_ranks,
            num_layers=args.layers,
            hidden_size=args.hidden_size,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_params=args.num_params,
            dtype=args.dtype,
            rng=rng,
        )

        trace_path = out / f"rank{rank}_trace.json"
        with open(trace_path, "w") as f:
            json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, f)

    print(f"[mock_trace_gen] Wrote {total_ranks} trace file(s) to {out}")
    print(f"[mock_trace_gen] Use with:")
    print(f"    python simulation/mock_pipeline.py --trace-dir {out}")
    print(f"    # or (with Docker): python simulation/pipeline.py --mode comm-only --trace-dir {out}")


if __name__ == "__main__":
    main()
