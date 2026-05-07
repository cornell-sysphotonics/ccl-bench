#!/usr/bin/env python3
"""
Synthetic trace generator for CCL-Search dry runs.

Reads parallelism config from environment variables (injected by execute.py),
computes a simulated step time using a parametric model of Llama-3.1-8B on a
4-GPU single-node setup, and writes a minimal kineto-format JSON trace so the
real avg_step_time metric tool can parse it without any GPU or cluster.

Exit codes:
  0 — success, trace written to TRACE_DIR
  1 — simulated OOM (config exceeds 40 GB per GPU)
"""

import json
import os
import random
import sys
from pathlib import Path

import yaml


def main() -> None:
    trace_dir = Path(os.environ.get("TRACE_DIR", "/tmp/ccl-bench-dry-run"))

    tp           = int(os.environ.get("TP", 1))
    dp           = int(os.environ.get("DP", 1))
    pp           = int(os.environ.get("PP", 1))
    micro_batch  = int(os.environ.get("MICRO_BATCH", 1))
    act_ckpt     = os.environ.get("ACTIVATION_CHECKPOINTING", "false").lower() in ("true", "1")

    # ── Memory check ──────────────────────────────────────────────────────────
    # Llama-3.1-8B: ~16 GB bf16 params + ~64 GB fp32 Adam states = 80 GB total.
    # TP and PP each shard the model, so per-GPU footprint = 80 / (tp * pp) GB.
    param_mem_gb = 80.0 / (tp * pp)
    if param_mem_gb > 40.0:
        print(
            f"[mock] OOM: {param_mem_gb:.1f} GB > 40 GB per GPU "
            f"(tp={tp}, pp={pp}). Increase tp or pp.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Parametric step-time model ────────────────────────────────────────────
    # Llama-3.1-8B, batch=4, seq=512.  Base compute on one GPU ≈ 1.1 s.
    #
    # tp_compute : TP splits each matmul across tp GPUs → linear speedup.
    # tp_comm    : one AllReduce per transformer layer per TP group; cost grows
    #              with tp because more ranks must synchronise.
    # dp_grad    : gradient AllReduce across dp replicas after every step.
    # pp_bubble  : pipeline idle time = (pp-1)/num_microbatches × stage_time.
    # recompute  : activation checkpointing re-runs the forward pass (~10 %).
    base_s       = 1.1
    tp_compute_s = base_s / tp
    tp_comm_s    = tp * 0.008
    dp_grad_s    = 0.04 * (1.0 - 1.0 / dp) if dp > 1 else 0.0
    pp_bubble_s  = (pp - 1) * 0.08 / max(micro_batch, 1)
    recompute_s  = 0.10 if act_ckpt else 0.0

    step_s = tp_compute_s + tp_comm_s + dp_grad_s + pp_bubble_s + recompute_s

    # ±8 % noise — seeded on config so reruns of the same config are stable.
    rng = random.Random(tp * 1000 + dp * 100 + pp * 10 + micro_batch)
    step_s *= rng.uniform(0.92, 1.08)

    # ── Write kineto JSON trace ───────────────────────────────────────────────
    # avg_step_time looks for ProfilerStep#N user-annotation events (ph="X",
    # cat="user_annotation") and averages the inner steps (drops first & last).
    # Seven steps gives five inner steps after trimming.
    trace_dir.mkdir(parents=True, exist_ok=True)

    step_us = int(step_s * 1e6)
    events, ts = [], 0
    for i in range(7):
        events.append({
            "ph": "X", "cat": "user_annotation",
            "name": f"ProfilerStep#{i}",
            "pid": 0, "tid": 0,
            "ts": ts, "dur": step_us,
        })
        ts += step_us + max(1000, step_us // 20)  # small inter-step gap

    with open(trace_dir / "kineto_trace_0.json", "w") as f:
        json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, f)

    # Minimal workload card so avg_step_time dispatches to the json backend.
    with open(trace_dir / "workload_card.yaml", "w") as f:
        yaml.dump(
            {"metric_source": {"traces": ["json"]},
             "workload": {"model": {"phase": "training"}}},
            f,
        )

    print(
        f"[mock] tp={tp} dp={dp} pp={pp} micro_batch={micro_batch} "
        f"act_ckpt={act_ckpt} → step_time={step_s:.3f}s  ({trace_dir})"
    )


if __name__ == "__main__":
    main()
