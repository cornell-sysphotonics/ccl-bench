#!/usr/bin/env python3
"""Run baseline and profiler-overhead tensor-parallel LLM training trials."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["none", "kineto", "execution_trace", "both"],
        default=["none", "kineto", "execution_trace", "both"],
    )
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=29571)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--ffn-dim", type=int, default=11008)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return parser.parse_args()


def load_rank0_summary(run_dir: Path) -> dict:
    with (run_dir / "summary_rank0.json").open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    script = Path(__file__).with_name("train_tp_profiler_overhead.py")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []
    baseline_mean = None

    for index, mode in enumerate(args.modes):
        run_dir = args.out_dir / mode
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={args.nproc_per_node}",
            f"--master-port={args.master_port + index}",
            str(script),
            "--mode",
            mode,
            "--out-dir",
            str(run_dir),
            "--steps",
            str(args.steps),
            "--warmup-steps",
            str(args.warmup_steps),
            "--batch-size",
            str(args.batch_size),
            "--seq-len",
            str(args.seq_len),
            "--d-model",
            str(args.d_model),
            "--n-heads",
            str(args.n_heads),
            "--ffn-dim",
            str(args.ffn_dim),
            "--n-layers",
            str(args.n_layers),
            "--dtype",
            args.dtype,
        ]
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
        env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        print("running:", " ".join(cmd), flush=True)
        start = time.perf_counter()
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        elapsed = time.perf_counter() - start

        summary = load_rank0_summary(run_dir)
        if mode == "none":
            baseline_mean = summary["mean_step_time_s"]
        overhead_pct = 0.0
        if baseline_mean:
            overhead_pct = (summary["mean_step_time_s"] / baseline_mean - 1.0) * 100.0
        rows.append(
            {
                "mode": mode,
                "mean_step_time_s": summary["mean_step_time_s"],
                "median_step_time_s": summary["median_step_time_s"],
                "tokens_per_s": summary["tokens_per_s"],
                "peak_memory_gib_rank0": summary["peak_memory_gib"],
                "overhead_vs_none_pct": overhead_pct,
                "wall_time_s": elapsed,
            }
        )

    csv_path = args.out_dir / "overhead_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")
    for row in rows:
        print(
            f"{row['mode']:>15} mean={row['mean_step_time_s']:.4f}s "
            f"overhead={row['overhead_vs_none_pct']:.2f}% "
            f"tokens/s={row['tokens_per_s']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
