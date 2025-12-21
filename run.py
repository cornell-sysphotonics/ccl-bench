#!/usr/bin/env python3
"""
Single-file TP analysis for folders:
- trace_collection/llama-8b-tp1/
- trace_collection/llama-8b-tp2/
- trace_collection/llama-8b-tp4/

Each folder should contain:
- timing_stats_rank0.json (fallback: timing_stats_0.json)

We use timing stats to compute:
- TTFT (avg / p99)
- TPOT (avg / p99)
- Iteration time
- Step rate (steps/sec, derived from iteration time; tokens per step unknown)

All plots + summary CSV are written into one folder and its path is printed.
"""

import json
import os
from typing import Dict, List

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TP_DIRS = {
    1: "trace_collection/llama-8b-tp1",
    2: "trace_collection/llama-8b-tp2",
    4: "trace_collection/llama-8b-tp4",
}


def load_json(path: str):
    with open(path, "r") as fh:
        return json.load(fh)


def timing_path(base: str) -> str:
    for name in ("timing_stats_rank0.json", "timing_stats_0.json"):
        candidate = os.path.join(base, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No timing stats found in {base}")


def load_timing(base: str) -> Dict[str, float]:
    data = load_json(timing_path(base))
    def _num(val):
        return val if isinstance(val, (int, float)) else 0.0
    return {
        "ttft_avg": _num(data.get("ttft_avg", 0.0)),  # seconds
        "ttft_p99": _num(data.get("ttft_p99", 0.0)),
        "tpot_avg": _num(data.get("tpot_avg", 0.0)),
        "tpot_p99": _num(data.get("tpot_p99", 0.0)),
        "iter_time": _num(data.get("avg_iteration_time", 0.0)),  # seconds
    }


def build_metrics():
    """Collect basic timing-derived metrics from trace directories."""
    rows: List[Dict] = []
    for tp, base in TP_DIRS.items():
        t = load_timing(base)
        iter_time_sec = t["iter_time"]

        rows.append(
            {
                "tp": tp,
                "ttft_avg_ms": t["ttft_avg"] * 1000.0,
                "ttft_p99_ms": t["ttft_p99"] * 1000.0,
                "tpot_avg_ms": t["tpot_avg"] * 1000.0,
                "tpot_p99_ms": t["tpot_p99"] * 1000.0,
                "iter_time_ms": iter_time_sec * 1000.0,
                "iter_time_sec": iter_time_sec,
                # This is iterations per second; tokens per step are unknown in this summary.
                "steps_per_sec": 1.0 / iter_time_sec if iter_time_sec > 0 else 0.0,
            }
        )

    df = pd.DataFrame(rows).sort_values("tp").reset_index(drop=True)

    return df


def plot_latency(df, out_dir):
    tps = df["tp"].tolist()
    x = range(len(tps))
    width = 0.2
    plt.figure(figsize=(9, 5))
    plt.bar([i - 1.5 * width for i in x], df["ttft_avg_ms"], width, label="TTFT avg")
    plt.bar([i - 0.5 * width for i in x], df["ttft_p99_ms"], width, label="TTFT p99")
    plt.bar([i + 0.5 * width for i in x], df["tpot_avg_ms"], width, label="TPOT avg")
    plt.bar([i + 1.5 * width for i in x], df["tpot_p99_ms"], width, label="TPOT p99")
    plt.xticks(x, tps)
    plt.ylabel("Latency (ms)")
    plt.xlabel("Tensor Parallel (TP)")
    plt.title("TTFT / TPOT vs TP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latency_vs_tp.png"), dpi=300)
    plt.close()


def plot_iter_time(df, out_dir):
    tps = df["tp"].tolist()
    plt.figure(figsize=(7, 5))
    plt.plot(tps, df["iter_time_ms"], marker="o", label="Measured")
    plt.xlabel("Tensor Parallel (TP)")
    plt.ylabel("Iteration Time (ms)")
    plt.title("Iteration Time Scaling")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iteration_time_scaling.png"), dpi=300)
    plt.close()


def plot_throughput(df, out_dir):
    tps = df["tp"].tolist()
    plt.figure(figsize=(7, 5))
    plt.plot(tps, df["steps_per_sec"], marker="o", label="Step rate (Hz)")
    plt.xlabel("Tensor Parallel (TP)")
    plt.ylabel("Steps per second")
    plt.title("Step Rate vs TP (derived from iteration time)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "throughput_vs_tp.png"), dpi=300)
    plt.close()


def main():
    out_dir = "tp_analysis_output"
    os.makedirs(out_dir, exist_ok=True)

    df = build_metrics()

    # 保存 summary 表
    df.to_csv(os.path.join(out_dir, "tp_metrics_summary.csv"), index=False)

    # 画图
    plot_latency(df, out_dir)
    plot_iter_time(df, out_dir)
    plot_throughput(df, out_dir)

    print(f"Outputs saved to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
