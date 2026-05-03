#!/usr/bin/env python3
"""
Plot hardware efficiency analysis from metrics_summary.txt files.

Usage:
    python plot_efficiency.py /path/to/metrics_summary.txt
    python plot_efficiency.py file1.txt file2.txt file3.txt
    python plot_efficiency.py --discover /path/to/traces/profiling/
"""

import argparse
import os
import sys
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def parse_metrics(path: str) -> dict:
    result = {"path": path}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line == "---":
                continue
            # Handle multi-value lines like "tp: 4  pp: 4  dp: 1  ep: 1"
            # or "batch: 32  seq: 2048  mbs: 1"
            if line.count(":") > 1 and "  " in line:
                parts = line.split("  ")
                for part in parts:
                    part = part.strip()
                    if ":" in part:
                        k, _, v = part.partition(":")
                        k, v = k.strip(), v.strip()
                        try:
                            result[k] = float(v)
                        except ValueError:
                            result[k] = v
            elif ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
    return result


def discover_results(base_dir: str) -> list:
    pattern = os.path.join(base_dir, "**", "metrics_summary.txt")
    return sorted(glob.glob(pattern, recursive=True))


def _derive(data: dict) -> dict:
    """Compute derived metrics for plotting."""
    gpu_util = data.get("aggregate_gpu_utilization", 0)
    comm_frac = data.get("communication_fraction", 0)
    comm_overlap = data.get("communication_overlap_ratio", 0)
    mfu = data.get("mfu", 0)

    idle = 100 - gpu_util
    total_comm = gpu_util * (comm_frac / 100)
    overlapped_comm = total_comm * comm_overlap
    exposed_comm = total_comm * (1 - comm_overlap)
    compute = gpu_util - total_comm

    return {
        "compute": max(compute, 0),
        "exposed_comm": max(exposed_comm, 0),
        "overlapped_comm": max(overlapped_comm, 0),
        "idle": max(idle, 0),
        "mfu": mfu,
        "gpu_util": gpu_util,
    }


def plot_single(data: dict, output_path: str):
    model = data.get("model", "Unknown")
    gpus = data.get("total_gpus", "?")
    config = data.get("tp", "?")
    step_time = data.get("avg_step_time", 0)
    batch = int(data.get("batch", 0))
    seq = int(data.get("seq", 0))
    d = _derive(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    gridspec_kw={"width_ratios": [1, 1.1]})
    fig.suptitle(f"{model}  —  {gpus} GPUs  |  config: {config}  |  BS={batch}, seq={seq}  |  step: {step_time:.3f}s",
                 fontsize=13, fontweight="bold", y=0.97)
    fig.subplots_adjust(top=0.85, bottom=0.08, left=0.10, right=0.95, wspace=0.35)

    # --- Efficiency funnel ---
    labels = ["Peak FLOPS", "GPU active", "Compute only", "MFU"]
    values = [100, d["gpu_util"], d["compute"], d["mfu"]]
    colors = ["#B4B2A9", "#378ADD", "#3266ad", "#1D9E75"]
    bars = ax1.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.55)
    for bar, val in zip(bars, values[::-1]):
        ax1.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=10)
    ax1.set_xlim(0, 118)
    ax1.set_title("Efficiency funnel", fontsize=11, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_xlabel("%")
    ax1.tick_params(axis="y", labelsize=10)

    # --- Time breakdown pie (with overlapped comm) ---
    sizes = [d["compute"], d["exposed_comm"], d["overlapped_comm"], d["idle"]]
    pie_labels = [
        f"Compute\n{d['compute']:.1f}%",
        f"Exposed comm\n{d['exposed_comm']:.1f}%",
        f"Overlapped comm\n{d['overlapped_comm']:.1f}%",
        f"Idle\n{d['idle']:.1f}%",
    ]
    pie_colors = ["#3266ad", "#D85A30", "#1D9E75", "#B4B2A9"]
    explode = [0, 0.03, 0.03, 0]

    wedges, texts = ax2.pie(sizes, labels=pie_labels, colors=pie_colors,
                            startangle=90, explode=explode,
                            textprops={"fontsize": 9},
                            wedgeprops={"linewidth": 0.5, "edgecolor": "white"})
    ax2.set_title("Step time breakdown", fontsize=11, fontweight="bold")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_comparison(datasets: list, output_path: str):
    models_raw = [d.get("model", "?") for d in datasets]
    # Add newline before parenthetical for readability
    models = [m.replace(" (", "\n(") if "(" in m else m for m in models_raw]
    n = len(models)
    derived = [_derive(d) for d in datasets]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Hardware efficiency comparison (16× A100 GPUs, TP=4, PP=4)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.06, right=0.97, wspace=0.3)

    x = np.arange(n)
    bw = 0.5 if n <= 3 else 0.35
    # Per-model colors (distinct from plot 2's semantic palette)
    model_colors = ["#5B8DBE", "#E8913A", "#6BB578", "#9B7FCC", "#D4537E"][:n]

    # --- 1. MFU ---
    mfu = [d["mfu"] for d in derived]
    bars1 = ax1.bar(x, mfu, color=model_colors, width=bw)
    for bar, val in zip(bars1, mfu):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylabel("MFU (%)")
    ax1.set_title("Model FLOPS utilization", fontsize=11, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_ylim(0, max(mfu) * 1.4 + 2)

    # --- 2. Stacked time breakdown ---
    compute = [d["compute"] for d in derived]
    exposed = [d["exposed_comm"] for d in derived]
    overlapped = [d["overlapped_comm"] for d in derived]
    idle = [d["idle"] for d in derived]

    ax2.bar(x, compute, color="#3266ad", width=bw, label="Compute")
    ax2.bar(x, exposed, bottom=compute, color="#D85A30", width=bw, label="Exposed comm")
    ax2.bar(x, overlapped,
            bottom=[c + e for c, e in zip(compute, exposed)],
            color="#1D9E75", width=bw, label="Overlapped comm")
    ax2.bar(x, idle,
            bottom=[c + e + o for c, e, o in zip(compute, exposed, overlapped)],
            color="#B4B2A9", width=bw, label="Idle")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylabel("% of step time")
    ax2.set_title("Step time breakdown", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_ylim(0, 110)

    # --- 3. Load imbalance ---
    load_imb = [d.get("load_imbalance_ratio", 1) for d in datasets]
    bars3 = ax3.bar(x, load_imb, color=model_colors, width=bw)
    for bar, val in zip(bars3, load_imb):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}×", ha="center", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=9)
    ax3.set_ylabel("Slowest / fastest rank")
    ax3.set_title("Slowest / Fastest rank ratio", fontsize=11, fontweight="bold")
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.axhline(y=1.0, color="#B4B2A9", linestyle="--", linewidth=0.8, label="Perfect balance")
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, max(load_imb) * 1.3 + 0.1)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot hardware efficiency metrics")
    parser.add_argument("files", nargs="*", help="metrics_summary.txt file(s)")
    parser.add_argument("--discover", type=str, default=None,
                        help="Auto-discover all metrics_summary.txt under this directory")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file path (default: plot_efficiency.png)")
    args = parser.parse_args()

    if args.discover:
        paths = discover_results(args.discover)
        if not paths:
            print(f"No metrics_summary.txt found under {args.discover}")
            sys.exit(1)
        print(f"Found {len(paths)} result(s):")
        for p in paths:
            print(f"  {p}")
    elif args.files:
        paths = args.files
    else:
        parser.print_help()
        sys.exit(1)

    datasets = [parse_metrics(p) for p in paths]

    output = args.output
    if output is None:
        if len(paths) == 1:
            output = os.path.join(os.path.dirname(paths[0]), "plot_efficiency.png")
        else:
            output = "plot_efficiency.png"

    if len(datasets) == 1:
        plot_single(datasets[0], output)
    else:
        plot_comparison(datasets, output)


if __name__ == "__main__":
    main()