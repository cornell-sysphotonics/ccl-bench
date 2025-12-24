#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import csv
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Parsing
# -----------------------------
TIME_RE = re.compile(
    r"""
    ^\s*
    (?P<pct>\d+(\.\d+)?)%\s+
    (?P<total>[\d\.]+)\s*(?P<unit>ms|s|us|μs)\s+
    (?P<instances>\d+)\s+
    (?P<avg>[\d\.]+)\s*(us|μs|ms|s)\s+
    (?P<med>[\d\.]+)\s*(us|μs|ms|s)\s+
    (?P<min>[\d\.]+)\s*(us|μs|ms|s)\s+
    (?P<max>[\d\.]+)\s*(us|μs|ms|s)\s+
    (?P<std>[\d\.]+)\s*(us|μs|ms|s)\s+
    (?P<name>.+?)\s*$
    """,
    re.VERBOSE,
)


def to_seconds(val: float, unit: str) -> float:
    unit = unit.strip()
    if unit == "s":
        return val
    if unit == "ms":
        return val / 1e3
    if unit in ("us", "μs"):
        return val / 1e6
    raise ValueError(f"Unknown unit: {unit}")


def parse_nsys_txt(path: Path):
    """
    Returns list of dict rows: {name, total_s, pct, instances}
    """
    rows = []
    txt = path.read_text(errors="ignore").splitlines()
    for line in txt:
        m = TIME_RE.match(line)
        if not m:
            continue
        total = float(m.group("total"))
        unit = m.group("unit")
        rows.append(
            {
                "name": m.group("name").strip(),
                "total_s": to_seconds(total, unit),
                "pct": float(m.group("pct")),
                "instances": int(m.group("instances")),
            }
        )
    if not rows:
        raise RuntimeError(f"No kernel rows parsed from {path}. Check txt format.")
    return rows


# -----------------------------
# Categorization
# -----------------------------
def cat_kernel(name: str) -> str:
    n = name.lower()
    if "nccl" in n:
        return "NCCL"
    if "fused_moe" in n or "moe_" in n or "vllm::moe" in n:
        return "MoE"
    # very rough GEMM detection
    if "gemm" in n or "cublas" in n or "cutlass" in n:
        return "GEMM"
    return "Other"


def short_nccl_name(name: str) -> str:
    # Make readable labels for top NCCL kernels
    # e.g. ncclDevKernel_Broadcast_RING_LL(...) -> Broadcast_RING_LL
    base = name
    base = base.replace("ncclDevKernel_", "")
    base = base.split("(")[0]
    return base


# -----------------------------
# Plot styles (less "AI", more "paper")
# -----------------------------
def set_style():
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# -----------------------------
# Main plots
# -----------------------------
def plot_overall_breakdown(labels, totals_by_cat, outpath: Path):
    """
    Stacked bars: NCCL / MoE / GEMM / Other
    """
    cats = ["NCCL", "MoE", "GEMM", "Other"]
    x = np.arange(len(labels))
    width = 0.62

    bottoms = np.zeros(len(labels))
    fig = plt.figure(figsize=(10.5, 5.2))
    ax = plt.gca()

    for c in cats:
        vals = np.array([totals_by_cat[l].get(c, 0.0) for l in labels])
        ax.bar(x, vals, width, bottom=bottoms, label=c)
        bottoms += vals

    ax.set_xticks(x, labels)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Overall kernel time breakdown (stacked)")
    ax.legend(ncols=4, frameon=False, loc="upper right")
    ax.margins(x=0.03)

    # annotate total on top
    for i, t in enumerate(bottoms):
        ax.text(i, t * 1.01, f"{t:.1f}s", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_top_nccl_grouped(labels, nccl_times_by_label, topk, outpath: Path):
    """
    Grouped bars for top NCCL kernels (union across configs), values = absolute seconds.
    """
    # union all NCCL kernels and pick global topk by total time (sum across labels)
    global_sum = Counter()
    for l in labels:
        for k, v in nccl_times_by_label[l].items():
            global_sum[k] += v
    top = [k for k, _ in global_sum.most_common(topk)]

    if not top:
        raise RuntimeError("No NCCL kernels found. Did parsing miss 'nccl' names?")

    # build matrix: kernels x labels
    mat = np.array([[nccl_times_by_label[l].get(k, 0.0) for l in labels] for k in top])

    fig = plt.figure(figsize=(12.5, 5.6))
    ax = plt.gca()

    group_x = np.arange(len(top))
    bar_w = 0.22
    offsets = np.linspace(-bar_w, bar_w, len(labels))

    for j, l in enumerate(labels):
        ax.bar(group_x + offsets[j], mat[:, j], width=bar_w, label=l)

    ax.set_xticks(group_x, [short_nccl_name(k) for k in top], rotation=20, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.set_title(f"Top NCCL kernels by time (grouped, top {topk})")
    ax.legend(frameon=False, ncols=len(labels), loc="upper right")
    ax.margins(x=0.01)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(labels, totals_by_cat, nccl_total, outpath: Path):
    cats = ["NCCL", "MoE", "GEMM", "Other"]
    with outpath.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["label", "total_kernel_time_s"]
            + [f"{c}_s" for c in cats]
            + ["nccl_fraction"]
        )
        for l in labels:
            total = sum(totals_by_cat[l].values())
            row = [l, f"{total:.6f}"] + [
                f"{totals_by_cat[l].get(c, 0.0):.6f}" for c in cats
            ]
            frac = (totals_by_cat[l].get("NCCL", 0.0) / total) if total > 0 else 0.0
            row.append(f"{frac:.6f}")
            w.writerow(row)


# -----------------------------
# CLI
# -----------------------------
def parse_inputs(items):
    out = []
    for it in items:
        if "=" not in it:
            raise ValueError(f"Bad --inputs item: {it}. Use label=path")
        label, p = it.split("=", 1)
        out.append((label.strip(), Path(p).expanduser()))
    return out


def main():
    set_style()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs", nargs="+", required=True, help="label=path label=path ..."
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    inputs = parse_inputs(args.inputs)
    labels = [l for l, _ in inputs]

    totals_by_cat = {l: defaultdict(float) for l in labels}
    nccl_times_by_label = {l: defaultdict(float) for l in labels}

    for label, path in inputs:
        rows = parse_nsys_txt(path)
        for r in rows:
            c = cat_kernel(r["name"])
            totals_by_cat[label][c] += r["total_s"]
            if c == "NCCL":
                nccl_times_by_label[label][r["name"]] += r["total_s"]

    # Outputs: ONLY 2 figures + summary.csv
    plot_overall_breakdown(
        labels,
        totals_by_cat,
        outdir / "figure_overall_breakdown.png",
    )

    plot_top_nccl_grouped(
        labels,
        nccl_times_by_label,
        args.topk,
        outdir / "figure_top_nccl_grouped.png",
    )

    write_summary_csv(
        labels,
        totals_by_cat,
        None,
        outdir / "summary.csv",
    )

    print(
        f"✅ Wrote:\n  {outdir / 'figure_overall_breakdown.png'}\n  {outdir / 'figure_top_nccl_grouped.png'}\n  {outdir / 'summary.csv'}"
    )


if __name__ == "__main__":
    main()
