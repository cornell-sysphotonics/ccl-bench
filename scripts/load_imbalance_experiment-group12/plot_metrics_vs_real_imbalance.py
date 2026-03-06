#!/usr/bin/env python3
"""
Plot performance metrics vs REAL MoE load imbalance (CV).

- X axis: real MoE load imbalance (CV)
- Y axis: performance metric
- Compare: Baseline vs All-to-All
- Plot: scatter (NO fitting line)
- Outliers are removed using IQR (1.5 x IQR) on both X and Y

Example:
python plot_metrics_vs_real_imbalance.py \
  --baseline-imbalance baseline_real_imbalance.json \
  --baseline-results baseline_results_json \
  --all2all-imbalance new_default_results.json \
  --all2all-results default_all2all_results_json \
  --metric mean_tpot_ms \
  --output tpot_vs_real_imbalance.png
"""

import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# -------------------------------------------------
# Load real imbalance (CV)
# -------------------------------------------------
def load_imbalance(json_path):
    """
    Load real imbalance mapping:
    {
      "gates_logs_01": 0.35,
      ...
    }
    """
    with open(json_path, "r") as f:
        return json.load(f)


# -------------------------------------------------
# Load performance metric from result JSONs
# -------------------------------------------------
def load_metric(results_dir, metric):
    """
    Expected filenames:
      001.json, 002.json, ..., 030.json

    Returned dict:
      {
        "01": metric_value,
        "02": metric_value,
        ...
      }
    """
    results_dir = Path(results_dir)
    data = {}

    for file in results_dir.glob("*.json"):
        try:
            with open(file, "r") as f:
                obj = json.load(f)

            if metric not in obj:
                continue

            # "001.json" -> "01"
            run_id = file.stem[-2:]
            data[run_id] = obj[metric]

        except Exception:
            continue

    return data


# -------------------------------------------------
# IQR-based outlier removal
# -------------------------------------------------
def filter_outliers_iqr(x, y):
    """
    Remove outliers using the IQR rule (1.5 × IQR)
    A point is kept ONLY if it is non-outlier in BOTH x and y.
    """
    if len(x) < 4:
        return x, y  # not enough points for stable IQR

    def iqr_mask(arr):
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return (arr >= lower) & (arr <= upper)

    mask_x = iqr_mask(x)
    mask_y = iqr_mask(y)
    mask = mask_x & mask_y

    return x[mask], y[mask]


# -------------------------------------------------
# Prepare matched X/Y pairs
# -------------------------------------------------
def prepare_xy(imbalance, results):
    """
    Match:
      gates_logs_01  <->  "01"
    """
    xs, ys = [], []

    for k, v in imbalance.items():
        run_id = k.replace("gates_logs_", "")
        if run_id in results:
            xs.append(v)
            ys.append(results[run_id])

    xs = np.array(xs)
    ys = np.array(ys)

    if len(xs) == 0:
        return xs, ys

    # sort by x (imbalance)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    # 🔹 outlier removal
    xs, ys = filter_outliers_iqr(xs, ys)

    return xs, ys


# -------------------------------------------------
# Plot
# -------------------------------------------------
def plot_one(metric,
             baseline_imb, baseline_res,
             all2all_imb, all2all_res,
             output):

    bx, by = prepare_xy(baseline_imb, baseline_res)
    ax, ay = prepare_xy(all2all_imb, all2all_res)

    print(f"[Baseline] points after filtering = {len(bx)}")
    print(f"[All2All]  points after filtering = {len(ax)}")

    if len(bx) == 0 and len(ax) == 0:
        print("❌ No matched points found.")
        return

    plt.figure(figsize=(7, 5))

    if len(bx) > 0:
        plt.scatter(bx, by, label="Baseline", marker="o", alpha=0.8)
    if len(ax) > 0:
        plt.scatter(ax, ay, label="All-to-All", marker="^", alpha=0.8)

    plt.xlabel("Real MoE Load Imbalance (CV)")
    plt.ylabel(metric.replace("_", " "))
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    # 🔹 Explain outlier filtering in the figure
    plt.text(
        0.01, 0.01,
        "Outliers removed using IQR (1.5×IQR) on both axes",
        transform=plt.gca().transAxes,
        fontsize=9,
        alpha=0.7
    )

    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"✅ Saved figure to {output}")


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline-imbalance", required=True)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--all2all-imbalance", required=True)
    parser.add_argument("--all2all-results", required=True)
    parser.add_argument("--metric", required=True,
                        choices=[
                            "mean_ttft_ms",
                            "mean_tpot_ms",
                            "output_throughput"
                        ])
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    baseline_imb = load_imbalance(args.baseline_imbalance)
    all2all_imb = load_imbalance(args.all2all_imbalance)

    baseline_res = load_metric(args.baseline_results, args.metric)
    all2all_res = load_metric(args.all2all_results, args.metric)

    plot_one(
        args.metric,
        baseline_imb,
        baseline_res,
        all2all_imb,
        all2all_res,
        args.output
    )