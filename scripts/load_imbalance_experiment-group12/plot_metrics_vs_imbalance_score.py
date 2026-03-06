#!/usr/bin/env python3
"""
Plot performance metrics vs imbalance score.

X axis:
  - imbalance_score from each result JSON

Y axis:
  - performance metric (TTFT / TPOT / throughput)

Comparison:
  - Baseline vs All-to-All

Plot:
  - scatter only (no fitting line)

Outlier handling:
  - IQR-based filtering (1.5 × IQR) applied on BOTH X and Y

Usage examples:

  # TTFT
  python plot_metrics_vs_imbalance_score.py \
    --baseline-results baseline_results_json \
    --all2all-results default_all2all_results_json \
    --metric mean_ttft_ms \
    --output ttft_vs_imbalance_score.png

  # TPOT
  python plot_metrics_vs_imbalance_score.py \
    --baseline-results baseline_results_json \
    --all2all-results default_all2all_results_json \
    --metric mean_tpot_ms \
    --output tpot_vs_imbalance_score.png

  # Throughput
  python plot_metrics_vs_imbalance_score.py \
    --baseline-results baseline_results_json \
    --all2all-results default_all2all_results_json \
    --metric output_throughput \
    --output throughput_vs_imbalance_score.png
"""

import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# -------------------------------------------------
# Load (X, Y) from result JSONs
# -------------------------------------------------
def load_xy(results_dir, metric):
    """
    Read imbalance_score as X
    Read metric as Y

    Expected JSON fields:
      - imbalance_score
      - metric (e.g., mean_ttft_ms)

    Returns:
      x: np.ndarray
      y: np.ndarray
    """
    results_dir = Path(results_dir)

    xs, ys = [], []

    for file in results_dir.glob("*.json"):
        try:
            with open(file, "r") as f:
                obj = json.load(f)

            if "imbalance_score" not in obj:
                continue
            if metric not in obj:
                continue

            xs.append(obj["imbalance_score"])
            ys.append(obj[metric])

        except Exception:
            continue

    return np.array(xs), np.array(ys)


# -------------------------------------------------
# IQR-based outlier removal
# -------------------------------------------------
def filter_outliers_iqr(x, y):
    """
    Remove outliers using the IQR rule (1.5 × IQR).

    A point is kept ONLY if it is non-outlier in BOTH x and y.
    """
    if len(x) < 4:
        return x, y

    def iqr_mask(arr):
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return (arr >= lower) & (arr <= upper)

    mask = iqr_mask(x) & iqr_mask(y)
    return x[mask], y[mask]


# -------------------------------------------------
# Plot
# -------------------------------------------------
def plot_one(metric,
             baseline_dir,
             all2all_dir,
             output):

    bx, by = load_xy(baseline_dir, metric)
    ax, ay = load_xy(all2all_dir, metric)

    # sort by X
    if len(bx) > 0:
        order = np.argsort(bx)
        bx, by = bx[order], by[order]
    if len(ax) > 0:
        order = np.argsort(ax)
        ax, ay = ax[order], ay[order]

    # outlier filtering (IDENTICAL logic as before)
    bx, by = filter_outliers_iqr(bx, by)
    ax, ay = filter_outliers_iqr(ax, ay)

    print(f"[Baseline] points after filtering = {len(bx)}")
    print(f"[All2All]  points after filtering = {len(ax)}")

    if len(bx) == 0 and len(ax) == 0:
        print("❌ No valid points found.")
        return

    plt.figure(figsize=(7, 5))

    if len(bx) > 0:
        plt.scatter(bx, by, label="Baseline", marker="o", alpha=0.8)
    if len(ax) > 0:
        plt.scatter(ax, ay, label="All-to-All", marker="^", alpha=0.8)

    plt.xlabel("Imbalance Score")
    plt.ylabel(metric.replace("_", " "))
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

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

    parser.add_argument("--baseline-results", required=True,
                        help="Directory with baseline result JSON files")
    parser.add_argument("--all2all-results", required=True,
                        help="Directory with all-to-all result JSON files")
    parser.add_argument("--metric", required=True,
                        choices=[
                            "mean_ttft_ms",
                            "mean_tpot_ms",
                            "output_throughput"
                        ])
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    plot_one(
        args.metric,
        args.baseline_results,
        args.all2all_results,
        args.output
    )