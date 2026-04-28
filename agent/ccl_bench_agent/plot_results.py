#!/usr/bin/env python3
"""Plot success rate and avg wall time across iterations from eval_results CSV."""

import sys
import csv
from pathlib import Path

import matplotlib.pyplot as plt

def load_results(csv_path: str) -> dict:
    """Parse CSV into per-iteration stats."""
    iters = {}  # iteration -> {"successes": [...], "total": int}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            it = int(row["iteration"])
            if it not in iters:
                iters[it] = {"successes": [], "total": 0}
            iters[it]["total"] += 1
            wt = row.get("wall_time", "").strip()
            if wt:
                iters[it]["successes"].append(int(wt))
    return iters


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent / "eval_results_20260319_150815.csv"
    )
    iters = load_results(csv_path)

    iterations = sorted(iters.keys())
    success_rates = []
    avg_wall_times = []

    for it in iterations:
        d = iters[it]
        sr = len(d["successes"]) / d["total"] if d["total"] else 0
        avg = sum(d["successes"]) / len(d["successes"]) if d["successes"] else 0
        success_rates.append(sr * 100)
        avg_wall_times.append(avg / 1e6)  # convert to millions

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = "#2563eb"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Success Rate (%)", color=color1)
    ax1.plot(iterations, success_rates, "o-", color=color1, linewidth=2, markersize=6, label="Success Rate")
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#dc2626"
    ax2.set_ylabel("Avg Wall Time of Successes (M)", color=color2)
    ax2.plot(iterations, avg_wall_times, "s--", color=color2, linewidth=2, markersize=6, label="Avg Wall Time")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.set_xticks(iterations)
    ax1.grid(axis="x", alpha=0.3)
    ax1.set_title("Policy Optimization Progress")
    fig.tight_layout()

    out_path = Path(csv_path).with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
