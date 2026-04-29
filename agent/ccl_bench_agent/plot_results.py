#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot CCL-Bench ADRS agent results from a results_<timestamp>.csv file.

Panels:
  1. Score per iteration  — scatter (colour = pass/fail) + running-best line
  2. Search time          — bar chart (seconds spent in update_policy per iter)
  3. Config heatmap       — integer/bool config dimensions across iterations

Usage:
    python plot_results.py                        # auto-finds latest results_*.csv
    python plot_results.py results_20260428.csv
    python plot_results.py results_*.csv          # glob OK in bash
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── CSV loading ────────────────────────────────────────────────────────────────

def _latest_csv() -> Path:
    runs_dir = Path(__file__).parent / "runs"
    candidates = sorted(runs_dir.glob("*/results.csv"))
    if not candidates:
        raise FileNotFoundError(f"No results.csv found under {runs_dir}")
    return candidates[-1]


_CONFIG_ALIASES = {
    "micro_batch_size": "micro_batch",
}


def _canonicalize_config(config):
    """Merge known config-key aliases so one tunable gets one heatmap row."""
    canonical = {}
    for key, value in config.items():
        canonical_key = _CONFIG_ALIASES.get(key, key)
        canonical[canonical_key] = value
    return canonical


def load_csv(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            it = int(row["iteration"])
            score = float(row["score"]) if row.get("score") else float("nan")
            best  = float(row["best_score"]) if row.get("best_score") else float("nan")
            stime = float(row["search_time_s"]) if row.get("search_time_s") else float("nan")
            config = _canonicalize_config(json.loads(row.get("config") or "{}"))
            rows.append({
                "iteration":     it,
                "version":       int(row.get("version", 0)),
                "score":         score,
                "best_score":    best,
                "status":        row.get("status", ""),
                "search_time_s": stime,
                "config":        config,
            })
    return rows


# ── Plot helpers ───────────────────────────────────────────────────────────────

_OK_COLOR   = "#16a34a"   # green — success
_FAIL_COLOR = "#dc2626"   # red   — error/timeout
_BEST_COLOR = "#2563eb"   # blue  — running best line
_BAR_COLOR  = "#9333ea"   # purple — search time bars


def _plot_scores(ax, rows):
    iters  = [r["iteration"] for r in rows]
    scores = [r["score"] for r in rows]
    bests  = [r["best_score"] for r in rows]
    colors = [_OK_COLOR if r["status"] == "success" else _FAIL_COLOR for r in rows]

    ax.scatter(iters, scores, c=colors, s=60, zorder=3, label="_nolegend_")
    ax.plot(iters, bests, "-", color=_BEST_COLOR, linewidth=2,
            label="Running best", zorder=2)

    # Legend proxies for pass/fail dots
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_OK_COLOR,
               markersize=8, label="Success"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_FAIL_COLOR,
               markersize=8, label="Failed"),
        Line2D([0], [0], color=_BEST_COLOR, linewidth=2, label="Running best"),
    ], loc="upper right", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Score per Iteration")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)


def _plot_search_time(ax, rows):
    iters = [r["iteration"] for r in rows]
    times = [r["search_time_s"] for r in rows]
    valid = [(i, t) for i, t in zip(iters, times) if not np.isnan(t)]
    if not valid:
        ax.set_visible(False)
        return
    xi, yt = zip(*valid)
    ax.bar(xi, yt, color=_BAR_COLOR, alpha=0.8)
    ax.set_ylabel("Search time (s)")
    ax.set_title("LLM Search Time per Iteration")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)


def _plot_config_heatmap(ax, rows):
    """Heatmap of numeric/bool config dimensions across iterations."""
    # Collect all config keys that ever appear
    all_keys = []
    seen = set()
    for r in rows:
        for k in r["config"]:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    if not all_keys:
        ax.set_visible(False)
        return

    iters = [r["iteration"] for r in rows]
    matrix = np.full((len(all_keys), len(iters)), float("nan"))

    for col, r in enumerate(rows):
        for row_idx, k in enumerate(all_keys):
            v = r["config"].get(k)
            if v is None:
                continue
            if isinstance(v, bool):
                matrix[row_idx, col] = float(v)
            else:
                try:
                    matrix[row_idx, col] = float(v)
                except (TypeError, ValueError):
                    pass

    # Normalise each row to [0, 1] for colour mapping
    normed = np.copy(matrix)
    for i in range(len(all_keys)):
        row = matrix[i]
        valid = row[~np.isnan(row)]
        if len(valid) == 0 or valid.max() == valid.min():
            normed[i] = 0.5
        else:
            normed[i] = (row - valid.min()) / (valid.max() - valid.min())

    im = ax.imshow(normed, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(all_keys)))
    ax.set_yticklabels(all_keys, fontsize=8)
    ax.set_xticks(range(len(iters)))
    ax.set_xticklabels(iters, fontsize=8)
    ax.set_xlabel("Iteration")
    ax.set_title("Config Dimensions (normalised)")

    # Annotate cells with raw values
    for row_idx in range(len(all_keys)):
        for col, r in enumerate(rows):
            v = r["config"].get(all_keys[row_idx])
            if v is not None:
                ax.text(col, row_idx, str(v), ha="center", va="center",
                        fontsize=7, color="black")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_csv()
    print(f"Reading: {csv_path}")

    rows = load_csv(csv_path)
    if not rows:
        print("No rows found.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 11),
                             gridspec_kw={"height_ratios": [3, 2, 2]})
    fig.suptitle(f"ADRS Results — {csv_path.name}", fontsize=12, y=0.98)

    _plot_scores(axes[0], rows)
    _plot_search_time(axes[1], rows)
    _plot_config_heatmap(axes[2], rows)

    for ax in axes:
        if ax.get_visible():
            ax.set_xlabel("Iteration")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = csv_path.with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved:   {out_path}")


if __name__ == "__main__":
    main()
