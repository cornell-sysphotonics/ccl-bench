#!/usr/bin/env python3
"""Plot NVLink TX/RX deltas from a CSV exported by utilization_reader.py."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pynvml import (
    NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX,
    NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX,
)

NVLINK_FIELDS = {
    NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX: "TX",
    NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX: "RX",
}


def add_deltas(df: pd.DataFrame, counter_bits: int) -> pd.DataFrame:
    """Return dataframe with delta_bytes per field/scope, handling wrap-around."""
    max_value = 1 << counter_bits
    df = df.sort_values("timestamp_ns").copy()

    def _delta(series: pd.Series) -> pd.Series:
        diff = series.diff()
        wrap_mask = diff < 0
        diff[wrap_mask] = diff[wrap_mask] + max_value
        diff.iloc[0] = 0
        return diff

    df["delta_bytes"] = (
        df.groupby(["field_id", "scope_id"], group_keys=False)["value"].apply(_delta)
    )
    df["delta_bytes"] = df["delta_bytes"].fillna(0)
    return df


def plot_deltas(df: pd.DataFrame, time_column: str, out_path: Path, log_scale: bool):
    """Create a TX/RX delta plot."""
    time_ms = df[time_column] * 1e-6
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    directions = ["TX", "RX"]

    for idx, direction in enumerate(directions):
        ax = axes[idx]
        subset = df[df["direction"] == direction]
        if subset.empty:
            ax.set_title(f"{direction} (no data)")
            ax.set_ylabel("Δ bytes")
            continue

        for link_id, group in subset.groupby("scope_id"):
            ax.plot(
                time_ms.loc[group.index],
                group["delta_bytes"],
                label=f"Link {link_id}",
            )
        ax.set_ylabel("Δ bytes")
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(f"NVLink {direction} delta per sample")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel(f"{time_column} (ms)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot NVLink deltas from CSV data.")
    parser.add_argument("csv", type=Path, help="CSV exported via utilization_reader --csv")
    parser.add_argument("--out", type=Path, default=Path("nvlink_deltas.png"), help="Output image path")
    parser.add_argument(
        "--counter-bits",
        type=int,
        default=32,
        help="Bit-width of NVLink counters for wrap handling (default: 32).",
    )
    parser.add_argument(
        "--time-base",
        choices=["host_timestamp_ns", "nvml_timestamp_ns"],
        default="host_timestamp_ns",
        help="Column to use for the x-axis timeline.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Plot delta bytes using a logarithmic y-axis.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit("CSV is empty; nothing to plot.")

    if "value" not in df.columns:
        raise SystemExit("CSV must contain a 'value' column with counter readings.")

    df["field_id"] = pd.to_numeric(df["field_id"], errors="coerce").astype("Int64")
    df["scope_id"] = pd.to_numeric(df["scope_id"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    filtered = df[df["field_id"].isin(NVLINK_FIELDS.keys())].copy()
    if filtered.empty:
        raise SystemExit("No NVLink throughput fields found in CSV.")

    if args.time_base not in filtered.columns:
        raise SystemExit(f"Column '{args.time_base}' not found in CSV.")

    filtered["direction"] = filtered["field_id"].map(NVLINK_FIELDS)
    filtered = filtered.rename(columns={args.time_base: "timestamp_ns"})

    delta_df = add_deltas(filtered, args.counter_bits)
    plot_deltas(delta_df, "timestamp_ns", args.out, args.log_scale)


if __name__ == "__main__":
    main()

