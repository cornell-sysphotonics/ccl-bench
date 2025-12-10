#!/usr/bin/env python3
"""Plot NVLink TX/RX deltas (summed or per-link) from utilization_reader CSV data."""
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


def load_delta_dataframe(csv_path: Path, time_base: str, counter_bits: int) -> pd.DataFrame:
    """Load a CSV, filter NVLink fields, and compute per-sample deltas."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"{csv_path} is empty; nothing to plot.")

    if "value" not in df.columns:
        raise SystemExit(f"{csv_path} must contain a 'value' column with counter readings.")

    df["field_id"] = pd.to_numeric(df["field_id"], errors="coerce").astype("Int64")
    df["scope_id"] = pd.to_numeric(df["scope_id"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    filtered = df[df["field_id"].isin(NVLINK_FIELDS.keys())].copy()
    if filtered.empty:
        raise SystemExit(f"No NVLink throughput fields found in {csv_path}.")

    if time_base not in filtered.columns:
        raise SystemExit(f"Column '{time_base}' not found in {csv_path}.")

    filtered["direction"] = filtered["field_id"].map(NVLINK_FIELDS)
    filtered = filtered.rename(columns={time_base: "timestamp_ns"})
    return add_deltas(filtered, counter_bits)


def dataset_label(path: Path) -> str:
    """Return a human-friendly label for plots derived from a CSV path."""
    if path.suffix:
        return path.stem
    return path.name


def plot_sum_deltas(df: pd.DataFrame, time_column: str, out_path: Path, log_scale: bool):
    """Create TX/RX delta plots showing the sum over all NVLinks per sample."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    directions = ["TX", "RX"]

    for idx, direction in enumerate(directions):
        ax = axes[idx]
        subset = df[df["direction"] == direction]
        if subset.empty:
            ax.set_title(f"{direction} (no data)")
            ax.set_ylabel("Δ KiBs")
            continue

        total_delta = (
            subset.groupby(time_column, as_index=False)["delta_bytes"]
            .sum()
            .sort_values(time_column)
        )
        if not total_delta.empty:
            ax.plot(
                total_delta[time_column] * 1e-6,
                total_delta["delta_bytes"],
                label="All Links (sum)",
                color="black",
                linewidth=2.5,
            )
        ax.set_ylabel("Δ KiBs")
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(f"NVLink {direction} Throughput Per Millisecond (all links)")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def plot_per_link_deltas(
    datasets: list[tuple[str, pd.DataFrame]],
    time_column: str,
    out_path: Path,
    log_scale: bool,
):
    """Create TX/RX delta plots for each NVLink, optionally comparing datasets side-by-side."""
    directions = ["TX", "RX"]
    num_datasets = len(datasets)
    if num_datasets == 0:
        raise ValueError("At least one dataset is required to plot per-link deltas.")

    if out_path.is_dir():
        base_dir = out_path
        base_stem = "nvlink_deltas"
        suffix = ".png"
    else:
        base_dir = out_path.parent if out_path.parent != Path("") else Path(".")
        suffix = out_path.suffix if out_path.suffix else ".png"
        base_stem = out_path.stem if out_path.suffix else out_path.name

    base_dir.mkdir(parents=True, exist_ok=True)

    link_ids = sorted(
        {
            int(scope)
            for _, df in datasets
            for scope in df["scope_id"].dropna().unique()
        }
    )
    if not link_ids:
        print("No NVLink scopes present; nothing to plot.")
        return

    sharey = "row" if num_datasets > 1 else False

    for link_id in link_ids:
        fig, axes = plt.subplots(
            len(directions),
            num_datasets,
            figsize=(12 * num_datasets, 8),
            sharey=sharey,
        )
        if num_datasets == 1:
            axes = axes.reshape(len(directions), 1)

        for col_idx, (label, df) in enumerate(datasets):
            link_group = df[df["scope_id"] == link_id]
            for row_idx, direction in enumerate(directions):
                ax = axes[row_idx, col_idx]
                subset = link_group[link_group["direction"] == direction]

                if col_idx == 0:
                    ax.set_ylabel("Δ KiBs")
                if subset.empty:
                    ax.set_title(f"{label} - NVLink {link_id} {direction} (no data)")
                    continue

                time_ms = subset[time_column] * 1e-6
                ax.plot(time_ms, subset["delta_bytes"], label=direction)
                if log_scale:
                    ax.set_yscale("log")
                ax.set_title(
                    f"{label} - NVLink {link_id} {direction} Throughput Per Millisecond"
                )
                ax.legend(loc="upper right")

            axes[-1, col_idx].set_xlabel("Time (ms)")

        fig.tight_layout()
        link_path = base_dir / f"{base_stem}_link{link_id}{suffix}"
        fig.savefig(link_path, dpi=150)
        print(f"Saved plot to {link_path}")


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
    parser.add_argument(
        "--csv-compare",
        type=Path,
        default=None,
        help=(
            "Optional second CSV to compare in per-link mode. When provided, plots are laid out "
            "side-by-side with shared y-axes but independent x-axes."
        ),
    )
    parser.add_argument(
        "--plot-mode",
        choices=["sum", "per-link"],
        default="sum",
        help=(
            "Select 'sum' to plot the aggregate delta across all NVLinks into a single file "
            "or 'per-link' to emit one plot per NVLink (output path used as prefix/directory)."
        ),
    )
    args = parser.parse_args()

    primary_df = load_delta_dataframe(args.csv, args.time_base, args.counter_bits)

    if args.plot_mode == "sum":
        if args.csv_compare:
            raise SystemExit("--csv-compare is only supported when --plot-mode=per-link.")
        plot_sum_deltas(primary_df, "timestamp_ns", args.out, args.log_scale)
        return

    datasets: list[tuple[str, pd.DataFrame]] = [(dataset_label(args.csv), primary_df)]
    if args.csv_compare:
        compare_df = load_delta_dataframe(args.csv_compare, args.time_base, args.counter_bits)
        datasets.append((dataset_label(args.csv_compare), compare_df))

    plot_per_link_deltas(datasets, "timestamp_ns", args.out, args.log_scale)


if __name__ == "__main__":
    main()

