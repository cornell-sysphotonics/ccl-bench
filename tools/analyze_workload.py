#!/usr/bin/env python3
"""CCL-Bench Workload Analysis Tool.

Analyzes trace data across multiple iterations and generates high-quality
visualizations for each metric.

Usage:
    python analyze_workload.py --trace-base <path> --output <output_dir>
    python analyze_workload.py --trace-base /pscratch/sd/i/imh39/ccl-bench-traces/llama3_8b_pp/torch_traces --output ./analysis_output

Features:
    - Processes all iterations in a trace directory
    - Computes all available metrics for each iteration
    - Generates publication-quality plots
    - Exports data to CSV and JSON formats
    - Creates a comprehensive HTML report
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


# Add tools directory to path for metric module imports
TOOLS_DIR = Path(__file__).parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def _load_metric_module(folder_name: str) -> Any:
    """Dynamically load a metric module from the tools directory.

    Args:
        folder_name: The folder name (e.g., "coll_call_num_16")

    Returns:
        The metric_cal function from the module.
    """
    module_path = TOOLS_DIR / folder_name / "metric.py"
    spec = importlib.util.spec_from_file_location(f"{folder_name}.metric", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.metric_cal


# Load metric modules dynamically
calc_coll_call_num = _load_metric_module("coll_call_num_16")
calc_comm_comp_overlap = _load_metric_module("comm_comp_overlap_16")
calc_iter_time = _load_metric_module("iter_time_16")
calc_pipeline_bubble = _load_metric_module("pipeline_bubble_16")
calc_straggler_lag = _load_metric_module("straggler_lag_16")
calc_throughput_tokens = _load_metric_module("throughput_tokens_16")
calc_traffic_distribution = _load_metric_module("traffic_distribution_16")


# Try to import plotting libraries
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for servers
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


# Metric registry using metric names with _16 suffix for group 16
METRICS = {
    "coll_call_num_16": calc_coll_call_num,
    "throughput_tokens_16": calc_throughput_tokens,
    "iter_time_16": calc_iter_time,
    "comm_comp_overlap_16": calc_comm_comp_overlap,
    "pipeline_bubble_16": calc_pipeline_bubble,
    "straggler_lag_16": calc_straggler_lag,
    "traffic_distribution_16": calc_traffic_distribution,
}


@dataclass
class IterationData:
    """Data collected for a single iteration."""

    iteration: int
    trace_dir: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadAnalysis:
    """Complete analysis of a workload across iterations."""

    workload_name: str
    trace_base: str
    iterations: list[IterationData] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "workload_name": self.workload_name,
            "trace_base": self.trace_base,
            "num_iterations": len(self.iterations),
            "iterations": [
                {
                    "iteration": it.iteration,
                    "metrics": it.metrics,
                }
                for it in self.iterations
            ],
        }


def find_iterations(trace_base: str) -> list[tuple[int, str]]:
    """Find all iteration directories in the trace base."""
    trace_path = Path(trace_base)
    iterations = []

    for item in sorted(trace_path.iterdir()):
        if item.is_dir() and item.name.startswith("iteration_"):
            try:
                iter_num = int(item.name.replace("iteration_", ""))
                iterations.append((iter_num, str(item)))
            except ValueError:
                continue

    return sorted(iterations, key=lambda x: x[0])


def collect_metrics(trace_dir: str, metrics_to_run: list[str] | None = None) -> dict[str, Any]:
    """Collect all metrics for a trace directory."""
    results = {}

    metrics_to_run = metrics_to_run or list(METRICS.keys())

    for metric_name in metrics_to_run:
        if metric_name not in METRICS:
            print(f"  Warning: Unknown metric '{metric_name}'", file=sys.stderr)
            continue

        try:
            result = METRICS[metric_name](trace_dir)
            results[metric_name] = result
        except Exception as e:
            print(f"  Error computing {metric_name}: {e}", file=sys.stderr)
            results[metric_name] = {"error": str(e)}

    return results


def analyze_workload(
    trace_base: str,
    workload_name: str | None = None,
    metrics_to_run: list[str] | None = None,
) -> WorkloadAnalysis:
    """Analyze a complete workload across all iterations."""
    if workload_name is None:
        workload_name = Path(trace_base).parent.name

    analysis = WorkloadAnalysis(
        workload_name=workload_name,
        trace_base=trace_base,
    )

    iterations = find_iterations(trace_base)

    if not iterations:
        print(f"No iterations found in {trace_base}", file=sys.stderr)
        return analysis

    print(f"Analyzing workload: {workload_name}")
    print(f"Found {len(iterations)} iterations: {[i[0] for i in iterations]}")
    print()

    for iter_num, iter_dir in iterations:
        print(f"Processing iteration {iter_num}...")

        iter_data = IterationData(
            iteration=iter_num,
            trace_dir=iter_dir,
        )

        iter_data.metrics = collect_metrics(iter_dir, metrics_to_run)
        analysis.iterations.append(iter_data)

    return analysis


# =============================================================================
# Plotting Functions
# =============================================================================


def setup_plot_style() -> None:
    """Configure matplotlib for publication-quality plots."""
    if not HAS_MATPLOTLIB:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "figure.titleweight": "bold",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def get_color_palette() -> dict[str, str]:
    """Get a consistent color palette for plots."""
    return {
        "primary": "#2E86AB",  # Blue
        "secondary": "#A23B72",  # Magenta
        "tertiary": "#F18F01",  # Orange
        "quaternary": "#C73E1D",  # Red
        "success": "#3A7D44",  # Green
        "neutral": "#6C757D",  # Gray
        "pp": "#2E86AB",  # Pipeline Parallel - Blue
        "tp": "#A23B72",  # Tensor Parallel - Magenta
        "dp": "#F18F01",  # Data Parallel - Orange
        "ep": "#C73E1D",  # Expert Parallel - Red
        "unknown": "#6C757D",  # Unknown - Gray
    }


def plot_iteration_time(analysis: WorkloadAnalysis, output_dir: Path) -> str | None:
    """Plot iteration time across iterations."""
    if not HAS_MATPLOTLIB:
        return None

    iterations = []
    avg_times = []
    min_times = []
    max_times = []

    for iter_data in analysis.iterations:
        if "iter_time" not in iter_data.metrics:
            continue
        data = iter_data.metrics["iter_time"]
        if "error" in data:
            continue

        iterations.append(iter_data.iteration)
        avg_times.append(data.get("avg_iter_time_ms", 0))
        min_times.append(data.get("min_iter_time_ms", 0))
        max_times.append(data.get("max_iter_time_ms", 0))

    if not iterations:
        return None

    colors = get_color_palette()
    _fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        iterations,
        avg_times,
        color=colors["primary"],
        alpha=0.8,
        label="Average",
        edgecolor="white",
        linewidth=1,
    )

    # Add error bars if min/max differ
    if min_times != avg_times or max_times != avg_times:
        errors = [
            [avg - min_t for avg, min_t in zip(avg_times, min_times)],
            [max_t - avg for avg, max_t in zip(avg_times, max_times)],
        ]
        ax.errorbar(
            iterations,
            avg_times,
            yerr=errors,
            fmt="none",
            color=colors["quaternary"],
            capsize=5,
            capthick=2,
            linewidth=2,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Iteration Time - {analysis.workload_name}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add value labels on bars
    for _i, (x, y) in enumerate(zip(iterations, avg_times)):
        ax.annotate(f"{y:.1f}", xy=(x, y), ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    filename = output_dir / "iter_time.png"
    plt.savefig(filename)
    plt.close()

    return str(filename)


def plot_throughput(analysis: WorkloadAnalysis, output_dir: Path) -> str | None:
    """Plot throughput across iterations."""
    if not HAS_MATPLOTLIB:
        return None

    iterations = []
    throughputs = []

    for iter_data in analysis.iterations:
        if "throughput_tokens" not in iter_data.metrics:
            continue
        data = iter_data.metrics["throughput_tokens"]
        if "error" in data:
            continue

        iterations.append(iter_data.iteration)
        throughputs.append(data.get("throughput_tokens_per_s", 0))

    if not iterations:
        return None

    colors = get_color_palette()
    _fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        iterations,
        [t / 1000 for t in throughputs],
        color=colors["success"],
        alpha=0.8,
        edgecolor="white",
        linewidth=1,
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Throughput (K tokens/sec)")
    ax.set_title(f"Training Throughput - {analysis.workload_name}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add value labels
    for bar, val in zip(bars, throughputs):
        ax.annotate(
            f"{val / 1000:.1f}K",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Add average line
    avg_throughput = sum(throughputs) / len(throughputs) / 1000
    ax.axhline(
        y=avg_throughput,
        color=colors["quaternary"],
        linestyle="--",
        linewidth=2,
        label=f"Average: {avg_throughput:.1f}K",
    )
    ax.legend()

    plt.tight_layout()
    filename = output_dir / "throughput.png"
    plt.savefig(filename)
    plt.close()

    return str(filename)


def plot_comm_comp_overlap(analysis: WorkloadAnalysis, output_dir: Path) -> str | None:
    """Plot communication/computation overlap."""
    if not HAS_MATPLOTLIB:
        return None

    iterations = []
    comm_times = []
    comp_times = []
    overlap_ratios = []

    for iter_data in analysis.iterations:
        if "comm_comp_overlap" not in iter_data.metrics:
            continue
        data = iter_data.metrics["comm_comp_overlap"]
        if "error" in data:
            continue

        iterations.append(iter_data.iteration)
        comm_times.append(data.get("comm_time_ms", 0))
        comp_times.append(data.get("comp_time_ms", 0))
        overlap_ratios.append(data.get("overlap_ratio_of_comm", 0) * 100)

    if not iterations:
        return None

    colors = get_color_palette()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Stacked bar of comm/comp time
    width = 0.35
    x = range(len(iterations))

    ax1.bar(
        [i - width / 2 for i in x],
        comm_times,
        width,
        label="Communication",
        color=colors["primary"],
        alpha=0.8,
    )
    ax1.bar(
        [i + width / 2 for i in x],
        comp_times,
        width,
        label="Computation",
        color=colors["tertiary"],
        alpha=0.8,
    )

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Communication vs Computation Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(iterations)
    ax1.legend()

    # Right plot: Overlap ratio
    bars = ax2.bar(iterations, overlap_ratios, color=colors["secondary"], alpha=0.8)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Overlap Ratio (%)")
    ax2.set_title("Comm/Comp Overlap Efficiency")
    ax2.set_ylim(0, 100)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add value labels
    for bar, val in zip(bars, overlap_ratios):
        ax2.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Add reference lines
    ax2.axhline(y=80, color=colors["success"], linestyle="--", alpha=0.7, label="Good (80%)")
    ax2.axhline(y=50, color=colors["quaternary"], linestyle="--", alpha=0.7, label="Fair (50%)")
    ax2.legend(loc="lower right")

    fig.suptitle(
        f"Communication/Computation Analysis - {analysis.workload_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    filename = output_dir / "comm_comp_overlap.png"
    plt.savefig(filename)
    plt.close()

    return str(filename)


def plot_pipeline_bubble(analysis: WorkloadAnalysis, output_dir: Path) -> str | None:
    """Plot pipeline bubble ratio."""
    if not HAS_MATPLOTLIB:
        return None

    iterations = []
    bubble_ratios = []
    bubble_times = []

    for iter_data in analysis.iterations:
        if "pipeline_bubble" not in iter_data.metrics:
            continue
        data = iter_data.metrics["pipeline_bubble"]
        if "error" in data or data.get("method") == "none":
            continue

        iterations.append(iter_data.iteration)
        bubble_ratios.append(data.get("bubble_ratio", 0) * 100)
        bubble_times.append(data.get("bubble_time_ms", 0))

    if not iterations:
        return None

    colors = get_color_palette()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Bubble ratio as percentage
    bars1 = ax1.bar(iterations, bubble_ratios, color=colors["quaternary"], alpha=0.8)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Bubble Ratio (%)")
    ax1.set_title("Pipeline Bubble Ratio")
    ax1.set_ylim(0, max(bubble_ratios) * 1.3 if bubble_ratios else 10)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    for bar, val in zip(bars1, bubble_ratios):
        ax1.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Add reference line for ideal
    ax1.axhline(y=5, color=colors["success"], linestyle="--", alpha=0.7, label="Target (<5%)")
    ax1.legend()

    # Right: Bubble time in ms
    bars2 = ax2.bar(iterations, bubble_times, color=colors["tertiary"], alpha=0.8)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Bubble Time (ms)")
    ax2.set_title("Pipeline Bubble Time")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    for bar, val in zip(bars2, bubble_times):
        ax2.annotate(
            f"{val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle(
        f"Pipeline Bubble Analysis - {analysis.workload_name}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    filename = output_dir / "pipeline_bubble.png"
    plt.savefig(filename)
    plt.close()

    return str(filename)


def plot_straggler_lag(analysis: WorkloadAnalysis, output_dir: Path) -> str | None:
    """Plot straggler lag metrics."""
    if not HAS_MATPLOTLIB:
        return None

    iterations = []
    mean_lags = []
    p50_lags = []
    p95_lags = []
    max_lags = []

    for iter_data in analysis.iterations:
        if "straggler_lag" not in iter_data.metrics:
            continue
        data = iter_data.metrics["straggler_lag"]
        if "error" in data or data.get("num_collectives", 0) == 0:
            continue

        iterations.append(iter_data.iteration)
        mean_lags.append(data.get("mean_lag_us", 0) / 1000)  # Convert to ms
        p50_lags.append(data.get("p50_lag_us", 0) / 1000)
        p95_lags.append(data.get("p95_lag_us", 0) / 1000)
        max_lags.append(data.get("max_lag_us", 0) / 1000)

    if not iterations:
        return None

    colors = get_color_palette()
    _fig, ax = plt.subplots(figsize=(12, 7))

    x = range(len(iterations))
    width = 0.2

    ax.bar(
        [i - 1.5 * width for i in x],
        mean_lags,
        width,
        label="Mean",
        color=colors["primary"],
        alpha=0.8,
    )
    ax.bar(
        [i - 0.5 * width for i in x],
        p50_lags,
        width,
        label="P50 (Median)",
        color=colors["secondary"],
        alpha=0.8,
    )
    ax.bar(
        [i + 0.5 * width for i in x],
        p95_lags,
        width,
        label="P95",
        color=colors["tertiary"],
        alpha=0.8,
    )
    ax.bar(
        [i + 1.5 * width for i in x],
        max_lags,
        width,
        label="Max",
        color=colors["quaternary"],
        alpha=0.8,
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Lag (ms)")
    ax.set_title(f"Straggler Lag Distribution - {analysis.workload_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in iterations])
    ax.legend()

    plt.tight_layout()
    filename = output_dir / "straggler_lag.png"
    plt.savefig(filename)
    plt.close()

    return str(filename)


@dataclass
class TrafficDistributionData:
    """Data for traffic distribution plotting."""

    iterations: list[int]
    dp_fracs: list[float]
    tp_fracs: list[float]
    pp_fracs: list[float]
    ep_fracs: list[float]
    unknown_fracs: list[float]


def _extract_traffic_distribution_data(
    analysis: WorkloadAnalysis,
) -> TrafficDistributionData | None:
    """Extract traffic distribution data from workload analysis."""
    iterations: list[int] = []
    dp_fracs: list[float] = []
    tp_fracs: list[float] = []
    pp_fracs: list[float] = []
    ep_fracs: list[float] = []
    unknown_fracs: list[float] = []

    for iter_data in analysis.iterations:
        if "traffic_distribution" not in iter_data.metrics:
            continue
        data = iter_data.metrics["traffic_distribution"]
        if "error" in data:
            continue

        fracs = data.get("fractions", {})
        iterations.append(iter_data.iteration)
        dp_fracs.append(fracs.get("dp", 0) * 100)
        tp_fracs.append(fracs.get("tp", 0) * 100)
        pp_fracs.append(fracs.get("pp", 0) * 100)
        ep_fracs.append(fracs.get("ep", 0) * 100)
        unknown_fracs.append(fracs.get("unknown", 0) * 100)

    if not iterations:
        return None

    return TrafficDistributionData(
        iterations=iterations,
        dp_fracs=dp_fracs,
        tp_fracs=tp_fracs,
        pp_fracs=pp_fracs,
        ep_fracs=ep_fracs,
        unknown_fracs=unknown_fracs,
    )


def _plot_traffic_stacked_bar(
    ax: Any,
    data: TrafficDistributionData,
    colors: dict[str, str],
) -> None:
    """Plot stacked bar chart for traffic distribution."""
    x = range(len(data.iterations))
    width = 0.6
    bottom = [0.0] * len(data.iterations)

    bar_configs = [
        (data.pp_fracs, "Pipeline (PP)", colors["pp"], 0.9),
        (data.tp_fracs, "Tensor (TP)", colors["tp"], 0.9),
        (data.dp_fracs, "Data (DP)", colors["dp"], 0.9),
        (data.ep_fracs, "Expert (EP)", colors["ep"], 0.9),
        (data.unknown_fracs, "Unknown", colors["unknown"], 0.7),
    ]

    for fracs, label, color, alpha in bar_configs:
        if any(fracs):
            ax.bar(x, fracs, width, label=label, color=color, bottom=bottom, alpha=alpha)
            bottom = [b + v for b, v in zip(bottom, fracs)]

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Traffic Distribution (%)")
    ax.set_title("Traffic by Parallelism Type")
    ax.set_xticks(x)
    ax.set_xticklabels(data.iterations)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")


def _plot_traffic_pie_chart(
    ax: Any,
    data: TrafficDistributionData,
    colors: dict[str, str],
) -> None:
    """Plot pie chart for average traffic distribution."""
    avg_pp = sum(data.pp_fracs) / len(data.pp_fracs) if data.pp_fracs else 0
    avg_tp = sum(data.tp_fracs) / len(data.tp_fracs) if data.tp_fracs else 0
    avg_dp = sum(data.dp_fracs) / len(data.dp_fracs) if data.dp_fracs else 0
    avg_ep = sum(data.ep_fracs) / len(data.ep_fracs) if data.ep_fracs else 0
    avg_unknown = sum(data.unknown_fracs) / len(data.unknown_fracs) if data.unknown_fracs else 0

    pie_data = [
        (avg_pp, f"PP ({avg_pp:.1f}%)", colors["pp"]),
        (avg_tp, f"TP ({avg_tp:.1f}%)", colors["tp"]),
        (avg_dp, f"DP ({avg_dp:.1f}%)", colors["dp"]),
        (avg_ep, f"EP ({avg_ep:.1f}%)", colors["ep"]),
        (avg_unknown, f"Unknown ({avg_unknown:.1f}%)", colors["unknown"]),
    ]

    sizes = []
    labels = []
    pie_colors = []

    for avg, label, color in pie_data:
        if avg > 0.1:
            sizes.append(avg)
            labels.append(label)
            pie_colors.append(color)

    if sizes:
        ax.pie(
            sizes,
            labels=labels,
            colors=pie_colors,
            autopct="",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax.set_title("Average Traffic Distribution")


def plot_traffic_distribution(analysis: WorkloadAnalysis, output_dir: Path) -> str | None:
    """Plot traffic distribution by parallelism type."""
    if not HAS_MATPLOTLIB:
        return None

    traffic_data = _extract_traffic_distribution_data(analysis)
    if traffic_data is None:
        return None

    colors = get_color_palette()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    _plot_traffic_stacked_bar(ax1, traffic_data, colors)
    _plot_traffic_pie_chart(ax2, traffic_data, colors)

    fig.suptitle(
        f"Communication Traffic Analysis - {analysis.workload_name}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    filename = output_dir / "traffic_distribution.png"
    plt.savefig(filename)
    plt.close()

    return str(filename)


def plot_coll_call_breakdown(analysis: WorkloadAnalysis, output_dir: Path) -> str | None:
    """Plot collective call breakdown."""
    if not HAS_MATPLOTLIB:
        return None

    # Aggregate call types across all iterations
    call_types: dict[str, int] = {}
    total_calls_per_iter = []
    iterations = []

    for iter_data in analysis.iterations:
        if "coll_call_num" not in iter_data.metrics:
            continue
        data = iter_data.metrics["coll_call_num"]
        if "error" in data:
            continue

        iterations.append(iter_data.iteration)
        total_calls_per_iter.append(data.get("total_calls", 0))

        for coll_type, count in data.get("calls_by_type", {}).items():
            call_types[coll_type] = call_types.get(coll_type, 0) + count

    if not iterations:
        return None

    colors = get_color_palette()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Total calls per iteration
    bars = ax1.bar(iterations, total_calls_per_iter, color=colors["primary"], alpha=0.8)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Number of Calls")
    ax1.set_title("Total Communication Calls")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    for bar, val in zip(bars, total_calls_per_iter):
        ax1.annotate(
            f"{val}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Right: Pie chart of call types
    if call_types:
        type_colors = {
            "SendRecv": colors["pp"],
            "AllReduce": colors["dp"],
            "ReduceScatter": colors["tertiary"],
            "AllGather": colors["secondary"],
            "AllToAll": colors["ep"],
            "Broadcast": colors["success"],
            "Other": colors["neutral"],
        }

        sizes = list(call_types.values())
        labels = [f"{k}\n({v})" for k, v in call_types.items()]
        pie_colors = [type_colors.get(k, colors["neutral"]) for k in call_types]

        ax2.pie(
            sizes,
            labels=labels,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax2.set_title("Call Type Distribution")

    fig.suptitle(
        f"Collective Calls Analysis - {analysis.workload_name}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    filename = output_dir / "coll_calls.png"
    plt.savefig(filename)
    plt.close()

    return str(filename)


def plot_summary_dashboard(analysis: WorkloadAnalysis, output_dir: Path) -> str | None:
    """Create a summary dashboard with key metrics."""
    if not HAS_MATPLOTLIB:
        return None

    fig = plt.figure(figsize=(16, 12))

    colors = get_color_palette()

    # Extract summary data
    throughputs = []
    iter_times = []
    bubble_ratios = []
    overlap_ratios = []
    iterations = []

    for iter_data in analysis.iterations:
        iterations.append(iter_data.iteration)

        if "throughput_tokens" in iter_data.metrics:
            data = iter_data.metrics["throughput_tokens"]
            throughputs.append(data.get("throughput_tokens_per_s", 0) / 1000)

        if "iter_time" in iter_data.metrics:
            data = iter_data.metrics["iter_time"]
            iter_times.append(data.get("avg_iter_time_ms", 0))

        if "pipeline_bubble" in iter_data.metrics:
            data = iter_data.metrics["pipeline_bubble"]
            bubble_ratios.append(data.get("bubble_ratio", 0) * 100)

        if "comm_comp_overlap" in iter_data.metrics:
            data = iter_data.metrics["comm_comp_overlap"]
            overlap_ratios.append(data.get("overlap_ratio_of_comm", 0) * 100)

    # Create 2x2 grid
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Plot 1: Throughput
    if throughputs:
        ax1.bar(iterations[: len(throughputs)], throughputs, color=colors["success"], alpha=0.8)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("K tokens/sec")
        ax1.set_title("Throughput", fontweight="bold")
        avg = sum(throughputs) / len(throughputs)
        ax1.axhline(y=avg, color=colors["quaternary"], linestyle="--", alpha=0.7)
        ax1.annotate(
            f"Avg: {avg:.1f}K",
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            fontweight="bold",
            color=colors["quaternary"],
        )

    # Plot 2: Iteration Time
    if iter_times:
        ax2.bar(iterations[: len(iter_times)], iter_times, color=colors["primary"], alpha=0.8)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Time (ms)")
        ax2.set_title("Iteration Time", fontweight="bold")
        avg = sum(iter_times) / len(iter_times)
        ax2.axhline(y=avg, color=colors["quaternary"], linestyle="--", alpha=0.7)
        ax2.annotate(
            f"Avg: {avg:.1f}ms",
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            fontweight="bold",
            color=colors["quaternary"],
        )

    # Plot 3: Pipeline Bubble
    if bubble_ratios:
        ax3.bar(
            iterations[: len(bubble_ratios)], bubble_ratios, color=colors["tertiary"], alpha=0.8
        )
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Bubble Ratio (%)")
        ax3.set_title("Pipeline Bubble", fontweight="bold")
        ax3.axhline(y=5, color=colors["success"], linestyle="--", alpha=0.7, label="Target <5%")
        ax3.legend(loc="upper right", fontsize=9)

    # Plot 4: Comm/Comp Overlap
    if overlap_ratios:
        ax4.bar(
            iterations[: len(overlap_ratios)], overlap_ratios, color=colors["secondary"], alpha=0.8
        )
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Overlap Ratio (%)")
        ax4.set_title("Comm/Comp Overlap", fontweight="bold")
        ax4.set_ylim(0, 100)
        ax4.axhline(y=80, color=colors["success"], linestyle="--", alpha=0.7, label="Good >80%")
        ax4.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        f"Performance Dashboard - {analysis.workload_name}", fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    filename = output_dir / "summary_dashboard.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    return str(filename)


# =============================================================================
# Export Functions
# =============================================================================


def export_csv(analysis: WorkloadAnalysis, output_dir: Path) -> str:
    """Export analysis to CSV format."""
    csv_path = output_dir / "metrics_summary.csv"

    # Flatten metrics for CSV
    rows = []
    headers = ["iteration"]

    # Collect all metric keys
    metric_keys = set()
    for iter_data in analysis.iterations:
        for metric_name, metric_data in iter_data.metrics.items():
            if isinstance(metric_data, dict) and "error" not in metric_data:
                for key in metric_data:
                    if not isinstance(metric_data[key], (dict, list)):
                        metric_keys.add(f"{metric_name}_{key}")

    headers.extend(sorted(metric_keys))

    for iter_data in analysis.iterations:
        row = {"iteration": iter_data.iteration}
        for metric_name, metric_data in iter_data.metrics.items():
            if isinstance(metric_data, dict) and "error" not in metric_data:
                for key, value in metric_data.items():
                    if not isinstance(value, (dict, list)):
                        row[f"{metric_name}_{key}"] = value
        rows.append(row)

    with csv_path.open("w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            values = [str(row.get(h, "")) for h in headers]
            f.write(",".join(values) + "\n")

    return str(csv_path)


def export_json(analysis: WorkloadAnalysis, output_dir: Path) -> str:
    """Export analysis to JSON format."""
    json_path = output_dir / "analysis.json"

    with json_path.open("w") as f:
        json.dump(analysis.to_dict(), f, indent=2)

    return str(json_path)


def generate_html_report(
    analysis: WorkloadAnalysis,
    output_dir: Path,
    plots: dict[str, str | None],
) -> str:
    """Generate an HTML report with all visualizations."""
    html_path = output_dir / "report.html"

    # Calculate summary statistics
    avg_throughput: float = 0.0
    avg_iter_time: float = 0.0
    avg_bubble: float = 0.0
    avg_overlap: float = 0.0
    total_calls = 0

    for iter_data in analysis.iterations:
        if "throughput_tokens" in iter_data.metrics:
            avg_throughput += iter_data.metrics["throughput_tokens"].get(
                "throughput_tokens_per_s", 0
            )
        if "iter_time" in iter_data.metrics:
            avg_iter_time += iter_data.metrics["iter_time"].get("avg_iter_time_ms", 0)
        if "pipeline_bubble" in iter_data.metrics:
            avg_bubble += iter_data.metrics["pipeline_bubble"].get("bubble_ratio", 0)
        if "comm_comp_overlap" in iter_data.metrics:
            avg_overlap += iter_data.metrics["comm_comp_overlap"].get("overlap_ratio_of_comm", 0)
        if "coll_call_num" in iter_data.metrics:
            total_calls += iter_data.metrics["coll_call_num"].get("total_calls", 0)

    n = len(analysis.iterations) or 1
    avg_throughput /= n
    avg_iter_time /= n
    avg_bubble /= n
    avg_overlap /= n

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CCL-Bench Analysis Report - {analysis.workload_name}</title>
    <style>
        :root {{
            --primary: #2E86AB;
            --secondary: #A23B72;
            --success: #3A7D44;
            --warning: #F18F01;
            --danger: #C73E1D;
            --dark: #1a1a2e;
            --light: #f8f9fa;
            --gray: #6c757d;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            min-height: 100vh;
            color: var(--dark);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        header {{
            background: linear-gradient(135deg, var(--dark) 0%, #16213e 100%);
            color: white;
            padding: 3rem 2rem;
            margin-bottom: 2rem;
            border-radius: 1rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}

        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }}

        header .subtitle {{
            opacity: 0.8;
            font-size: 1.1rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .stat-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        }}

        .stat-card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
        }}

        .stat-card .label {{
            color: var(--gray);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}

        .stat-card.success .value {{ color: var(--success); }}
        .stat-card.warning .value {{ color: var(--warning); }}
        .stat-card.danger .value {{ color: var(--danger); }}

        section {{
            background: white;
            border-radius: 1rem;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}

        section h2 {{
            color: var(--dark);
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 3px solid var(--primary);
            font-size: 1.5rem;
        }}

        .plot-container {{
            text-align: center;
            margin: 1rem 0;
        }}

        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 0.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}

        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}

        th, td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}

        th {{
            background: var(--light);
            font-weight: 600;
            color: var(--dark);
        }}

        tr:hover {{
            background: #f8f9fa;
        }}

        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--gray);
            font-size: 0.9rem;
        }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            font-weight: 600;
        }}

        .badge-success {{ background: #d4edda; color: #155724; }}
        .badge-warning {{ background: #fff3cd; color: #856404; }}
        .badge-danger {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 CCL-Bench Analysis Report</h1>
            <p class="subtitle">Workload: {analysis.workload_name} | Iterations Analyzed: {len(analysis.iterations)}</p>
        </header>

        <div class="summary-grid">
            <div class="stat-card success">
                <div class="value">{avg_throughput / 1000:.1f}K</div>
                <div class="label">Avg Throughput (tokens/sec)</div>
            </div>
            <div class="stat-card">
                <div class="value">{avg_iter_time:.1f}</div>
                <div class="label">Avg Iteration Time (ms)</div>
            </div>
            <div class="stat-card {"success" if avg_bubble < 0.05 else "warning" if avg_bubble < 0.1 else "danger"}">
                <div class="value">{avg_bubble * 100:.1f}%</div>
                <div class="label">Avg Pipeline Bubble</div>
            </div>
            <div class="stat-card {"success" if avg_overlap > 0.8 else "warning" if avg_overlap > 0.5 else "danger"}">
                <div class="value">{avg_overlap * 100:.1f}%</div>
                <div class="label">Avg Comm/Comp Overlap</div>
            </div>
            <div class="stat-card">
                <div class="value">{total_calls}</div>
                <div class="label">Total Collective Calls</div>
            </div>
        </div>
"""

    # Add summary dashboard
    summary_dashboard_path = plots.get("summary_dashboard")
    if summary_dashboard_path:
        html_content += f"""
        <section>
            <h2>📈 Performance Dashboard</h2>
            <div class="plot-container">
                <img src="{Path(summary_dashboard_path).name}" alt="Summary Dashboard">
            </div>
        </section>
"""

    # Add individual plots
    plot_sections = [
        (
            "throughput",
            "🚀 Throughput Analysis",
            "Training throughput measured in tokens per second.",
        ),
        ("iter_time", "⏱️ Iteration Time", "Wall-clock time for each training iteration."),
        (
            "comm_comp_overlap",
            "🔄 Communication/Computation Overlap",
            "How well communication is hidden behind computation.",
        ),
        (
            "pipeline_bubble",
            "🫧 Pipeline Bubble Analysis",
            "Idle time in pipeline parallel execution.",
        ),
        ("straggler_lag", "🐢 Straggler Lag", "Synchronization overhead between ranks."),
        (
            "traffic_distribution",
            "📡 Traffic Distribution",
            "Communication traffic breakdown by parallelism type.",
        ),
        (
            "coll_calls",
            "📞 Collective Calls",
            "Number and types of collective communication operations.",
        ),
    ]

    html_content += """
        <section>
            <h2>📊 Detailed Metrics</h2>
            <div class="plot-grid">
"""

    for plot_key, title, description in plot_sections:
        plot_path = plots.get(plot_key)
        if plot_path:
            html_content += f"""
                <div class="plot-container">
                    <h3>{title}</h3>
                    <p style="color: var(--gray); font-size: 0.9rem; margin-bottom: 1rem;">{description}</p>
                    <img src="{Path(plot_path).name}" alt="{title}">
                </div>
"""

    html_content += """
            </div>
        </section>
"""

    # Add raw data table
    html_content += """
        <section>
            <h2>📋 Raw Data by Iteration</h2>
            <table>
                <thead>
                    <tr>
                        <th>Iteration</th>
                        <th>Throughput (K tok/s)</th>
                        <th>Iter Time (ms)</th>
                        <th>Bubble Ratio</th>
                        <th>Overlap Ratio</th>
                        <th>Comm Calls</th>
                    </tr>
                </thead>
                <tbody>
"""

    for iter_data in analysis.iterations:
        throughput = (
            iter_data.metrics.get("throughput_tokens", {}).get("throughput_tokens_per_s", 0) / 1000
        )
        iter_time = iter_data.metrics.get("iter_time", {}).get("avg_iter_time_ms", 0)
        bubble = iter_data.metrics.get("pipeline_bubble", {}).get("bubble_ratio", 0) * 100
        overlap = (
            iter_data.metrics.get("comm_comp_overlap", {}).get("overlap_ratio_of_comm", 0) * 100
        )
        calls = iter_data.metrics.get("coll_call_num", {}).get("total_calls", 0)

        bubble_class = (
            "badge-success" if bubble < 5 else "badge-warning" if bubble < 10 else "badge-danger"
        )
        overlap_class = (
            "badge-success" if overlap > 80 else "badge-warning" if overlap > 50 else "badge-danger"
        )

        html_content += f"""
                    <tr>
                        <td><strong>{iter_data.iteration}</strong></td>
                        <td>{throughput:.1f}K</td>
                        <td>{iter_time:.1f}</td>
                        <td><span class="badge {bubble_class}">{bubble:.1f}%</span></td>
                        <td><span class="badge {overlap_class}">{overlap:.1f}%</span></td>
                        <td>{calls}</td>
                    </tr>
"""

    html_content += f"""
                </tbody>
            </table>
        </section>

        <footer>
            <p>Generated by CCL-Bench Analysis Tool | {Path(analysis.trace_base).name}</p>
        </footer>
    </div>
</body>
</html>
"""

    with html_path.open("w") as f:
        f.write(html_content)

    return str(html_path)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main entry point for the workload analysis tool."""
    parser = argparse.ArgumentParser(
        description="CCL-Bench Workload Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trace-base",
        type=str,
        required=True,
        help="Base directory containing iteration_* subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./analysis_output",
        help="Output directory for plots and reports",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Workload name (auto-detected from path if not provided)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Specific metrics to compute (default: all)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze workload
    analysis = analyze_workload(
        trace_base=args.trace_base,
        workload_name=args.name,
        metrics_to_run=args.metrics,
    )

    if not analysis.iterations:
        print("No data collected. Exiting.", file=sys.stderr)
        return 1

    print()
    print("=" * 60)
    print("Generating outputs...")
    print("=" * 60)

    # Export data
    csv_path = export_csv(analysis, output_dir)
    print(f"  CSV exported: {csv_path}")

    json_path = export_json(analysis, output_dir)
    print(f"  JSON exported: {json_path}")

    # Generate plots
    plots = {}
    if not args.no_plots and HAS_MATPLOTLIB:
        setup_plot_style()

        print("  Generating plots...")
        plots["iter_time"] = plot_iteration_time(analysis, output_dir)
        plots["throughput"] = plot_throughput(analysis, output_dir)
        plots["comm_comp_overlap"] = plot_comm_comp_overlap(analysis, output_dir)
        plots["pipeline_bubble"] = plot_pipeline_bubble(analysis, output_dir)
        plots["straggler_lag"] = plot_straggler_lag(analysis, output_dir)
        plots["traffic_distribution"] = plot_traffic_distribution(analysis, output_dir)
        plots["coll_calls"] = plot_coll_call_breakdown(analysis, output_dir)
        plots["summary_dashboard"] = plot_summary_dashboard(analysis, output_dir)

        generated = [k for k, v in plots.items() if v]
        print(f"    Generated {len(generated)} plots: {', '.join(generated)}")
    elif not HAS_MATPLOTLIB:
        print("  Skipping plots (matplotlib not available)")

    # Generate HTML report
    html_path = generate_html_report(analysis, output_dir, plots)
    print(f"  HTML report: {html_path}")

    print()
    print("=" * 60)
    print(f"Analysis complete! Output saved to: {output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
