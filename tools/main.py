"""CCL-Bench Metrics Tool.

Usage:
    python -m tools.main --trace <trace_directory> --metric <metric_name>
    ccl-metrics --trace <trace_directory> --metric <metric_name>

Available metrics:
    - coll_call_num: Number of NCCL communication calls (summed across all ranks)
    - throughput_tokens: Training throughput in tokens/sec
    - iter_time: Average iteration wall-clock time (ms)
    - comm_comp_overlap: Communication/computation overlap ratio
    - pipeline_bubble: Pipeline bubble ratio
    - traffic_distribution: Traffic distribution by parallelism type (returns dict)
    - straggler_lag: Straggler lag metric (normalized)

Examples:
    # Calculate communication call count
    ccl-metrics --trace ./traces/llama3_8b_tp --metric coll_call_num

    # Get traffic distribution as JSON
    ccl-metrics --trace ./traces/qwen3_32b_3d --metric traffic_distribution --output-json

    # Specify profile mode (torch or nsys)
    ccl-metrics --trace ./traces/llama3_8b --metric iter_time --profile-mode torch

    # List all available metrics
    ccl-metrics --list-metrics
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import importlib
import inspect
import json
import sys
from typing import Any, cast


# Type alias for metric functions
# Metrics can accept directory, profile_mode, and optional kwargs
MetricResult = int | float | dict[str, Any]
MetricFunction = Callable[..., MetricResult]

# Valid profile modes
PROFILE_MODES = ("torch", "nsys", "auto")


# Mapping from metric name -> module path (relative to tools package)
# Each module must export a `metric_cal(directory: str)` function
_METRIC_MODULES: dict[str, str] = {
    "coll_call_num": "coll_call_num.coll_call_num",
    "comm_comp_overlap": "comm_comp_overlap.comm_comp_overlap",
    "iter_time": "iter_time.iter_time",
    "pipeline_bubble": "pipeline_bubble.pipeline_bubble",
    "straggler_lag": "straggler_lag.straggler_lag",
    "throughput_tokens": "throughput_tokens.throughput_tokens",
    "traffic_distribution": "traffic_distribution.traffic_distribution",
}

# Sorted list for CLI display
AVAILABLE_METRICS = sorted(_METRIC_MODULES.keys())


def get_metric_function(metric_name: str) -> MetricFunction:
    """Get the metric calculation function for a given metric name.

    Uses dynamic import to load the appropriate module on demand,
    reducing startup time and memory usage.

    Args:
        metric_name: Name of the metric to calculate.

    Returns:
        The metric_cal function from the appropriate module.

    Raises:
        ValueError: If metric_name is not in the registry.
        AttributeError: If the module doesn't define metric_cal.
    """
    if metric_name not in _METRIC_MODULES:
        raise ValueError(
            f"Unsupported metric: '{metric_name}'\n"
            f"Available metrics: {', '.join(AVAILABLE_METRICS)}"
        )

    module_path = _METRIC_MODULES[metric_name]
    module = importlib.import_module(module_path)

    if not hasattr(module, "metric_cal"):
        raise AttributeError(f"Module '{module_path}' does not define metric_cal(directory: str)")

    return cast("MetricFunction", module.metric_cal)


def main() -> None:
    """Main entry point for the metrics tool.

    Parses command-line arguments and dispatches to the appropriate
    metric calculation function.
    """
    parser = argparse.ArgumentParser(
        description="CCL-Bench Metrics Tool - Calculate metrics from trace data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available metrics:\n  " + "\n  ".join(AVAILABLE_METRICS),
    )
    parser.add_argument(
        "--trace",
        type=str,
        help="Path to the trace directory containing kineto/torch_et traces",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=AVAILABLE_METRICS,
        help="Name of the metric to calculate",
    )
    parser.add_argument(
        "--profile-mode",
        type=str,
        choices=PROFILE_MODES,
        default="auto",
        help="Profile mode: 'torch' (Kineto/PyTorch), 'nsys' (Nsight Systems), or 'auto' (detect)",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Force JSON output (auto-enabled for dict results like traffic_distribution)",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List all available metrics and exit",
    )

    args = parser.parse_args()

    # Handle list-metrics flag first
    if args.list_metrics:
        print("Available metrics:")
        for metric in AVAILABLE_METRICS:
            print(f"  - {metric}")
        sys.exit(0)

    # Validate required arguments for metric calculation
    if not args.trace or not args.metric:
        parser.error("--trace and --metric are required when not using --list-metrics")

    # Get the metric function and calculate
    try:
        metric_cal_func = get_metric_function(args.metric)

        # Pass profile_mode to metric functions that support it
        # Use inspect to check if the function accepts profile_mode
        sig = inspect.signature(metric_cal_func)
        if "profile_mode" in sig.parameters:
            result = metric_cal_func(args.trace, profile_mode=args.profile_mode)
        else:
            result = metric_cal_func(args.trace)

        # Output the result
        if args.output_json or isinstance(result, dict):
            print(json.dumps(result, indent=2))
        else:
            print(result)

    except FileNotFoundError as e:
        print(f"Error: Trace directory or file not found: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error calculating metric '{args.metric}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
