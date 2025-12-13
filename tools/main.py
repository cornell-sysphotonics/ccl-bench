"""CCL-Bench Metrics Tool."""

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


# =============================================================================
# Metric Registry
# =============================================================================
# Metric names follow the convention: <metric_name>_<group_number>
# Each module must export a `metric_cal(directory: str)` function
# Folder structure: tools/<metric_name>_<group>/metric.py

_METRIC_MODULES: dict[str, str] = {
    # Default metrics
    "coll_call_num": "tools.coll_call_num.coll_call_num",
    # Group 16 metrics
    "coll_call_num_16": "tools.coll_call_num_16.metric",
    "comm_comp_overlap_16": "tools.comm_comp_overlap_16.metric",
    "comm_volume_16": "tools.comm_volume_16.metric",
    "config_metadata_16": "tools.config_metadata_16.metric",
    "hardware_saturation_16": "tools.hardware_saturation_16.metric",
    "iter_time_16": "tools.iter_time_16.metric",
    "pipeline_bubble_16": "tools.pipeline_bubble_16.metric",
    "straggler_lag_16": "tools.straggler_lag_16.metric",
    "throughput_tokens_16": "tools.throughput_tokens_16.metric",
    "traffic_distribution_16": "tools.traffic_distribution_16.metric",
    "training_quality_16": "tools.training_quality_16.metric",
    "variability_metrics_16": "tools.variability_metrics_16.metric",
}

METRICS = sorted(_METRIC_MODULES.keys())


def get_metric_function(metric_name: str) -> MetricFunction:
    """Get the metric calculation function for a given metric name.

    Uses dynamic import to load the appropriate module on demand.

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
            f"Unsupported metric: '{metric_name}'\nAvailable metrics: {', '.join(METRICS)}"
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
        epilog="Available metrics:\n  " + "\n  ".join(METRICS),
    )
    parser.add_argument(
        "--trace",
        type=str,
        help="Path to the trace directory containing kineto/torch_et traces",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=METRICS,
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
        for metric in METRICS:
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
