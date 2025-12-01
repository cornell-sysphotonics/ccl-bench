"""CCL-Bench Metrics Tool.

Usage:
    python main.py --trace <trace_directory> --metric <metric_name>

Available metrics:
    - coll_call_num: Number of NCCL communication calls
    - throughput_tokens: Training throughput in tokens/sec
    - iter_time: Average iteration wall-clock time (ms)
    - comm_comp_overlap: Communication/computation overlap ratio
    - pipeline_bubble: Pipeline bubble ratio
    - traffic_distribution: Traffic distribution by parallelism type
    - straggler_lag: Straggler lag metric (normalized)
"""

import argparse
from collections.abc import Callable
import json
import sys


# Type alias for metric functions
MetricFunction = Callable[[str], int | float | dict[str, float]]

# Registry of available metrics
AVAILABLE_METRICS = [
    "coll_call_num",
    "throughput_tokens",
    "iter_time",
    "comm_comp_overlap",
    "pipeline_bubble",
    "traffic_distribution",
    "straggler_lag",
]


def get_metric_function(metric_name: str) -> MetricFunction:
    """Get the metric calculation function for a given metric name.

    Args:
        metric_name: Name of the metric to calculate.

    Returns:
        The metric_cal function from the appropriate module.

    Raises:
        ValueError: If metric_name is not supported.
    """
    if metric_name == "coll_call_num":
        from coll_call_num.coll_call_num import (  # noqa: PLC0415
            metric_cal as coll_call_num_metric,
        )

        return coll_call_num_metric
    if metric_name == "throughput_tokens":
        from throughput_tokens.throughput_tokens import (  # noqa: PLC0415
            metric_cal as throughput_tokens_metric,
        )

        return throughput_tokens_metric
    if metric_name == "iter_time":
        from iter_time.iter_time import (  # noqa: PLC0415
            metric_cal as iter_time_metric,
        )

        return iter_time_metric
    if metric_name == "comm_comp_overlap":
        from comm_comp_overlap.comm_comp_overlap import (  # noqa: PLC0415
            metric_cal as comm_comp_overlap_metric,
        )

        return comm_comp_overlap_metric
    if metric_name == "pipeline_bubble":
        from pipeline_bubble.pipeline_bubble import (  # noqa: PLC0415
            metric_cal as pipeline_bubble_metric,
        )

        return pipeline_bubble_metric
    if metric_name == "traffic_distribution":
        from traffic_distribution.traffic_distribution import (  # noqa: PLC0415
            metric_cal as traffic_distribution_metric,
        )

        return traffic_distribution_metric
    if metric_name == "straggler_lag":
        from straggler_lag.straggler_lag import (  # noqa: PLC0415
            metric_cal as straggler_lag_metric,
        )

        return straggler_lag_metric
    raise ValueError(
        f"Unsupported metric name: {metric_name}\nAvailable metrics: {', '.join(AVAILABLE_METRICS)}"
    )


def main() -> None:
    """Main entry point for the metrics tool."""
    parser = argparse.ArgumentParser(
        description="CCL-Bench Metrics Tool - Calculate metrics from trace data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available metrics:\n  " + "\n  ".join(AVAILABLE_METRICS),
    )
    parser.add_argument("--trace", type=str, help="Path to the trace directory")
    parser.add_argument(
        "--metric", type=str, choices=AVAILABLE_METRICS, help="Name of the metric to calculate"
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Output result as JSON (useful for dict/complex results)",
    )
    parser.add_argument(
        "--list-metrics", action="store_true", help="List all available metrics and exit"
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
