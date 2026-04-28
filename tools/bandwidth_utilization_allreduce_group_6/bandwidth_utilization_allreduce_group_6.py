import pandas as pd

from bandwidth_utilization_group_6_common import (
    DTYPE_BYTES,
    add_bandwidth_utilization,
    extract_nccl_collective_events,
    extract_tpu_collective_events,
    get_bandwidth_utilization_from_trace,
    is_nccl_collective_kernel,
    load_events,
    metric_cal_for_collective,
)


_load_events = load_events


def _is_allreduce_kernel(event: dict) -> bool:
    """Check if an event is an nccl:all_reduce kernel event."""
    return is_nccl_collective_kernel(
        event,
        kernel_prefixes=("ncclDevKernel_AllReduce", "ncclKernel_AllReduce"),
        collective_names=("all_reduce", "allreduce"),
    )


def _extract_allreduce_events(trace_path: str) -> pd.DataFrame:
    """Extract all reduce kernel events from a trace JSON file."""
    return extract_nccl_collective_events(trace_path, _is_allreduce_kernel)


def _get_bandwidth_utilization(df: pd.DataFrame, bandwidth: float = 600.0) -> pd.DataFrame:
    """Calculate bandwidth utilization for all_reduce events.

    For all_reduce, the algorithm factor is 2*(N-1)/N where N is the group size.
    Effective bandwidth = data_size * algo_factor / duration.
    Utilization = effective_bandwidth / peak_bandwidth.
    """
    return add_bandwidth_utilization(df, algo_factor_multiplier=2, bandwidth=bandwidth)


def _get_bandwidth_utilization_from_trace(trace_path: str, bandwidth: float = 600.0) -> pd.DataFrame:
    return get_bandwidth_utilization_from_trace(
        trace_path,
        extractor=_extract_allreduce_events,
        algo_factor_multiplier=2,
        bandwidth=bandwidth,
    )


def _extract_allreduce_tpu_events(trace_path: str) -> pd.DataFrame:
    return extract_tpu_collective_events(
        trace_path,
        hlo_category="all-reduce",
        group_key=lambda event: event["args"].get("all_reduce_unique_id", ""),
    )


def _get_bandwidth_utilization_from_trace_tpu(trace_path: str, bandwidth: float = 600.0) -> pd.DataFrame:
    return get_bandwidth_utilization_from_trace(
        trace_path,
        extractor=_extract_allreduce_tpu_events,
        algo_factor_multiplier=2,
        bandwidth=bandwidth,
    )


def metric_cal(directory: str) -> float:
    """
    Calculate the median allreduce bandwidth (GB/s) from *trace.json files.

    Finds all nccl:all_reduce kernel events across all rank trace files in the
    directory and returns the median effective bandwidth in GB/s.

    Args:
        directory (str): The directory path containing the trace JSON files.

    Returns:
        float: The median allreduce bandwidth in GB/s, or float("nan") if no
               allreduce events are found.
    """
    return metric_cal_for_collective(
        directory,
        collective_name="allreduce",
        gpu_trace_parser=_get_bandwidth_utilization_from_trace,
        tpu_trace_parser=_get_bandwidth_utilization_from_trace_tpu,
    )
