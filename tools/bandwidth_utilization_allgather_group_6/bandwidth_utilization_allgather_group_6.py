import pandas as pd

from bandwidth_utilization_group_6_common import (
    DTYPE_BYTES,
    add_collective_bandwidth,
    extract_nccl_collective_events,
    extract_tpu_collective_events,
    is_nccl_collective_kernel,
    load_events,
    metric_cal_for_collective,
)


_load_events = load_events


def _is_allgather_kernel(event: dict) -> bool:
    """Check if an event is an nccl:all_gather kernel event."""
    return is_nccl_collective_kernel(
        event,
        kernel_prefixes=("ncclDevKernel_AllGather", "ncclKernel_AllGather"),
        collective_names=("all_gather", "allgather"),
    )


def _extract_allgather_events(trace_path: str) -> pd.DataFrame:
    """Extract all gather kernel events from a trace JSON file."""
    return extract_nccl_collective_events(trace_path, _is_allgather_kernel)


def _get_bandwidth_utilization(df: pd.DataFrame, bandwidth: float = 600.0) -> pd.DataFrame:
    """Calculate bandwidth utilization for all_gather events.

    For all_gather, the algorithm factor is (N-1)/N where N is the group size.
    Effective bandwidth = data_size * algo_factor / duration.
    Utilization = effective_bandwidth / peak_bandwidth.
    """
    return add_collective_bandwidth(df, algo_factor_multiplier=lambda n: (n - 1) / n, bandwidth=bandwidth)


def _extract_allgather_tpu_events(trace_path: str) -> pd.DataFrame:
    return extract_tpu_collective_events(
        trace_path,
        hlo_category="all-gather",
        group_key=lambda event: event.get("name", ""),
    )


def metric_cal(directory: str) -> float:
    """
    Calculate the median allgather bandwidth (GB/s) from *trace.json files.

    Finds all nccl:all_gather kernel events across all rank trace files in the
    directory and returns the median effective bandwidth in GB/s.

    Args:
        directory (str): The directory path containing the trace JSON files.

    Returns:
        float: The median allgather bandwidth in GB/s, or float("nan") if no
               allgather events are found.
    """
    return metric_cal_for_collective(
        directory,
        collective_name="allgather",
        gpu_extractor=_extract_allgather_events,
        tpu_extractor=_extract_allgather_tpu_events,
        algo_factor_multiplier=lambda n: (n - 1) / n,
    )
