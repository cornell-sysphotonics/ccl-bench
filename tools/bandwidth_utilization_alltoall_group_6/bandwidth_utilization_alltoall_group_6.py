import pandas as pd

from bandwidth_utilization_group_6_common import (
    DTYPE_BYTES,
    add_collective_bandwidth,
    extract_nccl_collective_events,
    extract_tpu_collective_events,
    load_events,
    metric_cal_for_collective,
)


_load_events = load_events


def _is_alltoall_kernel(event: dict) -> bool:
    """Check if an event is an nccl:all_to_all kernel event."""
    if event.get("ph") != "X" or event.get("cat") != "kernel":
        return False
    name = event.get("name", "")
    collective_name = event.get("args", {}).get("Collective name", "")
    if name.startswith("ncclDevKernel_SendRecv") or name.startswith("ncclKernel_SendRecv"):
        if collective_name in ("all_to_allv", "all_to_all", "alltoall"):
            return True
    return False


def _extract_alltoall_events(trace_path: str) -> pd.DataFrame:
    """Extract all_to_all kernel events from a trace JSON file."""
    return extract_nccl_collective_events(trace_path, _is_alltoall_kernel)


def _get_bandwidth_utilization(df: pd.DataFrame, bandwidth: float = 600.0) -> pd.DataFrame:
    """Calculate bandwidth utilization for all_to_all events.

    For all_to_all, the algorithm factor is (N-1)/N where N is the group size.
    Effective bandwidth = data_size * algo_factor / duration.
    Utilization = effective_bandwidth / peak_bandwidth.
    """
    return add_collective_bandwidth(df, algo_factor_multiplier=lambda n: (n - 1) / n, bandwidth=bandwidth)


def _extract_alltoall_tpu_events(trace_path: str) -> pd.DataFrame:
    return extract_tpu_collective_events(
        trace_path,
        hlo_category="all-to-all",
        group_key=lambda event: event.get("name", ""),
    )


def metric_cal(directory: str) -> float:
    """
    Calculate the median all_to_all bandwidth (GB/s) from trace JSON files.

    Finds all nccl:all_to_all kernel events across rank trace files in the
    directory and returns the median effective bandwidth in GB/s.

    Args:
        directory (str): The directory path containing the trace JSON files.

    Returns:
        float: The median all_to_all bandwidth in GB/s, or float("nan") if no
               all_to_all events are found.
    """
    return metric_cal_for_collective(
        directory,
        collective_name="alltoall",
        gpu_extractor=_extract_alltoall_events,
        tpu_extractor=_extract_alltoall_tpu_events,
        algo_factor_multiplier=lambda n: (n - 1) / n,
    )
