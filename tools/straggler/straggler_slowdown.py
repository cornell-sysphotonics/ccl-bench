from ._common import collect_comm_windows


def metric_cal(directory: str) -> float:
    """
    Calculate the straggler slowdown as the ratio of the slowest to fastest communication window durations.

    Args:
        directory (str): Path to the directory containing kineto traces.

    Returns:
        float: Slowdown factor (>= 1 when data is available, 0 when undefined).
    """

    trace_files, device_windows = collect_comm_windows(directory)
    if not trace_files:
        print(f"No kineto_trace_<rank>.json files found under: {directory}")
        return 0.0

    durations = [window[1] - window[0] for window in device_windows.values() if window[1] > window[0]]
    if len(durations) <= 1:
        return 0.0

    min_duration = min(durations)
    max_duration = max(durations)

    if min_duration <= 0:
        return 0.0

    return max_duration / min_duration
