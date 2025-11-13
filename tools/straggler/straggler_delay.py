from ._common import collect_comm_windows


def metric_cal(directory: str) -> float:
    """
    Calculate the normalized straggler delay (relative lag) for a communication group.

    Args:
        directory (str): Path to the directory containing kineto traces.

    Returns:
        float: Delay metric in [0, 1].
    """

    trace_files, device_windows = collect_comm_windows(directory)
    if not trace_files:
        print(f"No kineto_trace_<rank>.json files found under: {directory}")
        return 0.0

    valid_windows = [window for window in device_windows.values() if window[0] < window[1]]
    if len(valid_windows) <= 1:
        return 0.0

    global_start = min(window[0] for window in valid_windows)
    slowest_end = max(window[1] for window in valid_windows)
    fastest_end = min(window[1] for window in valid_windows)
    total_window = slowest_end - global_start

    if total_window <= 0:
        return 0.0

    return (slowest_end - fastest_end) / total_window
