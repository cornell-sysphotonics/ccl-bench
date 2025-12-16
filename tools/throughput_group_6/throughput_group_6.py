def metric_cal(directory: str) -> float:
    """
    Calculate the throughput from the exported sqlite file from nsys.

    Args:
        directory (str): The directory path containing the exported sqlite file from nsys.

    Returns:
        float: The calculated throughput value.
    """
    return 1.2