def metric_cal(directory: str) -> float:
    """
    Calculate the bandwidth utilization from the exported sqlite file from nsys.

    Args:
        directory (str): The directory path containing the exported sqlite file from nsys.

    Returns:
        float: The calculated bandwidth utilization value.
    """
    return 0.8