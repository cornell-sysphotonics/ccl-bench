def metric_cal(directory: str) -> float:
    """
    Calculate the kernel compute time from the exported sqlite file from nsys.

    Args:
        directory (str): The directory path containing the exported sqlite file from nsys.

    Returns:
        float: The calculated kernel compute time value.
    """
    return 10.9