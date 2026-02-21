from ._common import collect_kernel_durations

# This sums the NCCL kernel durations for each rank and compares the slowest to the fastest.
# Straggler Delay (First element in tuple) tells us how much extra time the slowest rank took relative to max time
# Straggler Slowdown (Second element in tuple) is a ratio of the slowest to fastest total kernel execution time.
def metric_cal(directory: str) -> tuple[float, float]:
    trace_files, rank_durations = collect_kernel_durations(directory)
    if not trace_files:
        print(f"No trace files found under: {directory}")
        return 0.0

    if not rank_durations or len(rank_durations) <= 1:
        return 0.0

    rank_total_durations = {rank: sum(durations) for rank, durations in rank_durations.items()}
    
    if len(rank_total_durations) <= 1:
        return 0.0
    
    min_total = min(rank_total_durations.values())
    max_total = max(rank_total_durations.values())
    
    if max_total <= 0:
        return 0.0
    
    return ((max_total - min_total) / max_total), (max_total / min_total)
