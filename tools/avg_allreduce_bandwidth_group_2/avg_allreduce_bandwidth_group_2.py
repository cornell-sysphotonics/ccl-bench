"""
AllReduce bandwidth metric calculation module.
Analyzes AllReduce collective operations from NSys traces.
"""

from collections import defaultdict
import statistics


def calculate_allreduce_metrics(all_metrics, world_size=4):
    """
    Calculate AllReduce-specific bandwidth metrics.
    
    Args:
        all_metrics: Dictionary of metrics per rank
        world_size: Number of GPUs in the training run
        
    Returns:
        Dictionary containing AllReduce bandwidth statistics
    """
    allreduce_stats = defaultdict(list)
    
    for rank_name, metrics in all_metrics.items():
        comm_events = metrics.get('communication', [])
        
        for event in comm_events:
            name = event['name']
            if 'allreduce' not in name.lower():
                continue
                
            duration_sec = event['duration'] / 1e6
            message_bytes = event.get('bytes', 0)
            
            if duration_sec == 0 or message_bytes == 0:
                continue
            
            # AllReduce data transferred calculation
            data_transferred = 2 * message_bytes * (world_size - 1) / world_size if world_size > 0 else 0
            
            if data_transferred == 0:
                continue
            
            bandwidth_gbps = (data_transferred / 1e9) / duration_sec
            
            allreduce_stats[rank_name].append({
                'bandwidth_gbps': bandwidth_gbps,
                'message_bytes': message_bytes,
                'data_transferred': data_transferred,
                'duration_ms': duration_sec * 1000
            })
    
    # Aggregate statistics
    result = {}
    for rank_name, measurements in allreduce_stats.items():
        if measurements:
            bandwidths = [m['bandwidth_gbps'] for m in measurements]
            result[rank_name] = {
                'count': len(measurements),
                'avg_bandwidth_gbps': statistics.mean(bandwidths),
                'median_bandwidth_gbps': statistics.median(bandwidths),
                'min_bandwidth_gbps': min(bandwidths),
                'max_bandwidth_gbps': max(bandwidths),
                'total_data_gb': sum(m['data_transferred'] for m in measurements) / 1e9
            }
    
    return result
