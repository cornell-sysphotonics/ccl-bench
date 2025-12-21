import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import statistics
import math
import importlib.util

# Add the directory containing this script to the Python path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

# Import the three metric calculation modules from subdirectories
def load_module_from_path(module_name, file_path):
    """Dynamically load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load the metric calculation modules
allgather_module = load_module_from_path(
    "avg_allgather_bandwidth_group_2",
    script_dir / "avg_allgather_bandwidth_group_2" / "avg_allgather_bandwidth_group_2.py"
)
allreduce_module = load_module_from_path(
    "avg_allreduce_bandwidth_group_2", 
    script_dir / "avg_allreduce_bandwidth_group_2" / "avg_allreduce_bandwidth_group_2.py"
)
reducescatter_module = load_module_from_path(
    "avg_reducescatter_bandwidth_group_2",
    script_dir / "avg_reducescatter_bandwidth_group_2" / "avg_reducescatter_bandwidth_group_2.py"
)

# Extract the functions
calculate_allgather_metrics = allgather_module.calculate_allgather_metrics
calculate_allreduce_metrics = allreduce_module.calculate_allreduce_metrics
calculate_reducescatter_metrics = reducescatter_module.calculate_reducescatter_metrics


def infer_message_bytes(args):
    """Best-effort inference of tensor size in bytes for NCCL ops."""
    nelems = args.get('In msg nelems', 0) or args.get('Out msg nelems', 0)
    if not nelems:
        return 0

    dtype = (
        args.get('Data type')
        or args.get('data_type')
        or args.get('Type')
        or args.get('dtype')
        or args.get('dataType')
        or ''
    ).lower()
    dtype_sizes = {
        'ncclfloat32': 4,
        'float32': 4,
        'single': 4,
        'ncclfloat16': 2,
        'float16': 2,
        'half': 2,
        'ncclbfloat16': 2,
        'bfloat16': 2,
        'ncclfloat8': 1,
        'float8': 1,
        'ncclint8': 1,
        'int8': 1,
        'nccluint8': 1,
        'uint8': 1,
        'ncclfloat64': 8,
        'float64': 8,
        'double': 8,
        'ncclint32': 4,
        'int32': 4,
        'ncclint64': 8,
        'int64': 8,
    }
    element_size = dtype_sizes.get(dtype, 4)
    return nelems * element_size

def analyze_profile(data, rank_name):
    """Extract events from NSys JSON trace."""
    metrics = {
        'rank_name': rank_name,
        'communication': [],
        'compute': [],
        'all_events': []
    }
    
    events = data.get('traceEvents', [])
    if not events:
        return metrics
    
    # Filter valid events
    valid_events = [e for e in events if e.get('ts', 0) > 0 and e.get('dur', 0) > 0]
    if not valid_events:
        return metrics
    
    min_ts = min(e['ts'] for e in valid_events)
    max_ts = max(e['ts'] + e.get('dur', 0) for e in valid_events)
    metrics['total_time'] = max_ts - min_ts
    metrics['min_ts'] = min_ts
    metrics['max_ts'] = max_ts
    
    # Keywords for identifying NCCL operations
    NCCL_KEYWORDS = ['nccl', 'NCCL', 'AllReduce', 'AllGather', 'ReduceScatter', 'Broadcast']
    
    for event in valid_events:
        ts = event.get('ts', 0)
        dur = event.get('dur', 0)
        name = event.get('name', 'unknown')
        cat = event.get('cat', '')
        
        event_info = {
            'name': name,
            'start': ts,
            'end': ts + dur,
            'duration': dur,
            'category': cat,
            'bytes': 0
        }
        
        # Identify NCCL communication
        is_nccl = any(keyword in name for keyword in NCCL_KEYWORDS)
        if is_nccl:
            args = event.get('args', {})
            inferred_bytes = infer_message_bytes(args)
            if inferred_bytes:
                event_info['bytes'] = inferred_bytes
            metrics['communication'].append(event_info)
        
        # Identify compute kernels - anything in 'kernel' category that's NOT NCCL
        if cat == 'kernel' and not is_nccl:
            metrics['compute'].append(event_info)
        
        metrics['all_events'].append(event_info)
    
    return metrics

def calculate_throughput(all_metrics, batch_size, seq_length, num_iterations):
    """Calculate throughput metrics."""
    min_ts = min(m['min_ts'] for m in all_metrics.values())
    max_ts = max(m['max_ts'] for m in all_metrics.values())
    total_time_sec = (max_ts - min_ts) / 1e6 if max_ts > min_ts else 0
    
    total_tokens = batch_size * seq_length * num_iterations
    tokens_per_sec = total_tokens / total_time_sec if total_time_sec > 0 else 0
    total_samples = batch_size * num_iterations
    
    return {
        'tokens_per_second': tokens_per_sec,
        'samples_per_second': total_samples / total_time_sec if total_time_sec > 0 else 0,
        'time_per_iteration_ms': (total_time_sec / num_iterations * 1000) if num_iterations > 0 else 0,
        'total_time_sec': total_time_sec
    }


def merge_intervals(events):
    intervals = sorted((e['start'], e['end']) for e in events if e['end'] > e['start'])
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def sum_intervals(intervals):
    return sum(end - start for start, end in intervals)


def overlap_between_intervals(a_intervals, b_intervals):
    i = j = 0
    overlap = 0
    while i < len(a_intervals) and j < len(b_intervals):
        start = max(a_intervals[i][0], b_intervals[j][0])
        end = min(a_intervals[i][1], b_intervals[j][1])
        if start < end:
            overlap += (end - start)
        if a_intervals[i][1] < b_intervals[j][1]:
            i += 1
        else:
            j += 1
    return overlap

def calculate_overlap(all_metrics):
    """Calculate communication-computation overlap."""
    overlaps = {}
    
    for rank_name, metrics in all_metrics.items():
        comm_events = metrics['communication']
        compute_events = metrics['compute']
        
        if not comm_events:
            overlaps[rank_name] = 0
            continue
        
        comm_intervals = merge_intervals(comm_events)
        total_comm_time = sum_intervals(comm_intervals)
        if total_comm_time == 0:
            overlaps[rank_name] = 0
            continue
        
        compute_intervals = merge_intervals(compute_events)
        overlap_time = overlap_between_intervals(comm_intervals, compute_intervals)
        
        overlap_percent = min((overlap_time / total_comm_time * 100), 100.0)
        overlaps[rank_name] = overlap_percent
    
    overlap_values = list(overlaps.values())
    return {
        'avg_overlap_percent': statistics.mean(overlap_values) if overlap_values else 0,
        'min_overlap_percent': min(overlap_values) if overlap_values else 0,
        'max_overlap_percent': max(overlap_values) if overlap_values else 0,
        'per_rank_overlap': overlaps
    }

def classify_collective(name, message_bytes, world_size):
    lname = name.lower()
    if 'allreduce' in lname:
        coll_type = 'AllReduce'
        data_transferred = 2 * message_bytes * (world_size - 1) / world_size if world_size > 0 else 0
    elif 'reducescatter' in lname:
        coll_type = 'ReduceScatter'
        data_transferred = message_bytes * (world_size - 1) / world_size if world_size > 0 else 0
    elif 'allgather' in lname:
        coll_type = 'AllGather'
        data_transferred = message_bytes * (world_size - 1) / world_size if world_size > 0 else 0
    elif 'broadcast' in lname:
        coll_type = 'Broadcast'
        data_transferred = message_bytes * math.log2(world_size) if world_size > 1 else message_bytes
    else:
        coll_type = 'Other'
        data_transferred = message_bytes
    return coll_type, data_transferred


def calculate_collective_bandwidth(comm_events, world_size=4):
    """Calculate bandwidth using alpha-beta model for collective operations."""
    collective_stats = defaultdict(list)
    
    for event in comm_events:
        name = event['name']
        duration_sec = event['duration'] / 1e6
        message_bytes = event.get('bytes', 0)
        
        if duration_sec == 0 or message_bytes == 0:
            continue
        
        coll_type, data_transferred = classify_collective(name, message_bytes, world_size)
        if data_transferred == 0:
            continue
        
        bandwidth_gbps = (data_transferred / 1e9) / duration_sec
        collective_stats[coll_type].append({
            'bandwidth_gbps': bandwidth_gbps,
            'message_bytes': message_bytes,
            'data_transferred': data_transferred,
            'duration_ms': duration_sec * 1000
        })
    
    stats = {}
    for coll_type, measurements in collective_stats.items():
        if measurements:
            bandwidths = [m['bandwidth_gbps'] for m in measurements]
            stats[coll_type] = {
                'count': len(measurements),
                'avg_bandwidth_gbps': statistics.mean(bandwidths),
                'median_bandwidth_gbps': statistics.median(bandwidths),
                'min_bandwidth_gbps': min(bandwidths),
                'max_bandwidth_gbps': max(bandwidths),
                'total_data_gb': sum(m['data_transferred'] for m in measurements) / 1e9
            }
    
    return stats

def calculate_network_load(all_metrics, world_size=4, nvlink_peak_bw=600):
    """Calculate network load and bandwidth utilization."""
    per_rank_stats = {}

    for rank_name, metrics in all_metrics.items():
        comm_events = metrics['communication']

        comm_intervals = merge_intervals(comm_events)
        total_comm_time = sum_intervals(comm_intervals)
        total_comm_time_sec = total_comm_time / 1e6

        # Collective bandwidth using alpha-beta model
        coll_stats = calculate_collective_bandwidth(comm_events, world_size)
        total_data_gb = sum(s['total_data_gb'] for s in coll_stats.values())
        total_data_bytes = total_data_gb * 1e9

        if total_comm_time_sec > 0 and total_data_bytes > 0:
            simple_bw = (total_data_bytes / 1e9) / total_comm_time_sec
        else:
            simple_bw = 0

        # Weighted average bandwidth
        if coll_stats:
            total_data = sum(s['total_data_gb'] for s in coll_stats.values())
            weighted_bw = sum(
                s['avg_bandwidth_gbps'] * s['total_data_gb']
                for s in coll_stats.values()
            ) / total_data if total_data > 0 else 0
        else:
            weighted_bw = 0

        per_rank_stats[rank_name] = {
            'total_bytes': total_data_bytes,
            'total_comm_time_ms': total_comm_time / 1e3,
            'simple_bandwidth_gbps': simple_bw,
            'collective_bandwidth_gbps': weighted_bw,
            'num_comm_ops': len(comm_events),
            'collective_breakdown': coll_stats
        }

    # Aggregate across ranks
    collective_values = [s['collective_bandwidth_gbps'] for s in per_rank_stats.values()]
    avg_collective_bw = statistics.mean(collective_values) if collective_values else 0
    total_traffic = sum(s['total_bytes'] for s in per_rank_stats.values())
    utilization = (avg_collective_bw / nvlink_peak_bw * 100) if nvlink_peak_bw > 0 else 0

    return {
        'avg_bandwidth_gbps': avg_collective_bw,
        'peak_bandwidth_gbps': nvlink_peak_bw,
        'bandwidth_utilization_percent': utilization,
        'total_traffic_gb': total_traffic / 1e9,
        'per_rank_stats': per_rank_stats
    }

def calculate_traffic_distribution(all_metrics, world_size=4):
    """Calculate traffic distribution by collective type."""
    collective_types = defaultdict(lambda: {'count': 0, 'total_bytes': 0, 'total_time': 0})

    for metrics in all_metrics.values():
        for event in metrics['communication']:
            name = event['name']
            message_bytes = event.get('bytes', 0)
            coll_type, data_transferred = classify_collective(name, message_bytes, world_size)

            collective_types[coll_type]['count'] += 1
            collective_types[coll_type]['total_bytes'] += data_transferred
            collective_types[coll_type]['total_time'] += event['duration']
    
    # Calculate percentages
    total_bytes = sum(ct['total_bytes'] for ct in collective_types.values())
    
    distribution = {}
    for coll_type, stats in collective_types.items():
        distribution[coll_type] = {
            'count': stats['count'],
            'total_bytes': stats['total_bytes'],
            'percent_of_traffic': (stats['total_bytes'] / total_bytes * 100) if total_bytes > 0 else 0,
            'avg_time_ms': (stats['total_time'] / stats['count'] / 1e3) if stats['count'] > 0 else 0
        }
    
    return distribution

def write_report(all_metrics, output_file, batch_size, seq_length, num_iterations, parallel_cfg):
    """Write comprehensive performance report."""
    throughput = calculate_throughput(all_metrics, batch_size, seq_length, num_iterations)
    overlap = calculate_overlap(all_metrics)
    world_size = len(all_metrics)
    network = calculate_network_load(all_metrics, world_size=world_size)
    traffic_dist = calculate_traffic_distribution(all_metrics, world_size=world_size)
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NSYS DISTRIBUTED TRAINING PERFORMANCE METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Configuration: {world_size} GPUs, {num_iterations} iterations\n")
        f.write(f"Batch Size: {batch_size}, Sequence Length: {seq_length}\n")
        if parallel_cfg:
            f.write("Parallelism:\n")
            for key, value in sorted(parallel_cfg.items()):
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # 1. Throughput
        f.write("=" * 80 + "\n")
        f.write("1. THROUGHPUT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Tokens per Second: {throughput['tokens_per_second']:.2f}\n")
        f.write(f"Samples per Second: {throughput['samples_per_second']:.2f}\n")
        f.write(f"Time per Iteration: {throughput['time_per_iteration_ms']:.2f} ms\n")
        f.write(f"Total Execution Time: {throughput['total_time_sec']:.2f} s\n\n")
        
        # 2. Overlap
        f.write("=" * 80 + "\n")
        f.write("2. COMMUNICATION-COMPUTATION OVERLAP\n")
        f.write("=" * 80 + "\n")
        f.write(f"Average Overlap: {overlap['avg_overlap_percent']:.2f}%\n")
        f.write(f"Min Overlap: {overlap['min_overlap_percent']:.2f}%\n")
        f.write(f"Max Overlap: {overlap['max_overlap_percent']:.2f}%\n\n")
        f.write("Per-Rank Overlap:\n")
        for rank, pct in overlap['per_rank_overlap'].items():
            f.write(f"  {rank}: {pct:.2f}%\n")
        f.write("\n")
        
        # 3. Network Load
        f.write("=" * 80 + "\n")
        f.write("3. NETWORK LOAD (Alpha-Beta Model)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Average Collective Bandwidth: {network['avg_bandwidth_gbps']:.2f} GB/s\n")
        f.write(f"Peak Bandwidth (NVLink): {network['peak_bandwidth_gbps']:.2f} GB/s\n")
        f.write(f"Bandwidth Utilization: {network['bandwidth_utilization_percent']:.2f}%\n")
        f.write(f"Total Traffic: {network['total_traffic_gb']:.2f} GB\n\n")
        
        f.write("Per-Rank Network Stats:\n")
        for rank, stats in network['per_rank_stats'].items():
            f.write(f"\n  {rank}:\n")
            f.write(f"    Collective Bandwidth: {stats['collective_bandwidth_gbps']:.2f} GB/s\n")
            f.write(f"    Simple Bandwidth: {stats['simple_bandwidth_gbps']:.2f} GB/s\n")
            f.write(f"    Total Bytes: {stats['total_bytes'] / 1e9:.2f} GB\n")
            f.write(f"    Communication Operations: {stats['num_comm_ops']}\n")
            
            if stats['collective_breakdown']:
                f.write(f"\n    Collective Breakdown:\n")
                for coll_type, coll_stats in sorted(stats['collective_breakdown'].items()):
                    f.write(f"      {coll_type}:\n")
                    f.write(f"        Count: {coll_stats['count']}\n")
                    f.write(f"        Avg BW: {coll_stats['avg_bandwidth_gbps']:.2f} GB/s\n")
                    f.write(f"        Median BW: {coll_stats['median_bandwidth_gbps']:.2f} GB/s\n")
                    f.write(f"        Min/Max BW: {coll_stats['min_bandwidth_gbps']:.2f} / {coll_stats['max_bandwidth_gbps']:.2f} GB/s\n")
                    f.write(f"        Data Transferred: {coll_stats['total_data_gb']:.2f} GB\n")
        f.write("\n")
        
        # 4. Traffic Distribution
        f.write("=" * 80 + "\n")
        f.write("4. TRAFFIC DISTRIBUTION\n")
        f.write("=" * 80 + "\n")
        for coll_type, stats in sorted(traffic_dist.items(), key=lambda x: x[1]['count'], reverse=True):
            f.write(f"\n{coll_type}:\n")
            f.write(f"  Count: {stats['count']}\n")
            f.write(f"  Total Traffic: {stats['total_bytes'] / 1e9:.2f} GB\n")
            f.write(f"  Percent of Total Traffic: {stats['percent_of_traffic']:.2f}%\n")
            f.write(f"  Average Time: {stats['avg_time_ms']:.2f} ms\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

def load_training_config(config_path: Path):
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    training_cfg = config.get("training", {})
    parallel_cfg = config.get("parallelism", {})
    model_cfg = config.get("model", {})
    comm_cfg = config.get("comm", {})

    local_batch = training_cfg.get("local_batch_size", 1)
    data_repl = parallel_cfg.get("data_parallel_replicate_degree", 1)
    data_shard = parallel_cfg.get("data_parallel_shard_degree", 1)

    batch_size = int(local_batch) * int(data_repl) * int(data_shard)
    seq_length = int(training_cfg.get("seq_len", 0))
    num_iterations = int(training_cfg.get("steps", training_cfg.get("num_iterations", 0)))

    model_info = {
        "name": model_cfg.get("name", "model"),
        "flavor": model_cfg.get("flavor", "")
    }

    comm_backend = comm_cfg.get("backend", "nsys")

    return batch_size, seq_length, num_iterations, parallel_cfg, model_info, comm_backend


def build_output_filename(model_info, parallel_cfg, world_size, comm_backend):
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    model_name = model_info.get("name", "model")
    flavor = model_info.get("flavor", "")
    model_part = f"{model_name}{flavor}".replace(" ", "-")

    dp_rep = int(parallel_cfg.get("data_parallel_replicate_degree", 1))
    dp_shard = int(parallel_cfg.get("data_parallel_shard_degree", 1))
    tp = int(parallel_cfg.get("tensor_parallel_degree", 1))

    prefix = comm_backend.lower() if comm_backend else "nsys"
    return f"{prefix}_{timestamp}_{model_part}_dp{dp_rep}x{dp_shard}_tp{tp}_gpu{world_size}.txt"


def main():
    default_config = Path("torchtitan/torchtitan/models/llama3/train_configs/llama3_8b.toml")
    parser = argparse.ArgumentParser(
        description="Analyze all NSys JSON traces within one or more directories."
    )
    parser.add_argument(
        "trace_dirs",
        nargs="+",
        help="Folder(s) containing per-rank NSys JSON traces"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help=f"Path to training config TOML (default: {default_config})"
    )

    args = parser.parse_args()

    trace_files = []
    for trace_dir in args.trace_dirs:
        dir_path = Path(trace_dir)
        if not dir_path.exists():
            print(f"Warning: Directory not found: {trace_dir}")
            continue
        if not dir_path.is_dir():
            print(f"Warning: Not a directory, skipping: {trace_dir}")
            continue
        json_files = sorted(dir_path.glob("*.json"))
        if not json_files:
            print(f"Warning: No JSON traces found in {trace_dir}")
        trace_files.extend(json_files)

    if not trace_files:
        print("Error: No trace files found in the provided directories")
        sys.exit(1)

    batch_size, seq_length, num_iterations, parallel_cfg, model_info, comm_backend = load_training_config(Path(args.config))
    print(f"Loaded training config from {args.config}")
    print(f"Analyzing {len(trace_files)} trace files from {len(args.trace_dirs)} directories...")
    print(
        f"Config: batch_size={batch_size}, seq_length={seq_length}, iterations={num_iterations}\n"
    )

    all_metrics = {}
    for path in trace_files:
        print(f"Processing {path}...")
        with open(path, 'r') as f:
            data = json.load(f)

        rank_name = path.stem
        if rank_name in all_metrics:
            rank_name = f"{path.parent.name}_{path.stem}"
        metrics = analyze_profile(data, rank_name)
        all_metrics[rank_name] = metrics

    if not all_metrics:
        print("Error: No valid trace files processed")
        sys.exit(1)

    world_size = len(all_metrics)
    analysis_dir = Path("trace_analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_file = analysis_dir / build_output_filename(model_info, parallel_cfg, world_size, comm_backend)
    print(f"\nWriting report to {output_file}...")
    write_report(all_metrics, output_file, batch_size, seq_length, num_iterations, parallel_cfg)
    print(f"\n✓ Analysis complete! Report saved to: {output_file}")


if __name__ == "__main__":
    main()