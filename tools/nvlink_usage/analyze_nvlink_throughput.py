#!/usr/bin/env python3
"""
Analyze NVLink throughput from binary trace files.

Metrics available:
- max_throughput: Maximum throughput (GB/s) across specified links/directions
- avg_throughput: Average throughput (GB/s) weighted by interval duration
- total_communication: Total bytes transferred (sum of RX+TX divided by 2 for symmetry)

Usage:
  python analyze_nvlink_throughput.py /path/to/trace_dir --metrics max_throughput avg_throughput
  python analyze_nvlink_throughput.py /path/to/trace_dir --metrics all
  python analyze_nvlink_throughput.py /path/to/trace_dir --metrics max_throughput --link 0 --direction tx
"""

import argparse
import struct
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


# NVLink trace format
FILE_MAGIC = b"NVF1"
HEADER_STRUCT = struct.Struct("<4sHHi")  # magic, version, field_size, host_ts_size

# nvmlFieldValue_t offsets
NVML_FIELD_ID_OFFSET = 0
NVML_SCOPE_ID_OFFSET = 4
NVML_TIMESTAMP_OFFSET = 8      # microseconds since Unix epoch (wall-clock)
NVML_VALUE_OFFSET = 32

# NVML field IDs for NVLink throughput
NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX = 138
NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX = 139


def field_id_to_direction(field_id: int) -> str:
    """Convert NVML field ID to TX/RX direction string."""
    if field_id == NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX:
        return "TX"
    elif field_id == NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX:
        return "RX"
    else:
        return f"F{field_id}"


@dataclass
class NVLinkSample:
    """Single NVLink utilization sample."""
    nvml_timestamp_us: int
    field_id: int
    link_id: int
    value: int


@dataclass
class NVLinkInterval:
    """NVLink throughput for a specific time interval."""
    start_us: int
    end_us: int
    field_id: int
    link_id: int
    delta_bytes: int  # in KiB from NVML
    
    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us
    
    @property
    def direction(self) -> str:
        return field_id_to_direction(self.field_id)
    
    @property
    def delta_bytes_actual(self) -> int:
        """Actual bytes transferred (NVML reports in KiB)."""
        return self.delta_bytes * 1024
    
    @property
    def throughput_gbps(self) -> float:
        """Throughput in GB/s."""
        if self.duration_us <= 0:
            return 0.0
        bytes_transferred = self.delta_bytes * 1024
        bytes_per_sec = bytes_transferred * 1e6 / self.duration_us
        return bytes_per_sec / 1e9


def load_nvlink_trace(trace_file: Path) -> List[NVLinkSample]:
    """Load NVLink utilization trace from binary file."""
    samples = []
    
    with open(trace_file, "rb") as f:
        header_data = f.read(HEADER_STRUCT.size)
        magic, version, stored_field_size, host_ts_size = HEADER_STRUCT.unpack(header_data)
        
        if magic != FILE_MAGIC:
            raise ValueError(f"Invalid magic: {magic}")
        
        record_stride = host_ts_size + stored_field_size
        while True:
            record = f.read(record_stride)
            if len(record) < record_stride:
                break
            
            field_data = record[host_ts_size:]
            field_id = struct.unpack("<I", field_data[NVML_FIELD_ID_OFFSET:NVML_FIELD_ID_OFFSET+4])[0]
            scope_id = struct.unpack("<I", field_data[NVML_SCOPE_ID_OFFSET:NVML_SCOPE_ID_OFFSET+4])[0]
            nvml_ts_us = struct.unpack("<Q", field_data[NVML_TIMESTAMP_OFFSET:NVML_TIMESTAMP_OFFSET+8])[0]
            value = struct.unpack("<Q", field_data[NVML_VALUE_OFFSET:NVML_VALUE_OFFSET+8])[0]
            
            samples.append(NVLinkSample(
                nvml_timestamp_us=nvml_ts_us,
                field_id=field_id,
                link_id=scope_id,
                value=value,
            ))
    
    return samples


def compute_nvlink_intervals(samples: List[NVLinkSample]) -> List[NVLinkInterval]:
    """Compute throughput intervals between consecutive samples."""
    by_key: Dict[Tuple[int, int], List[NVLinkSample]] = {}
    for s in samples:
        key = (s.field_id, s.link_id)
        if key not in by_key:
            by_key[key] = []
        by_key[key].append(s)
    
    intervals = []
    for key, key_samples in by_key.items():
        key_samples.sort(key=lambda x: x.nvml_timestamp_us)
        
        for i in range(1, len(key_samples)):
            prev = key_samples[i-1]
            curr = key_samples[i]
            delta = curr.value - prev.value
            if delta < 0:
                delta = 0  # Handle wraparound
            
            intervals.append(NVLinkInterval(
                start_us=prev.nvml_timestamp_us,
                end_us=curr.nvml_timestamp_us,
                field_id=curr.field_id,
                link_id=curr.link_id,
                delta_bytes=delta,
            ))
    
    return intervals


def filter_intervals(
    intervals: List[NVLinkInterval],
    link: Optional[int] = None,
    direction: Optional[str] = None
) -> List[NVLinkInterval]:
    """Filter intervals by link and/or direction."""
    result = intervals
    
    if link is not None:
        result = [iv for iv in result if iv.link_id == link]
    
    if direction is not None:
        dir_upper = direction.upper()
        result = [iv for iv in result if iv.direction == dir_upper]
    
    return result


def load_and_filter_intervals(
    directory: str,
    link: Optional[int] = None,
    direction: Optional[str] = None
) -> List[NVLinkInterval]:
    """Load trace and return filtered intervals."""
    trace_file = Path(directory) / "nvlink_trace.bin"
    if not trace_file.exists():
        raise FileNotFoundError(f"NVLink trace not found: {trace_file}")
    
    samples = load_nvlink_trace(trace_file)
    intervals = compute_nvlink_intervals(samples)
    
    # Filter to valid intervals (positive duration)
    intervals = [iv for iv in intervals if iv.duration_us > 0]
    
    return filter_intervals(intervals, link, direction)


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def metric_max_throughput(
    directory: str,
    link: Optional[int] = None,
    direction: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate maximum throughput across specified links/directions.
    
    Returns dict with:
    - max_throughput_gbps: Maximum throughput in GB/s
    - link_id: Link where max occurred
    - direction: Direction (TX/RX) where max occurred  
    - timestamp_us: Unix epoch microseconds when max occurred
    - interval_duration_us: Duration of the max interval
    - interval_bytes: Bytes transferred in max interval
    """
    intervals = load_and_filter_intervals(directory, link, direction)
    
    if not intervals:
        return {
            "max_throughput_gbps": 0.0,
            "link_id": None,
            "direction": None,
            "timestamp_us": None,
            "interval_duration_us": None,
            "interval_bytes": None,
        }
    
    max_iv = max(intervals, key=lambda x: x.throughput_gbps)
    
    return {
        "max_throughput_gbps": max_iv.throughput_gbps,
        "link_id": max_iv.link_id,
        "direction": max_iv.direction,
        "timestamp_us": max_iv.start_us,
        "interval_duration_us": max_iv.duration_us,
        "interval_bytes": max_iv.delta_bytes_actual,
    }


def metric_avg_throughput(
    directory: str,
    link: Optional[int] = None,
    direction: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate average throughput weighted by interval duration.
    
    Returns dict with:
    - avg_throughput_gbps: Weighted average throughput in GB/s
    - total_duration_us: Total duration of all intervals
    - num_intervals: Number of intervals
    - num_active_intervals: Number of intervals with non-zero throughput
    """
    intervals = load_and_filter_intervals(directory, link, direction)
    
    if not intervals:
        return {
            "avg_throughput_gbps": 0.0,
            "total_duration_us": 0,
            "num_intervals": 0,
            "num_active_intervals": 0,
        }
    
    total_duration = sum(iv.duration_us for iv in intervals)
    
    if total_duration > 0:
        weighted_sum = sum(iv.throughput_gbps * iv.duration_us for iv in intervals)
        avg_throughput = weighted_sum / total_duration
    else:
        avg_throughput = 0.0
    
    num_active = sum(1 for iv in intervals if iv.throughput_gbps > 0)
    
    return {
        "avg_throughput_gbps": avg_throughput,
        "total_duration_us": total_duration,
        "num_intervals": len(intervals),
        "num_active_intervals": num_active,
    }


def metric_total_communication(
    directory: str,
    link: Optional[int] = None,
    direction: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate total bytes communicated.
    
    If direction is "all" (None), sums RX and TX then divides by 2 for symmetry.
    If a specific direction is given, returns just that direction's total.
    
    Returns dict with:
    - total_bytes: Total bytes transferred
    - total_gb: Total GB transferred
    - rx_bytes: RX bytes (if direction is all)
    - tx_bytes: TX bytes (if direction is all)
    """
    intervals = load_and_filter_intervals(directory, link, direction)
    
    if not intervals:
        return {
            "total_bytes": 0,
            "total_gb": 0.0,
            "rx_bytes": 0,
            "tx_bytes": 0,
        }
    
    if direction is None:
        # Sum RX and TX separately, then divide by 2 for symmetry
        rx_bytes = sum(iv.delta_bytes_actual for iv in intervals if iv.direction == "RX")
        tx_bytes = sum(iv.delta_bytes_actual for iv in intervals if iv.direction == "TX")
        total_bytes = (rx_bytes + tx_bytes) // 2
        
        return {
            "total_bytes": total_bytes,
            "total_gb": total_bytes / 1e9,
            "rx_bytes": rx_bytes,
            "tx_bytes": tx_bytes,
        }
    else:
        # Just sum the filtered direction
        total_bytes = sum(iv.delta_bytes_actual for iv in intervals)
        
        return {
            "total_bytes": total_bytes,
            "total_gb": total_bytes / 1e9,
        }


# Metric registry
METRICS = {
    "max_throughput": metric_max_throughput,
    "avg_throughput": metric_avg_throughput,
    "total_communication": metric_total_communication,
}


def metric_cal(
    directory: str,
    metric_name: str = "all",
    link: Optional[int] = None,
    direction: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate specified metric(s) for a trace directory.
    
    This is the main entry point for use with main.py.
    
    Args:
        directory: Path to directory containing nvlink_trace.bin
        metric_name: One of "max_throughput", "avg_throughput", "total_communication", or "all"
        link: Optional link ID filter (None = all links)
        direction: Optional direction filter "tx"/"rx" (None = all directions)
    
    Returns:
        Dictionary with metric results
    """
    if metric_name == "all":
        results = {}
        for name, func in METRICS.items():
            results[name] = func(directory, link, direction)
        return results
    elif metric_name in METRICS:
        return {metric_name: METRICS[metric_name](directory, link, direction)}
    else:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(METRICS.keys())}")


def format_timestamp(timestamp_us: int) -> str:
    """Format Unix epoch microseconds as ISO datetime string."""
    if timestamp_us is None:
        return "N/A"
    return datetime.utcfromtimestamp(timestamp_us / 1e6).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def print_results(results: Dict[str, Any], json_output: bool = False):
    """Print metric results in human-readable or JSON format."""
    if json_output:
        print(json.dumps(results, indent=2))
        return
    
    for metric_name, data in results.items():
        print(f"\n{metric_name}:")
        print("-" * 40)
        
        if metric_name == "max_throughput":
            print(f"  Max Throughput:   {data['max_throughput_gbps']:.4f} GB/s")
            if data['link_id'] is not None:
                print(f"  Link:             {data['link_id']}")
                print(f"  Direction:        {data['direction']}")
                print(f"  Timestamp (us):   {data['timestamp_us']}")
                print(f"  Timestamp:        {format_timestamp(data['timestamp_us'])}")
                print(f"  Interval:         {data['interval_duration_us']} us")
                print(f"  Bytes:            {data['interval_bytes']:,}")
        
        elif metric_name == "avg_throughput":
            print(f"  Avg Throughput:   {data['avg_throughput_gbps']:.4f} GB/s")
            print(f"  Total Duration:   {data['total_duration_us']:,} us")
            print(f"  Intervals:        {data['num_intervals']} ({data['num_active_intervals']} active)")
        
        elif metric_name == "total_communication":
            print(f"  Total Bytes:      {data['total_bytes']:,}")
            print(f"  Total GB:         {data['total_gb']:.4f}")
            if 'rx_bytes' in data:
                print(f"  RX Bytes:         {data['rx_bytes']:,}")
                print(f"  TX Bytes:         {data['tx_bytes']:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze NVLink throughput metrics from binary trace files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics:
  max_throughput      Maximum throughput (GB/s) with link, direction, timestamp
  avg_throughput      Average throughput (GB/s) weighted by interval duration
  total_communication Total bytes transferred (RX+TX)/2 for symmetry

Examples:
  # Calculate all metrics
  %(prog)s /path/to/trace_dir --metrics all

  # Calculate specific metrics
  %(prog)s /path/to/trace_dir --metrics max_throughput avg_throughput

  # Filter to specific link
  %(prog)s /path/to/trace_dir --metrics all --link 0

  # Filter to TX only
  %(prog)s /path/to/trace_dir --metrics all --direction tx

  # Output as JSON
  %(prog)s /path/to/trace_dir --metrics all --json
"""
    )
    parser.add_argument("trace_dir", type=str,
                        help="Directory containing nvlink_trace.bin")
    parser.add_argument("--metrics", "-m", type=str, nargs="+",
                        default=["all"],
                        choices=["all", "max_throughput", "avg_throughput", "total_communication"],
                        help="Metrics to calculate (default: all)")
    parser.add_argument("--link", "-l", type=int, default=None,
                        help="Filter to specific NVLink link ID (default: all links)")
    parser.add_argument("--direction", "-d", type=str, 
                        choices=["tx", "rx", "TX", "RX"],
                        default=None, 
                        help="Filter to TX or RX only (default: all directions)")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output results as JSON")
    
    args = parser.parse_args()
    
    # Combine metrics if multiple specified
    if "all" in args.metrics:
        metrics_to_calc = ["all"]
    else:
        metrics_to_calc = args.metrics
    
    try:
        all_results = {}
        for metric in metrics_to_calc:
            results = metric_cal(
                args.trace_dir,
                metric_name=metric,
                link=args.link,
                direction=args.direction
            )
            all_results.update(results)
        
        print_results(all_results, json_output=args.json)
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)


if __name__ == "__main__":
    main()
