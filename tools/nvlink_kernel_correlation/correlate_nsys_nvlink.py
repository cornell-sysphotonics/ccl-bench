#!/usr/bin/env python3
"""
Correlate NVLink utilization with nsys GPU kernel activity.

Metrics available:
- peak_kernels: Kernels (especially NCCL) during the highest throughput interval(s)

Usage:
  python correlate_nsys_nvlink.py /path/to/trace_dir --metrics peak_kernels
  python correlate_nsys_nvlink.py /path/to/trace_dir --metrics peak_kernels --gpu 0 --top 5
  python correlate_nsys_nvlink.py /path/to/trace_dir --metrics peak_kernels --link 0 --direction tx
"""

import argparse
import json
import sqlite3
import struct
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime


# NVLink trace format
FILE_MAGIC = b"NVF1"
HEADER_STRUCT = struct.Struct("<4sHHi")

# nvmlFieldValue_t offsets
NVML_FIELD_ID_OFFSET = 0
NVML_SCOPE_ID_OFFSET = 4
NVML_TIMESTAMP_OFFSET = 8
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
    delta_bytes: int

    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us

    @property
    def direction(self) -> str:
        return field_id_to_direction(self.field_id)

    @property
    def delta_bytes_actual(self) -> int:
        return self.delta_bytes * 1024

    @property
    def throughput_gbps(self) -> float:
        if self.duration_us <= 0:
            return 0.0
        bytes_transferred = self.delta_bytes * 1024
        bytes_per_sec = bytes_transferred * 1e6 / self.duration_us
        return bytes_per_sec / 1e9


@dataclass
class GpuKernel:
    """GPU kernel from nsys trace."""
    name: str
    start_us: float
    end_us: float
    duration_us: float
    device_id: int
    stream_id: int


@dataclass
class KernelOverlap:
    """Kernel with overlap information for a specific interval."""
    name: str
    start_us: float
    end_us: float
    duration_us: float
    device_id: int
    stream_id: int
    overlap_us: float
    overlap_pct: float
    is_nccl: bool


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
                delta = 0

            intervals.append(NVLinkInterval(
                start_us=prev.nvml_timestamp_us,
                end_us=curr.nvml_timestamp_us,
                field_id=curr.field_id,
                link_id=curr.link_id,
                delta_bytes=delta,
            ))

    return intervals


def get_nsys_session_start(sqlite_file: Path) -> Optional[int]:
    """Extract session start timestamp from nsys SQLite export."""
    conn = sqlite3.connect(sqlite_file)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        if "TARGET_INFO_SESSION_START_TIME" in tables:
            cursor.execute("SELECT * FROM TARGET_INFO_SESSION_START_TIME LIMIT 1")
            row = cursor.fetchone()
            if row:
                session_start_ns = row[0]
                return session_start_ns // 1000
        return None
    finally:
        conn.close()


def load_nsys_kernels(sqlite_file: Path, wall_clock_start_us: int = None, quiet: bool = False) -> List[GpuKernel]:
    """Load GPU kernels from nsys SQLite export."""
    conn = sqlite3.connect(sqlite_file)
    kernels = []

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        if wall_clock_start_us is None:
            if "TARGET_INFO_SESSION_START_TIME" in tables:
                cursor.execute("SELECT * FROM TARGET_INFO_SESSION_START_TIME LIMIT 1")
                row = cursor.fetchone()
                if row:
                    session_start_ns = row[0]
                    wall_clock_start_us = session_start_ns // 1000
                else:
                    raise ValueError("TARGET_INFO_SESSION_START_TIME table is empty")
            else:
                raise ValueError("No wall_clock_start_us provided and TARGET_INFO_SESSION_START_TIME not found")

        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
            cursor.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")
            columns = {row[1]: row[0] for row in cursor.fetchall()}

            name_col = None
            for candidate in ["shortName", "demangledName", "name", "mangledName"]:
                if candidate in columns:
                    name_col = candidate
                    break

            if name_col is None:
                return kernels

            query = f"""
                SELECT {name_col}, start, end, deviceId, streamId
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                ORDER BY start
            """
            cursor.execute(query)

            string_lookup = {}
            if "StringIds" in tables:
                cursor2 = conn.cursor()
                cursor2.execute("SELECT id, value FROM StringIds")
                string_lookup = {row[0]: row[1] for row in cursor2.fetchall()}

            for row in cursor:
                name, start_ns, end_ns, device_id, stream_id = row

                if isinstance(name, int):
                    name = string_lookup.get(name, f"kernel_{name}")

                start_us = wall_clock_start_us + (start_ns / 1000)
                end_us = wall_clock_start_us + (end_ns / 1000)
                duration_us = (end_ns - start_ns) / 1000

                kernels.append(GpuKernel(
                    name=name,
                    start_us=start_us,
                    end_us=end_us,
                    duration_us=duration_us,
                    device_id=device_id,
                    stream_id=stream_id,
                ))

    finally:
        conn.close()

    return kernels


def find_kernels_in_interval(
    kernels: List[GpuKernel],
    interval_start_us: float,
    interval_end_us: float
) -> List[KernelOverlap]:
    """Find kernels that overlap with a specific time interval, with overlap info."""
    results = []
    interval_duration = interval_end_us - interval_start_us

    for k in kernels:
        if k.start_us < interval_end_us and k.end_us > interval_start_us:
            overlap_start = max(k.start_us, interval_start_us)
            overlap_end = min(k.end_us, interval_end_us)
            overlap_us = overlap_end - overlap_start
            overlap_pct = 100 * overlap_us / interval_duration if interval_duration > 0 else 0

            results.append(KernelOverlap(
                name=k.name,
                start_us=k.start_us,
                end_us=k.end_us,
                duration_us=k.duration_us,
                device_id=k.device_id,
                stream_id=k.stream_id,
                overlap_us=overlap_us,
                overlap_pct=overlap_pct,
                is_nccl="nccl" in k.name.lower(),
            ))

    # Sort by overlap (descending)
    results.sort(key=lambda x: -x.overlap_us)
    return results


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


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def metric_peak_kernels(
    directory: str,
    gpu: int = 0,
    link: Optional[int] = None,
    direction: Optional[str] = None,
    top: int = 1,
    nccl_only: bool = False,
) -> Dict[str, Any]:
    """
    Find kernels during the highest throughput interval(s).

    Args:
        directory: Path to trace directory
        gpu: GPU device ID to filter kernels (default: 0)
        link: NVLink link ID filter (None = all links)
        direction: Direction filter "tx"/"rx" (None = all directions)
        top: Number of peak intervals to analyze (default: 1)
        nccl_only: If True, only return NCCL kernels

    Returns dict with:
    - peak_interval: Info about the peak throughput interval
    - kernels: List of kernels sorted by overlap with the interval
    """
    trace_dir = Path(directory)

    # Load NVLink trace
    bin_files = list(trace_dir.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No .bin trace file found in: {trace_dir}")
    nvlink_file = bin_files[0]  # Use first .bin file found

    samples = load_nvlink_trace(nvlink_file)
    intervals = compute_nvlink_intervals(samples)

    # Filter intervals
    intervals = [iv for iv in intervals if iv.duration_us > 0]
    intervals = filter_intervals(intervals, link, direction)

    if not intervals:
        return {
            "peak_interval": None,
            "kernels": [],
            "error": "No NVLink intervals found after filtering",
        }

    # Find peak interval(s) by throughput
    sorted_intervals = sorted(intervals, key=lambda x: -x.throughput_gbps)
    peak_intervals = sorted_intervals[:top]

    # Load nsys kernels
    sqlite_file = trace_dir / "vllm_profile.sqlite"
    if not sqlite_file.exists():
        sqlite_files = list(trace_dir.glob("*.sqlite"))
        if sqlite_files:
            sqlite_file = sqlite_files[0]
        else:
            return {
                "peak_interval": None,
                "kernels": [],
                "error": f"No SQLite file found in {trace_dir}",
            }

    kernels = load_nsys_kernels(sqlite_file, quiet=True)

    # Filter kernels by GPU
    if gpu is not None:
        kernels = [k for k in kernels if k.device_id == gpu]

    if not kernels:
        return {
            "peak_interval": None,
            "kernels": [],
            "error": f"No kernels found for GPU {gpu}",
        }

    # Build results for each peak interval
    results = []
    for iv in peak_intervals:
        overlapping = find_kernels_in_interval(kernels, iv.start_us, iv.end_us)

        if nccl_only:
            overlapping = [k for k in overlapping if k.is_nccl]

        interval_info = {
            "start_us": iv.start_us,
            "end_us": iv.end_us,
            "duration_us": iv.duration_us,
            "throughput_gbps": iv.throughput_gbps,
            "bytes_transferred": iv.delta_bytes_actual,
            "link_id": iv.link_id,
            "direction": iv.direction,
            "timestamp_iso": datetime.utcfromtimestamp(iv.start_us / 1e6).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
        }

        kernel_list = []
        for k in overlapping:
            kernel_list.append({
                "name": k.name,
                "start_us": k.start_us,
                "end_us": k.end_us,
                "duration_us": k.duration_us,
                "device_id": k.device_id,
                "stream_id": k.stream_id,
                "overlap_us": k.overlap_us,
                "overlap_pct": k.overlap_pct,
                "is_nccl": k.is_nccl,
            })

        results.append({
            "interval": interval_info,
            "kernels": kernel_list,
            "num_kernels": len(kernel_list),
            "num_nccl_kernels": sum(1 for k in kernel_list if k["is_nccl"]),
        })

    # Return single result if top=1, otherwise list
    if top == 1:
        return {
            "peak_interval": results[0]["interval"] if results else None,
            "kernels": results[0]["kernels"] if results else [],
            "num_kernels": results[0]["num_kernels"] if results else 0,
            "num_nccl_kernels": results[0]["num_nccl_kernels"] if results else 0,
        }
    else:
        return {
            "peaks": results,
            "num_peaks": len(results),
        }


# Metric registry
METRICS = {
    "peak_kernels": metric_peak_kernels,
}


def metric_cal(
    directory: str,
    metric_name: str = "peak_kernels",
    gpu: int = 0,
    link: Optional[int] = None,
    direction: Optional[str] = None,
    top: int = 1,
    nccl_only: bool = False,
) -> Dict[str, Any]:
    """
    Calculate specified metric for a trace directory.

    This is the main entry point for use with main.py.

    Args:
        directory: Path to directory containing *.bin and *.sqlite files
        metric_name: Currently only "peak_kernels" supported
        gpu: GPU device ID for kernel filtering (default: 0)
        link: Optional NVLink link ID filter (None = all links)
        direction: Optional direction filter "tx"/"rx" (None = all directions)
        top: Number of peak intervals to analyze (default: 1)
        nccl_only: If True, only return NCCL kernels

    Returns:
        Dictionary with metric results
    """
    if metric_name == "peak_kernels":
        return {metric_name: METRICS[metric_name](
            directory, gpu=gpu, link=link, direction=direction,
            top=top, nccl_only=nccl_only
        )}
    else:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(METRICS.keys())}")


def format_timestamp(timestamp_us: float) -> str:
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
        print("=" * 60)

        if "peaks" in data:
            # Multiple peaks
            for i, peak in enumerate(data["peaks"]):
                print_peak(peak, i + 1)
        elif "peak_interval" in data:
            # Single peak
            print_peak(data, 1)
        elif "error" in data:
            print(f"  ERROR: {data['error']}")


def print_peak(data: Dict[str, Any], peak_num: int):
    """Print a single peak interval and its kernels."""
    interval = data.get("interval") or data.get("peak_interval")
    kernels = data.get("kernels", [])

    if interval is None:
        print(f"  Peak {peak_num}: No data")
        return

    print(f"\n  Peak {peak_num} Interval:")
    print(f"    Throughput:    {interval['throughput_gbps']:.4f} GB/s")
    print(f"    Link:          {interval['link_id']} {interval['direction']}")
    print(f"    Timestamp:     {interval['timestamp_iso']}")
    print(f"    Timestamp (us): {interval['start_us']}")
    print(f"    Duration:      {interval['duration_us']:.0f} us ({interval['duration_us']/1000:.3f} ms)")
    print(f"    Bytes:         {interval['bytes_transferred']:,}")

    num_kernels = data.get("num_kernels", len(kernels))
    num_nccl = data.get("num_nccl_kernels", sum(1 for k in kernels if k.get("is_nccl")))

    print(f"\n  Kernels during interval: {num_kernels} ({num_nccl} NCCL)")
    print("-" * 60)

    if not kernels:
        print("    (no kernels found)")
        return

    # Print kernels sorted by overlap
    for k in kernels[:20]:  # Limit output
        nccl_marker = "[NCCL] " if k["is_nccl"] else ""
        print(f"    {nccl_marker}{k['name'][:50]}")
        print(f"      overlap: {k['overlap_pct']:.1f}% ({k['overlap_us']:.0f} us), "
              f"duration: {k['duration_us']/1000:.3f} ms")

    if len(kernels) > 20:
        print(f"    ... and {len(kernels) - 20} more kernels")


def main():
    parser = argparse.ArgumentParser(
        description="Correlate NVLink peaks with GPU kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics:
  peak_kernels    Kernels during the highest throughput interval(s)

Examples:
  # Find kernels during peak throughput (GPU 0, all links)
  %(prog)s /path/to/trace_dir --metrics peak_kernels

  # Filter to specific GPU
  %(prog)s /path/to/trace_dir --metrics peak_kernels --gpu 1

  # Filter to specific link and direction
  %(prog)s /path/to/trace_dir --metrics peak_kernels --link 0 --direction tx

  # Show top 5 peak intervals
  %(prog)s /path/to/trace_dir --metrics peak_kernels --top 5

  # Only show NCCL kernels
  %(prog)s /path/to/trace_dir --metrics peak_kernels --nccl-only

  # Output as JSON
  %(prog)s /path/to/trace_dir --metrics peak_kernels --json
"""
    )
    parser.add_argument("trace_dir", type=str,
                        help="Directory containing *.bin and *.sqlite files")
    parser.add_argument("--metrics", "-m", type=str, nargs="+",
                        default=["peak_kernels"],
                        choices=["peak_kernels"],
                        help="Metrics to calculate (default: peak_kernels)")
    parser.add_argument("--gpu", "-g", type=int, default=0,
                        help="GPU device ID for kernel filtering (default: 0)")
    parser.add_argument("--link", "-l", type=int, default=None,
                        help="Filter to specific NVLink link ID (default: all)")
    parser.add_argument("--direction", "-d", type=str,
                        choices=["tx", "rx", "TX", "RX"],
                        default=None,
                        help="Filter to TX or RX only (default: all)")
    parser.add_argument("--top", "-n", type=int, default=1,
                        help="Number of peak intervals to analyze (default: 1)")
    parser.add_argument("--nccl-only", action="store_true",
                        help="Only show NCCL communication kernels")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output results as JSON")

    args = parser.parse_args()

    try:
        all_results = {}
        for metric in args.metrics:
            results = metric_cal(
                args.trace_dir,
                metric_name=metric,
                gpu=args.gpu,
                link=args.link,
                direction=args.direction,
                top=args.top,
                nccl_only=args.nccl_only,
            )
            all_results.update(results)

        print_results(all_results, json_output=args.json)

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
