#!/usr/bin/env python3
"""
Correlate NVLink utilization with nsys GPU kernel activity.

This script:
1. Loads timing markers to get wall-clock reference
2. Loads NVLink utilization data (binary format with wall-clock timestamps from NVML)
3. Loads nsys SQLite database to get kernel execution times
4. Correlates NVLink peaks with kernel activity

Key insight: NVML timestamps are wall-clock (Unix epoch microseconds),
and nsys kernel timestamps can be converted to wall-clock using session metadata.
"""

import argparse
import json
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime


# NVLink trace format
FILE_MAGIC = b"NVF1"
HEADER_STRUCT = struct.Struct("<4sHHi")  # magic, version, field_size, host_ts_size
HOST_TS_STRUCT = struct.Struct("<Q")      # host timestamp (nanoseconds)

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
    nvml_timestamp_us: int     # Wall-clock microseconds (from NVML)
    field_id: int
    link_id: int
    value: int


@dataclass
class TimingMarker:
    """Clock synchronization marker."""
    name: str
    wall_clock_ns: int
    wall_clock_us: int
    monotonic_ns: int


@dataclass
class GpuKernel:
    """GPU kernel from nsys trace."""
    name: str
    start_us: float       # Wall-clock start time (microseconds)
    end_us: float         # Wall-clock end time (microseconds)
    duration_us: float
    device_id: int
    stream_id: int


def load_timing_marker(profile_dir: Path, marker_name: str) -> Optional[TimingMarker]:
    """Load a specific timing marker."""
    marker_file = profile_dir / f"timing_{marker_name}.json"
    if not marker_file.exists():
        return None
    
    with open(marker_file) as f:
        data = json.load(f)
    
    return TimingMarker(
        name=data["name"],
        wall_clock_ns=data["wall_clock_ns"],
        wall_clock_us=data.get("wall_clock_us", data["wall_clock_ns"] // 1000),
        monotonic_ns=data.get("monotonic_ns", 0),
    )


def load_nvlink_trace(trace_file: Path) -> List[NVLinkSample]:
    """Load NVLink utilization trace from binary file."""
    samples = []
    
    with open(trace_file, "rb") as f:
        # Read header
        header_data = f.read(HEADER_STRUCT.size)
        magic, version, stored_field_size, host_ts_size = HEADER_STRUCT.unpack(header_data)
        
        if magic != FILE_MAGIC:
            raise ValueError(f"Invalid magic: {magic}")
        
        # Read records
        record_stride = host_ts_size + stored_field_size
        while True:
            record = f.read(record_stride)
            if len(record) < record_stride:
                break
            
            # Parse nvmlFieldValue_t
            field_data = record[host_ts_size:]
            field_id = struct.unpack("<I", field_data[NVML_FIELD_ID_OFFSET:NVML_FIELD_ID_OFFSET+4])[0]
            scope_id = struct.unpack("<I", field_data[NVML_SCOPE_ID_OFFSET:NVML_SCOPE_ID_OFFSET+4])[0]
            
            # NVML timestamp - microseconds since Unix epoch (wall-clock)
            nvml_ts_us = struct.unpack("<Q", field_data[NVML_TIMESTAMP_OFFSET:NVML_TIMESTAMP_OFFSET+8])[0]
            
            # Value (at offset 32)
            value = struct.unpack("<Q", field_data[NVML_VALUE_OFFSET:NVML_VALUE_OFFSET+8])[0]
            
            samples.append(NVLinkSample(
                nvml_timestamp_us=nvml_ts_us,
                field_id=field_id,
                link_id=scope_id,
                value=value,
            ))
    
    return samples


@dataclass
class NVLinkInterval:
    """NVLink throughput for a specific time interval."""
    start_us: int      # Start of measurement interval
    end_us: int        # End of measurement interval
    field_id: int
    link_id: int
    delta_bytes: int   # Bytes transferred during this interval (in KiB from NVML)
    
    @property
    def duration_us(self) -> int:
        """Duration of this interval in microseconds."""
        return self.end_us - self.start_us
    
    @property
    def direction(self) -> str:
        """TX or RX direction."""
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
        # NVML counter is in KiB, convert to bytes
        bytes_transferred = self.delta_bytes * 1024
        # bytes / microseconds * 1e6 = bytes/second
        bytes_per_sec = bytes_transferred * 1e6 / self.duration_us
        return bytes_per_sec / 1e9  # Convert to GB/s


def compute_nvlink_intervals(samples: List[NVLinkSample]) -> List[NVLinkInterval]:
    """Compute throughput intervals between consecutive samples.
    
    Each interval represents the time between two NVLink samples,
    with the delta representing bytes transferred during that period.
    """
    # Group by (field_id, link_id)
    by_key = {}
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


def aggregate_intervals_by_time(intervals: List[NVLinkInterval], bucket_us: int = 1000) -> List[Tuple[int, int]]:
    """Aggregate NVLink intervals into time buckets.
    
    Returns list of (bucket_start_us, total_bytes).
    """
    buckets = {}
    for iv in intervals:
        # Use midpoint of interval for bucketing
        midpoint = (iv.start_us + iv.end_us) // 2
        bucket_key = (midpoint // bucket_us) * bucket_us
        buckets[bucket_key] = buckets.get(bucket_key, 0) + iv.delta_bytes
    
    return sorted(buckets.items())


def get_nsys_session_start(sqlite_file: Path) -> Optional[int]:
    """Extract session start timestamp from nsys SQLite export.
    
    Returns:
        Wall-clock time in microseconds since Unix epoch, or None if not found
    """
    conn = sqlite3.connect(sqlite_file)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        if "TARGET_INFO_SESSION_START_TIME" in tables:
            cursor.execute("SELECT * FROM TARGET_INFO_SESSION_START_TIME LIMIT 1")
            row = cursor.fetchone()
            if row:
                # First column is nanoseconds since Unix epoch
                session_start_ns = row[0]
                session_start_us = session_start_ns // 1000
                return session_start_us
        return None
    finally:
        conn.close()


def load_nsys_kernels(sqlite_file: Path, wall_clock_start_us: int = None) -> List[GpuKernel]:
    """Load GPU kernels from nsys SQLite export.
    
    Args:
        sqlite_file: Path to nsys SQLite export
        wall_clock_start_us: Wall-clock time (microseconds) corresponding to nsys time 0.
                            If None, will be extracted from TARGET_INFO_SESSION_START_TIME.
    
    Returns:
        List of GpuKernel with wall-clock timestamps
    """
    conn = sqlite3.connect(sqlite_file)
    kernels = []
    
    try:
        cursor = conn.cursor()
        
        # Check what tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        print(f"  Available tables: {len(tables)}")
        
        # Get session start time from SQLite if not provided
        if wall_clock_start_us is None:
            if "TARGET_INFO_SESSION_START_TIME" in tables:
                cursor.execute("SELECT * FROM TARGET_INFO_SESSION_START_TIME LIMIT 1")
                row = cursor.fetchone()
                if row:
                    session_start_ns = row[0]
                    wall_clock_start_us = session_start_ns // 1000
                    from datetime import datetime
                    print(f"  Session start from SQLite: {wall_clock_start_us} us")
                    print(f"    = {datetime.utcfromtimestamp(wall_clock_start_us / 1e6).isoformat()}Z")
                else:
                    raise ValueError("TARGET_INFO_SESSION_START_TIME table is empty")
            else:
                raise ValueError("No wall_clock_start_us provided and TARGET_INFO_SESSION_START_TIME not found")
        
        # Look for GPU kernels in CUPTI_ACTIVITY_KIND_KERNEL
        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
            # Get column names
            cursor.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")
            columns = {row[1]: row[0] for row in cursor.fetchall()}
            col_names = list(columns.keys())
            print(f"  Kernel table columns: {col_names[:10]}...")
            
            # Find the name column
            name_col = None
            for candidate in ["shortName", "demangledName", "name", "mangledName"]:
                if candidate in columns:
                    name_col = candidate
                    break
            
            if name_col is None:
                print(f"  WARNING: No name column found. Available: {col_names}")
                return kernels
            
            # Query kernels (limit for performance)
            query = f"""
                SELECT {name_col}, start, end, deviceId, streamId
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                ORDER BY start
            """
            
            cursor.execute(query)
            
            # Build StringIds lookup if needed
            string_lookup = {}
            if "StringIds" in tables:
                cursor2 = conn.cursor()
                cursor2.execute("SELECT id, value FROM StringIds")
                string_lookup = {row[0]: row[1] for row in cursor2.fetchall()}
                print(f"  Loaded {len(string_lookup)} string IDs")
            
            row_count = 0
            for row in cursor:
                name, start_ns, end_ns, device_id, stream_id = row
                row_count += 1
                
                # Look up kernel name from StringIds if it's an integer
                if isinstance(name, int):
                    name = string_lookup.get(name, f"kernel_{name}")
                
                # Convert nsys nanoseconds to wall-clock microseconds
                # nsys times are relative to session start (when nsys profile started)
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
            
            print(f"  Processed {row_count} kernel records")
            
        else:
            print("  WARNING: CUPTI_ACTIVITY_KIND_KERNEL table not found")
            print(f"  Available tables with 'KERNEL': {[t for t in tables if 'KERNEL' in t.upper()]}")
    
    finally:
        conn.close()
    
    return kernels


def find_kernels_in_interval(kernels: List[GpuKernel], interval_start_us: float, interval_end_us: float) -> List[GpuKernel]:
    """Find kernels that overlap with a specific time interval.
    
    A kernel overlaps if: kernel_start < interval_end AND kernel_end > interval_start
    """
    active = []
    for k in kernels:
        # Check if kernel overlaps with interval
        if k.start_us < interval_end_us and k.end_us > interval_start_us:
            active.append(k)
    return active


def find_kernels_at_time(kernels: List[GpuKernel], target_us: float, window_us: float = 2000) -> List[GpuKernel]:
    """Find kernels active within a time window (legacy, for compatibility)."""
    return find_kernels_in_interval(kernels, target_us - window_us, target_us + window_us)


def correlate(args):
    """Main correlation logic."""
    profile_dir = Path(args.profile_dir)
    gpu_filter = getattr(args, 'gpu', None)
    link_filter = getattr(args, 'link', None)
    direction_filter = getattr(args, 'direction', None)
    if direction_filter:
        direction_filter = direction_filter.upper()
    top_n = getattr(args, 'top', 10)
    
    print("=" * 60)
    print("NVLink + nsys Correlation")
    print("=" * 60)
    
    # Show active filters
    if gpu_filter is not None or link_filter is not None or direction_filter is not None:
        print("\nFilters:")
        if gpu_filter is not None:
            print(f"  GPU device: {gpu_filter}")
        if link_filter is not None:
            print(f"  NVLink ID: {link_filter}")
        if direction_filter is not None:
            print(f"  Direction: {direction_filter}")
    
    # Load timing markers
    print("\n[1] Loading timing markers...")
    
    # nsys_start is when nsys profile started (nsys timestamps are relative to this)
    nsys_start = load_timing_marker(profile_dir, "nsys_start")
    if nsys_start:
        print(f"  nsys start: {nsys_start.wall_clock_us} us")
        print(f"    ISO: {datetime.utcfromtimestamp(nsys_start.wall_clock_us / 1e6).isoformat()}Z")
    
    # profile_start is when inference profiling started (after model load)
    profile_start = load_timing_marker(profile_dir, "profile_start")
    if profile_start:
        print(f"  Profile start: {profile_start.wall_clock_us} us")
        print(f"    ISO: {datetime.utcfromtimestamp(profile_start.wall_clock_us / 1e6).isoformat()}Z")
    
    # Use nsys_start for timestamp conversion (or fall back to profile_start)
    nsys_reference = nsys_start if nsys_start else profile_start
    if not nsys_reference:
        print("  WARNING: No timing markers found")
        nsys_reference = TimingMarker("inferred", 0, 0, 0)
    
    # Load NVLink trace
    print("\n[2] Loading NVLink trace...")
    nvlink_file = profile_dir / "nvlink_trace.bin"
    nvlink_samples = []
    
    if nvlink_file.exists():
        nvlink_samples = load_nvlink_trace(nvlink_file)
        print(f"  Loaded {len(nvlink_samples)} samples")
        
        if nvlink_samples:
            valid_samples = [s for s in nvlink_samples if s.nvml_timestamp_us > 0]
            nvlink_min = min(s.nvml_timestamp_us for s in valid_samples)
            nvlink_max = max(s.nvml_timestamp_us for s in valid_samples)
            
            print(f"  Time range: {nvlink_min} - {nvlink_max} us")
            print(f"    {datetime.utcfromtimestamp(nvlink_min / 1e6).isoformat()}Z")
            print(f"    {datetime.utcfromtimestamp(nvlink_max / 1e6).isoformat()}Z")
            print(f"  Duration: {(nvlink_max - nvlink_min) / 1e6:.2f} seconds")
            
            # Use first NVLink timestamp if no timing marker
            if nsys_reference.wall_clock_us == 0:
                nsys_reference = TimingMarker("inferred", nvlink_min * 1000, nvlink_min, 0)
                print(f"  Using first NVLink sample as nsys reference: {nvlink_min} us")
    else:
        print(f"  ERROR: NVLink trace not found: {nvlink_file}")
        return
    
    # Load nsys SQLite
    print("\n[3] Loading nsys kernels...")
    sqlite_file = profile_dir / "vllm_profile.sqlite"
    
    if not sqlite_file.exists():
        # Try to find any .sqlite file
        sqlite_files = list(profile_dir.glob("*.sqlite"))
        if sqlite_files:
            sqlite_file = sqlite_files[0]
            print(f"  Using: {sqlite_file}")
        else:
            print(f"  ERROR: No SQLite file found in {profile_dir}")
            print("  Run: nsys export --type sqlite --output <output> <input.nsys-rep>")
            return
    
    # Get session start time directly from SQLite (more reliable than timing marker)
    sqlite_session_start_us = get_nsys_session_start(sqlite_file)
    if sqlite_session_start_us:
        print(f"  Session start from SQLite: {sqlite_session_start_us} us")
        print(f"    = {datetime.utcfromtimestamp(sqlite_session_start_us / 1e6).isoformat()}Z")
        
        # Compare with our timing marker for verification
        if nsys_reference.wall_clock_us > 0:
            diff_ms = (nsys_reference.wall_clock_us - sqlite_session_start_us) / 1000
            print(f"  Timing marker diff: {diff_ms:.3f} ms")
        
        # Use SQLite timestamp (authoritative)
        nsys_start_us = sqlite_session_start_us
    else:
        print("  WARNING: Could not extract session start from SQLite, using timing marker")
        nsys_start_us = nsys_reference.wall_clock_us
    
    # Load kernels - pass None to auto-extract from SQLite
    kernels = load_nsys_kernels(sqlite_file)
    print(f"  Loaded {len(kernels)} GPU kernels")
    
    # Filter by GPU if specified
    if gpu_filter is not None:
        kernels = [k for k in kernels if k.device_id == gpu_filter]
        print(f"  After GPU {gpu_filter} filter: {len(kernels)} kernels")
    
    # Show available GPU devices
    if kernels:
        devices = sorted(set(k.device_id for k in kernels))
        print(f"  GPU devices in trace: {devices}")
    
    if kernels:
        kernel_min = min(k.start_us for k in kernels)
        kernel_max = max(k.end_us for k in kernels)
        print(f"  Kernel time range: {kernel_min:.0f} - {kernel_max:.0f} us")
        print(f"    {datetime.utcfromtimestamp(kernel_min / 1e6).isoformat()}Z")
        print(f"    {datetime.utcfromtimestamp(kernel_max / 1e6).isoformat()}Z")
    
    # Compute NVLink intervals (each interval = time between samples with delta)
    print("\n[4] Computing NVLink intervals...")
    
    # Debug: show sample value statistics first
    if nvlink_samples:
        values = [s.value for s in nvlink_samples]
        print(f"  Sample value range: {min(values):,} - {max(values):,}")
        print(f"  First 5 sample values: {values[:5]}")
        if len(values) > 5:
            print(f"  Last 5 sample values: {values[-5:]}")
    
    nvlink_intervals = compute_nvlink_intervals(nvlink_samples)
    print(f"  {len(nvlink_intervals)} intervals computed")
    
    # Debug: show interval statistics
    if nvlink_intervals:
        deltas = [iv.delta_bytes for iv in nvlink_intervals]
        durations = [iv.duration_us for iv in nvlink_intervals]
        nonzero_deltas = [d for d in deltas if d > 0]
        print(f"  Delta range: {min(deltas):,} - {max(deltas):,} bytes")
        print(f"  Non-zero deltas: {len(nonzero_deltas)} / {len(deltas)}")
        print(f"  Duration range: {min(durations):,} - {max(durations):,} us")
        if nonzero_deltas:
            print(f"  First 3 non-zero deltas: {nonzero_deltas[:3]}")
    
    # Show available link IDs and directions
    if nvlink_intervals:
        links = sorted(set(iv.link_id for iv in nvlink_intervals))
        directions = sorted(set(iv.direction for iv in nvlink_intervals))
        print(f"  NVLink IDs in trace: {links}")
        print(f"  Directions in trace: {directions}")
    
    # Filter by link if specified
    if link_filter is not None:
        nvlink_intervals = [iv for iv in nvlink_intervals if iv.link_id == link_filter]
        print(f"  After link {link_filter} filter: {len(nvlink_intervals)} intervals")
    
    # Filter by direction if specified
    if direction_filter is not None:
        nvlink_intervals = [iv for iv in nvlink_intervals if iv.direction == direction_filter]
        print(f"  After {direction_filter} filter: {len(nvlink_intervals)} intervals")
    
    if nvlink_intervals:
        # NVML reports in KiB, convert to actual bytes
        total_bytes = sum(iv.delta_bytes_actual for iv in nvlink_intervals)
        nonzero = [iv for iv in nvlink_intervals if iv.delta_bytes > 0]
        max_interval = max(nvlink_intervals, key=lambda x: x.delta_bytes)
        avg_duration = sum(iv.end_us - iv.start_us for iv in nvlink_intervals) / len(nvlink_intervals)
        print(f"  Total transferred: {total_bytes / 1e9:.3f} GB")
        print(f"  Non-zero intervals: {len(nonzero)}")
        print(f"  Average interval duration: {avg_duration/1000:.2f} ms")
        print(f"  Peak: {max_interval.delta_bytes_actual/1e6:.2f} MB in interval {datetime.utcfromtimestamp(max_interval.start_us / 1e6).strftime('%H:%M:%S.%f')[:-3]}")
    
    # Correlation analysis
    print("\n[5] Correlating NVLink intervals with GPU kernels...")
    print("=" * 60)
    
    if not nvlink_intervals or not kernels:
        print("  Cannot correlate: missing data")
        return
    
    # Check time overlap
    nvlink_min = min(iv.start_us for iv in nvlink_intervals)
    nvlink_max = max(iv.end_us for iv in nvlink_intervals)
    kernel_min = min(k.start_us for k in kernels)
    kernel_max = max(k.end_us for k in kernels)
    
    overlap_start = max(nvlink_min, kernel_min)
    overlap_end = min(nvlink_max, kernel_max)
    
    if overlap_end < overlap_start:
        print(f"\n  WARNING: No time overlap between NVLink and kernels!")
        print(f"    NVLink: {nvlink_min:.0f} - {nvlink_max:.0f} us")
        print(f"    Kernels: {kernel_min:.0f} - {kernel_max:.0f} us")
        print(f"    Gap: {(min(abs(nvlink_min - kernel_max), abs(kernel_min - nvlink_max)) / 1e6):.2f} seconds")
        return
    
    print(f"\n  Overlap window: {(overlap_end - overlap_start) / 1e6:.2f} seconds")
    
    # Filter to intervals that overlap with kernel time range
    overlap_intervals = [iv for iv in nvlink_intervals 
                         if iv.end_us >= overlap_start and iv.start_us <= overlap_end]
    print(f"  NVLink intervals in overlap: {len(overlap_intervals)}")
    
    # Filter out zero-duration intervals and sort by THROUGHPUT (not raw bytes)
    valid_intervals = [iv for iv in overlap_intervals if iv.duration_us > 0]
    top_intervals = sorted(valid_intervals, key=lambda x: -x.throughput_gbps)[:top_n * 2]
    
    # Find NCCL-related kernels
    nccl_kernels = [k for k in kernels if "nccl" in k.name.lower()]
    print(f"  NCCL kernels: {len(nccl_kernels)}")
    
    # Show throughput statistics
    if valid_intervals:
        throughputs = [iv.throughput_gbps for iv in valid_intervals if iv.throughput_gbps > 0]
        if throughputs:
            print(f"\n  Throughput stats (non-zero intervals):")
            print(f"    Max: {max(throughputs):.2f} GB/s")
            print(f"    Avg: {sum(throughputs)/len(throughputs):.2f} GB/s")
            print(f"    Count: {len(throughputs)} intervals with activity")
    
    print(f"\n  Top {top_n} NVLink activity intervals (by throughput):")
    print("-" * 60)
    
    for i, iv in enumerate(top_intervals[:top_n]):
        start_str = datetime.utcfromtimestamp(iv.start_us / 1e6).strftime('%H:%M:%S.%f')[:-3]
        end_str = datetime.utcfromtimestamp(iv.end_us / 1e6).strftime('%H:%M:%S.%f')[:-3]
        
        # Find kernels that overlap with THIS EXACT interval
        overlapping = find_kernels_in_interval(kernels, iv.start_us, iv.end_us)
        overlapping_nccl = [k for k in overlapping if "nccl" in k.name.lower()]
        
        print(f"\n  [{i+1}] {start_str} - {end_str} ({iv.duration_us/1000:.2f}ms)")
        print(f"       {iv.throughput_gbps:.2f} GB/s ({iv.delta_bytes_actual/1e6:.2f} MB), Link {iv.link_id} {iv.direction}")
        
        if overlapping_nccl:
            print(f"      NCCL kernels during interval ({len(overlapping_nccl)}):")
            for k in overlapping_nccl[:5]:
                k_start = datetime.utcfromtimestamp(k.start_us / 1e6).strftime('%H:%M:%S.%f')[:-3]
                # Calculate overlap percentage
                overlap_start_k = max(k.start_us, iv.start_us)
                overlap_end_k = min(k.end_us, iv.end_us)
                overlap_pct = 100 * (overlap_end_k - overlap_start_k) / (iv.end_us - iv.start_us) if iv.end_us > iv.start_us else 0
                print(f"        - {k.name[:50]}")
                print(f"          @ {k_start} dur={k.duration_us/1000:.2f}ms overlap={overlap_pct:.0f}%")
        elif overlapping:
            print(f"      Other kernels during interval ({len(overlapping)}):")
            for k in overlapping[:50]:
                k_start = datetime.utcfromtimestamp(k.start_us / 1e6).strftime('%H:%M:%S.%f')[:-3]
                print(f"        - {k.name[:50]} @ {k_start} ({k.duration_us/1000:.2f}ms)")
        else:
            print("      No kernels during this interval")
    
    # Summary: aggregate by kernel name
    print("\n" + "-" * 60)
    print(f"  Kernel contribution summary (top {top_n} intervals):")
    kernel_bytes = {}
    for iv in top_intervals[:top_n]:
        overlapping = find_kernels_in_interval(kernels, iv.start_us, iv.end_us)
        for k in overlapping:
            # Use actual bytes (KiB * 1024)
            kernel_bytes[k.name] = kernel_bytes.get(k.name, 0) + iv.delta_bytes_actual
    
    sorted_kernels = sorted(kernel_bytes.items(), key=lambda x: -x[1])[:top_n]
    for name, bytes_sum in sorted_kernels:
        print(f"    {bytes_sum/1e6:8.2f} MB - {name[:60]}")
    
    print("\n" + "=" * 60)
    print("Correlation complete!")


def main():
    parser = argparse.ArgumentParser(description="Correlate NVLink utilization with nsys kernels")
    parser.add_argument("profile_dir", type=str, help="Directory containing profile data")
    parser.add_argument("--gpu", "-g", type=int, default=None,
                        help="Filter to specific GPU device ID (for kernels)")
    parser.add_argument("--link", "-l", type=int, default=None,
                        help="Filter to specific NVLink link ID")
    parser.add_argument("--direction", "-d", type=str, choices=["tx", "rx", "TX", "RX"],
                        default=None, help="Filter to TX or RX only")
    parser.add_argument("--top", "-n", type=int, default=10,
                        help="Number of top intervals to show (default: 10)")
    args = parser.parse_args()
    correlate(args)


if __name__ == "__main__":
    main()

