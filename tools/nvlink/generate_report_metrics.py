#!/usr/bin/env python3
"""
Generate consolidated metrics for the CS5470 Final Project Report.
Extracts system performance, hardware utilization, and communication efficiency.
Includes advanced burst analysis: Intensity, Duration, and Bandwidth Usage %.
"""

import argparse
import json
import sqlite3
import struct
import re
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Constants for A100-40GB on Perlmutter
THEORETICAL_MAX_GBPS = 600.0  # Bidirectional Aggregate (12 links * 50 GB/s)
BURST_THRESHOLD_GBPS = 60.0   # 10% of theoretical max as a burst threshold

# Reuse definitions from correlate_nsys_nvlink.py
FILE_MAGIC = b"NVF1"
HEADER_STRUCT = struct.Struct("<4sHHi")
NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX = 138
NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX = 139
NVML_FIELD_ID_OFFSET = 0
NVML_SCOPE_ID_OFFSET = 4
NVML_TIMESTAMP_OFFSET = 8
NVML_VALUE_OFFSET = 32

@dataclass
class NVLinkSample:
    nvml_timestamp_us: int
    field_id: int
    link_id: int
    value: int

@dataclass
class NVLinkInterval:
    start_us: int
    end_us: int
    field_id: int
    link_id: int
    delta_bytes: int
    
    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us
    
    @property
    def throughput_gbps(self) -> float:
        if self.duration_us <= 0: return 0.0
        return (self.delta_bytes * 1024 * 1e6 / self.duration_us) / 1e9

def load_nvlink_trace(trace_file: Path) -> List[NVLinkSample]:
    samples = []
    if not trace_file.exists(): return []
    with open(trace_file, "rb") as f:
        header_data = f.read(HEADER_STRUCT.size)
        if len(header_data) < HEADER_STRUCT.size: return []
        magic, version, stored_field_size, host_ts_size = HEADER_STRUCT.unpack(header_data)
        if magic != FILE_MAGIC: return []
        record_stride = host_ts_size + stored_field_size
        while True:
            record = f.read(record_stride)
            if len(record) < record_stride: break
            field_data = record[host_ts_size:]
            field_id = struct.unpack("<I", field_data[NVML_FIELD_ID_OFFSET:NVML_FIELD_ID_OFFSET+4])[0]
            scope_id = struct.unpack("<I", field_data[NVML_SCOPE_ID_OFFSET:NVML_SCOPE_ID_OFFSET+4])[0]
            nvml_ts_us = struct.unpack("<Q", field_data[NVML_TIMESTAMP_OFFSET:NVML_TIMESTAMP_OFFSET+8])[0]
            value = struct.unpack("<Q", field_data[NVML_VALUE_OFFSET:NVML_VALUE_OFFSET+8])[0]
            samples.append(NVLinkSample(nvml_ts_us, field_id, scope_id, value))
    return samples

def compute_intervals(samples: List[NVLinkSample]) -> List[NVLinkInterval]:
    by_key = {}
    for s in samples:
        key = (s.field_id, s.link_id)
        if key not in by_key: by_key[key] = []
        by_key[key].append(s)
    intervals = []
    for key_samples in by_key.values():
        key_samples.sort(key=lambda x: x.nvml_timestamp_us)
        for i in range(1, len(key_samples)):
            prev, curr = key_samples[i-1], key_samples[i]
            delta = max(0, curr.value - prev.value)
            intervals.append(NVLinkInterval(prev.nvml_timestamp_us, curr.nvml_timestamp_us, curr.field_id, curr.link_id, delta))
    return intervals

def parse_benchmark_log(log_file: Path) -> Dict:
    metrics = {}
    if not log_file.exists(): return metrics
    content = log_file.read_text()
    
    patterns = {
        "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([\d\.]+)",
        "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([\d\.]+)",
        "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([\d\.]+)",
        "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([\d\.]+)",
        "throughput_tok_s": r"Output token throughput \(tok/s\):\s+([\d\.]+)",
        "throughput_req_s": r"Request throughput \(req/s\):\s+([\d\.]+)",
        "total_input_tokens": r"Total input tokens:\s+(\d+)",
        "total_output_tokens": r"Total generated tokens:\s+(\d+)",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))
            
    return metrics

def parse_benchmark_json(json_file: Path) -> Dict:
    metrics = {}
    if not json_file.exists(): return metrics
    try:
        data = json.loads(json_file.read_text())
        mapping = {
            "mean_ttft_ms": "mean_ttft_ms",
            "p99_ttft_ms": "p99_ttft_ms",
            "mean_tpot_ms": "mean_tpot_ms",
            "p99_tpot_ms": "p99_tpot_ms",
            "throughput_tok_s": "output_throughput",
            "throughput_req_s": "request_throughput",
            "total_input_tokens": "total_input_tokens",
            "total_output_tokens": "total_output_tokens",
        }
        for csv_key, json_key in mapping.items():
            if json_key in data: metrics[csv_key] = data[json_key]
    except: pass
    return metrics

def get_kernel_stats(sqlite_file: Path, start_us: float = None, end_us: float = None) -> Dict:
    stats = {"nccl_duration_ms": 0, "total_kernel_duration_ms": 0}
    if not sqlite_file.exists(): return stats
    
    session_start_us = 0
    conn = sqlite3.connect(sqlite_file)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        if "TARGET_INFO_SESSION_START_TIME" in tables:
            cursor.execute("SELECT * FROM TARGET_INFO_SESSION_START_TIME LIMIT 1")
            row = cursor.fetchone()
            if row: session_start_us = row[0] // 1000

        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
            string_lookup = {}
            if "StringIds" in tables:
                cursor.execute("SELECT id, value FROM StringIds")
                string_lookup = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.execute("SELECT shortName, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL")
            for name_id, start_ns, end_ns in cursor:
                k_start_us = session_start_us + (start_ns / 1000)
                k_end_us = session_start_us + (end_ns / 1000)
                if start_us and k_end_us < start_us: continue
                if end_us and k_start_us > end_us: continue
                
                name = string_lookup.get(name_id, "") if isinstance(name_id, int) else name_id
                duration = (end_ns - start_ns) / 1e6
                stats["total_kernel_duration_ms"] += duration
                if "nccl" in name.lower(): stats["nccl_duration_ms"] += duration
    finally: conn.close()
    return stats

def process_directory(dir_path: Path):
    print(f"Processing {dir_path}...")
    
    # 1. Metadata
    meta_file = dir_path / "experiment_metadata.json"
    metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    
    start_file = dir_path / "timing_profile_start.json"
    end_file = dir_path / "timing_profile_end.json"
    start_ts = json.loads(start_file.read_text())["wall_clock_us"] if start_file.exists() else None
    end_ts = json.loads(end_file.read_text())["wall_clock_us"] if end_file.exists() else None

    # 2. Benchmark metrics
    bench_metrics = {}
    json_results = list(dir_path.glob("sglang-*.json")) + list(dir_path.glob("vllm-*.json"))
    if json_results:
        bench_metrics = parse_benchmark_json(json_results[0])
    
    if not bench_metrics:
        log_locations = [
            dir_path / "logs" / "server_nsys.log",
            Path("/pscratch/sd/e/emk255/ccl-bench-nvlink-utilization-1/trace_collection") / dir_path.name / "logs" / "server_nsys.log"
        ]
        for loc in log_locations:
            if loc.exists():
                bench_metrics = parse_benchmark_log(loc)
                break
    
    # 3. NVLink metrics
    nvlink_file = dir_path / "nvlink_trace.bin"
    samples = load_nvlink_trace(nvlink_file)
    intervals = compute_intervals(samples)
    
    # Advanced metrics calculation
    total_bytes = sum(iv.delta_bytes * 1024 for iv in intervals)
    
    # Aggregate into buckets for peak and burst analysis
    buckets = {}
    bucket_size_s = 0.01  # 10ms buckets
    for iv in intervals:
        t = (iv.start_us - (start_ts if start_ts else intervals[0].start_us)) / 1e6
        bucket = round(t, 2)
        buckets[bucket] = buckets.get(bucket, 0) + (iv.delta_bytes * 1024)
    
    times = sorted(buckets.keys())
    bws = [buckets[t] / bucket_size_s / 1e9 for t in times] if times else []
    
    peak_bw = max(bws, default=0)
    
    # Duration calculation
    active_duration_s = (max(iv.end_us for iv in intervals) - min(iv.start_us for iv in intervals)) / 1e6 if intervals else 1
    average_bw = (total_bytes / active_duration_s) / 1e9 if active_duration_s > 0 else 0
    
    burst_intensity = peak_bw / average_bw if average_bw > 0 else 0
    bandwidth_usage_pct = (average_bw / THEORETICAL_MAX_GBPS) * 100
    burst_duration_s = sum(bucket_size_s for bw in bws if bw > BURST_THRESHOLD_GBPS)
    
    # 4. Kernel metrics
    sqlite_files = list(dir_path.glob("*.sqlite"))
    kernel_stats = get_kernel_stats(sqlite_files[0], start_ts, end_ts) if sqlite_files else {}
    
    # 5. Phase-specific (Prefill/Decode)
    prefill_bw = 0
    decode_bw = 0
    if intervals:
        inf_start = max(start_ts, intervals[0].start_us) if start_ts else intervals[0].start_us
        prefill_end = inf_start + 1000000 
        prefill_ivs = [iv for iv in intervals if iv.start_us >= inf_start and iv.end_us <= prefill_end]
        decode_ivs = [iv for iv in intervals if iv.start_us > prefill_end]
        
        if prefill_ivs:
            prefill_bw = sum(iv.delta_bytes * 1024 for iv in prefill_ivs) * 1e6 / (prefill_end - inf_start) / 1e9
        if decode_ivs:
            last_end = max(iv.end_us for iv in decode_ivs)
            duration = last_end - prefill_end
            if duration > 0:
                decode_bw = sum(iv.delta_bytes * 1024 for iv in decode_ivs) * 1e6 / duration / 1e9

    total_kernel_ms = kernel_stats.get("total_kernel_duration_ms", 0)
    results = {
        "experiment": dir_path.name,
        "backend": metadata.get("backend", "vllm"),
        "tp": metadata.get("tensor_parallel_size"),
        "batch": metadata.get("batch_size"),
        **bench_metrics,
        "total_nvlink_gb": total_bytes / 1e9,
        "peak_nvlink_gb_s": peak_bw,
        "avg_nvlink_gb_s": average_bw,
        "burst_intensity": burst_intensity,
        "burst_duration_s": burst_duration_s,
        "bw_usage_pct": bandwidth_usage_pct,
        "prefill_avg_bw_gb_s": prefill_bw,
        "decode_avg_bw_gb_s": decode_bw,
        "nccl_contribution_pct": (kernel_stats.get("nccl_duration_ms", 0) / total_kernel_ms * 100) if total_kernel_ms > 0 else 0
    }
    
    # Plotting
    if HAS_MATPLOTLIB and times:
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(times, bws, label="Aggregate Throughput", color="royalblue", linewidth=1.5)
            
            # Theoretical Max Line
            plt.axhline(y=THEORETICAL_MAX_GBPS, color="red", linestyle="--", alpha=0.5, label=f"Theoretical Max ({THEORETICAL_MAX_GBPS} GB/s)")
            
            # Shaded Burst Region
            plt.axhline(y=BURST_THRESHOLD_GBPS, color="orange", linestyle=":", alpha=0.5, label=f"Burst Threshold ({BURST_THRESHOLD_GBPS} GB/s)")
            plt.fill_between(times, 0, bws, where=[bw > BURST_THRESHOLD_GBPS for bw in bws], color="orange", alpha=0.2)
            
            plt.title(f"NVLink Throughput Dynamics: {dir_path.name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Aggregate Bandwidth (GB/s)")
            plt.legend(loc="upper right")
            plt.grid(True, alpha=0.3)
            
            # Mark Phases
            if start_ts and intervals:
                plt.axvline(x=0, color="green", linestyle="-.", alpha=0.4)
                plt.text(0.1, peak_bw*0.9, "Prefill Start", color="green", fontsize=9)
            
            plt.tight_layout()
            plt.savefig(dir_path / "nvlink_throughput.png")
            plt.close()
        except Exception as e:
            print(f"  Warning: Plotting failed for {dir_path.name}: {e}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="Experiment directories")
    parser.add_argument("--output", "-o", default="experiment_metrics.csv", help="Output CSV file")
    args = parser.parse_args()
    
    all_results = []
    for d in args.dirs:
        if Path(d).is_dir():
            all_results.append(process_directory(Path(d)))
        
    if all_results:
        # Prioritize key identification fields at the start
        priority_keys = ["experiment", "backend", "tp", "batch"]
        all_keys = []
        for res in all_results:
            for k in res.keys():
                if k not in all_keys: all_keys.append(k)
        
        fieldnames = priority_keys + [k for k in all_keys if k not in priority_keys]
                
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSummary saved to {args.output}")

if __name__ == "__main__":
    main()
