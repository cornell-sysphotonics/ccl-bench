#!/usr/bin/env python3
"""
GPU Trace Analysis Script for Kineto/PyTorch Profiler Traces.
Computes a comprehensive set of metrics from a single-rank trace.
"""

import json
import sys
from collections import defaultdict

TRACE_FILE = "/Users/ays57/Documents/opus/sysagent/ccl-bench/trace_collection/llama-3.1-8b-torchtitan-perlmutter/kineto_trace_0.json"
OUTPUT_FILE = "/Users/ays57/Documents/opus/sysagent/ccl-bench/trace_collection/llama-3.1-8b-torchtitan-perlmutter/metrics_output.json"

# ── Helpers ──────────────────────────────────────────────────────────────────

def dtype_bytes(dtype_str):
    """Return bytes per element for a dtype string."""
    mapping = {
        "Float": 4, "float": 4, "float32": 4, "Float32": 4,
        "Double": 8, "double": 8, "float64": 8, "Float64": 8,
        "Half": 2, "half": 2, "float16": 2, "Float16": 2,
        "BFloat16": 2, "bfloat16": 2, "bf16": 2,
        "Int": 4, "int": 4, "int32": 4, "Int32": 4,
        "Long": 8, "long": 8, "int64": 8, "Int64": 8,
        "Short": 2, "short": 2, "int16": 2, "Int16": 2,
        "Byte": 1, "byte": 1, "uint8": 1, "UInt8": 1,
        "Int8": 1, "int8": 1,
    }
    return mapping.get(dtype_str, 2)  # default to 2 bytes (bf16 common)


def merge_intervals(intervals):
    """Merge overlapping [start, end) intervals. Returns merged list and total covered duration."""
    if not intervals:
        return [], 0.0
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    total = sum(e - s for s, e in merged)
    return merged, total


def overlap_duration(intervals_a, intervals_b):
    """Compute total overlapping duration between two sets of merged intervals."""
    i, j = 0, 0
    total = 0.0
    while i < len(intervals_a) and j < len(intervals_b):
        start = max(intervals_a[i][0], intervals_b[j][0])
        end = min(intervals_a[i][1], intervals_b[j][1])
        if start < end:
            total += end - start
        if intervals_a[i][1] < intervals_b[j][1]:
            i += 1
        else:
            j += 1
    return total


def classify_nccl_op(name, collective_name=""):
    """Classify an NCCL kernel into operation type."""
    cn = collective_name.lower() if collective_name else ""
    n = name.lower()
    if "allgather" in n or "all_gather" in cn or "allgather" in cn:
        return "AllGather"
    elif "reducescatter" in n or "reduce_scatter" in cn:
        return "ReduceScatter"
    elif "allreduce" in n or "all_reduce" in cn or "allreduce" in cn:
        return "AllReduce"
    elif "alltoall" in n or "all_to_all" in cn:
        return "AllToAll"
    elif "sendrecv" in n or "send" in cn:
        return "SendRecv"
    elif "broadcast" in n or "broadcast" in cn:
        return "Broadcast"
    else:
        return "Other_NCCL"


def classify_compute_kernel(name):
    """Classify a non-NCCL kernel into compute categories."""
    n = name.lower()
    if "flash_fwd" in n or ("flash" in n and "attention" in n and "bwd" not in n and "backward" not in n):
        return "attention"
    elif "flash_bwd" in n or ("flash" in n and ("bwd" in n or "backward" in n)):
        return "attention_backward"
    elif any(p in n for p in ["gemm", "cutlass", "ampere_bf16", "sm80_xmma", "sm90_xmma", "matmul", "cublas"]):
        return "gemm"
    elif any(p in n for p in ["elementwise", "vectorized", "silu", "gelu", "relu", "swish"]):
        return "elementwise"
    elif any(p in n for p in ["layernorm", "layer_norm", "rmsnorm", "rms_norm"]):
        return "normalization"
    elif any(p in n for p in ["softmax"]):
        return "softmax"
    elif any(p in n for p in ["moe", "expert", "topk", "gating"]):
        return "moe"
    elif any(p in n for p in ["memcpy", "memset"]):
        return "memory_transfer"
    elif "copy" in n:
        return "memory_transfer"
    elif "reduce" in n:
        return "reduction"
    elif "adam" in n or "sgd" in n or "optimizer" in n or "multi_tensor_apply" in n:
        return "optimizer"
    else:
        return "other_compute"


# ── Main Analysis ────────────────────────────────────────────────────────────

def main():
    print("Loading trace file...")
    with open(TRACE_FILE, "r") as f:
        data = json.load(f)
    print("Trace loaded.")

    events = data.get("traceEvents", [])
    dist_info = data.get("distributedInfo", {})
    device_props = data.get("deviceProperties", [])

    # ── Extract basic info ───────────────────────────────────────────────
    rank = dist_info.get("rank", None)
    world_size = dist_info.get("world_size", None)
    pg_config = dist_info.get("pg_config", [])

    gpu_name = device_props[0]["name"] if device_props else "Unknown"
    num_sms = device_props[0].get("numSms", None) if device_props else None

    # ── Separate events by category ──────────────────────────────────────
    kernel_events = []
    nccl_kernel_events = []
    non_nccl_kernel_events = []
    gpu_user_ann = []

    for e in events:
        cat = e.get("cat", "")
        if cat == "kernel":
            kernel_events.append(e)
            if "nccl" in e.get("name", "").lower():
                nccl_kernel_events.append(e)
            else:
                non_nccl_kernel_events.append(e)
        elif cat == "gpu_user_annotation":
            gpu_user_ann.append(e)

    # ── ProfilerStep (iteration) ─────────────────────────────────────────
    profiler_steps = [e for e in events
                      if "ProfilerStep" in e.get("name", "") and e.get("cat") == "user_annotation"]
    iteration_wall_clock_time_us = None
    if profiler_steps:
        iteration_wall_clock_time_us = profiler_steps[0].get("dur")

    # Also get GPU-side profiler step
    gpu_profiler_steps = [e for e in events
                          if "ProfilerStep" in e.get("name", "") and e.get("cat") == "gpu_user_annotation"]
    gpu_iteration_dur_us = gpu_profiler_steps[0]["dur"] if gpu_profiler_steps else None

    # ── 1. coll_call_num ─────────────────────────────────────────────────
    coll_call_num = len(nccl_kernel_events)

    # ── Kernel time calculations ─────────────────────────────────────────
    total_kernel_dur_sum = sum(e.get("dur", 0) for e in kernel_events)
    total_nccl_dur_sum = sum(e.get("dur", 0) for e in nccl_kernel_events)
    total_non_nccl_dur_sum = sum(e.get("dur", 0) for e in non_nccl_kernel_events)

    # Merged kernel intervals for true utilization
    all_kernel_intervals = [(e["ts"], e["ts"] + e["dur"]) for e in kernel_events if "ts" in e and "dur" in e]
    _, total_kernel_time_merged = merge_intervals(all_kernel_intervals)

    nccl_intervals = [(e["ts"], e["ts"] + e["dur"]) for e in nccl_kernel_events if "ts" in e and "dur" in e]
    merged_nccl, total_nccl_time_merged = merge_intervals(nccl_intervals)

    non_nccl_intervals = [(e["ts"], e["ts"] + e["dur"]) for e in non_nccl_kernel_events if "ts" in e and "dur" in e]
    merged_non_nccl, total_non_nccl_time_merged = merge_intervals(non_nccl_intervals)

    # ── 4. total_communication_time (us) ─────────────────────────────────
    total_communication_time_us = total_nccl_dur_sum

    # ── 5. total_kernel_time (us) ────────────────────────────────────────
    total_kernel_time_us = total_kernel_dur_sum

    # ── 3. communication_ratio / 6. communication_fraction ───────────────
    communication_ratio = (total_nccl_dur_sum / total_kernel_dur_sum * 100) if total_kernel_dur_sum > 0 else 0

    # ── 2. break_down_steps ──────────────────────────────────────────────
    # Classify all kernels
    breakdown = defaultdict(lambda: {"count": 0, "duration_us": 0.0})

    for e in nccl_kernel_events:
        op_type = classify_nccl_op(e.get("name", ""), e.get("args", {}).get("Collective name", ""))
        key = f"communication_{op_type}"
        breakdown[key]["count"] += 1
        breakdown[key]["duration_us"] += e.get("dur", 0)

    for e in non_nccl_kernel_events:
        cat = classify_compute_kernel(e.get("name", ""))
        breakdown[cat]["count"] += 1
        breakdown[cat]["duration_us"] += e.get("dur", 0)

    # Compute percentages
    break_down_steps = {}
    for key, val in sorted(breakdown.items(), key=lambda x: -x[1]["duration_us"]):
        break_down_steps[key] = {
            "count": val["count"],
            "duration_us": round(val["duration_us"], 2),
            "percentage_of_total_kernel_time": round(val["duration_us"] / total_kernel_dur_sum * 100, 4) if total_kernel_dur_sum > 0 else 0
        }

    # ── 7. moe_fraction ──────────────────────────────────────────────────
    moe_dur = breakdown.get("moe", {}).get("duration_us", 0)
    moe_fraction = (moe_dur / total_kernel_dur_sum * 100) if total_kernel_dur_sum > 0 else 0

    # ── 8. dominant_kernel_concentration ──────────────────────────────────
    # Find the single kernel name that takes most time
    kernel_name_dur = defaultdict(float)
    for e in kernel_events:
        kernel_name_dur[e.get("name", "unknown")] += e.get("dur", 0)

    dominant_kernel_name = max(kernel_name_dur, key=kernel_name_dur.get) if kernel_name_dur else "unknown"
    dominant_kernel_dur = kernel_name_dur[dominant_kernel_name]
    dominant_kernel_concentration = (dominant_kernel_dur / total_kernel_dur_sum * 100) if total_kernel_dur_sum > 0 else 0

    # ── 9. aggregate_gpu_utilization ─────────────────────────────────────
    # GPU utilization = fraction of wall clock time where at least one kernel is running
    if gpu_iteration_dur_us and gpu_iteration_dur_us > 0:
        wall_clock_for_util = gpu_iteration_dur_us
    elif iteration_wall_clock_time_us and iteration_wall_clock_time_us > 0:
        wall_clock_for_util = iteration_wall_clock_time_us
    else:
        wall_clock_for_util = max(e["ts"] + e["dur"] for e in kernel_events) - min(e["ts"] for e in kernel_events) if kernel_events else 1

    aggregate_gpu_utilization = (total_kernel_time_merged / wall_clock_for_util * 100)

    # ── 10. mean_sm_coverage ─────────────────────────────────────────────
    # est. achieved occupancy % is available per kernel; use warps per SM / max warps per SM
    # or blocks per SM as a proxy
    sm_coverages = []
    for e in kernel_events:
        args = e.get("args", {})
        occ = args.get("est. achieved occupancy %", None)
        if occ is not None and occ > 0:
            sm_coverages.append(occ)
        else:
            # Use blocks per SM as proxy for SM coverage
            bpsm = args.get("blocks per SM", 0)
            if bpsm and bpsm > 0 and num_sms:
                # blocks per SM > 0 means at least that fraction of SMs are used
                grid = args.get("grid", [1, 1, 1])
                total_blocks = 1
                for g in grid:
                    total_blocks *= g
                sm_used_frac = min(total_blocks / num_sms, 1.0) * 100
                sm_coverages.append(sm_used_frac)

    mean_sm_coverage = (sum(sm_coverages) / len(sm_coverages)) if sm_coverages else None

    # ── 11. memory_transfer_overhead ─────────────────────────────────────
    mem_transfer_dur = breakdown.get("memory_transfer", {}).get("duration_us", 0)
    memory_transfer_overhead = (mem_transfer_dur / total_kernel_dur_sum * 100) if total_kernel_dur_sum > 0 else 0

    # ── 12. average_memory_bandwidth ─────────────────────────────────────
    # Cannot compute from trace without memory transaction counters
    average_memory_bandwidth = None  # Requires NSight Compute or memory counters

    # ── 13. compute_bound_fraction ───────────────────────────────────────
    # Heuristic: GEMM + attention kernels are compute-bound
    compute_bound_dur = sum(breakdown.get(k, {}).get("duration_us", 0)
                            for k in ["gemm", "attention", "attention_backward"])
    compute_bound_fraction = (compute_bound_dur / total_kernel_dur_sum * 100) if total_kernel_dur_sum > 0 else 0

    # ── 14. memory_bound_fraction ────────────────────────────────────────
    # Heuristic: elementwise, normalization, softmax, reduction, memory_transfer are memory-bound
    memory_bound_dur = sum(breakdown.get(k, {}).get("duration_us", 0)
                           for k in ["elementwise", "normalization", "softmax", "reduction", "memory_transfer"])
    memory_bound_fraction = (memory_bound_dur / total_kernel_dur_sum * 100) if total_kernel_dur_sum > 0 else 0

    # ── 15. load_imbalance_ratio ─────────────────────────────────────────
    load_imbalance_ratio = None  # Requires multi-rank data

    # ── 16. communication_overlap_ratio / 9_orig. comm_comp_overlap ──────
    # Overlap = time where NCCL kernels run concurrently with compute kernels
    comm_comp_overlap_dur = overlap_duration(merged_nccl, merged_non_nccl)
    communication_overlap_ratio = (comm_comp_overlap_dur / total_nccl_time_merged * 100) if total_nccl_time_merged > 0 else 0
    comm_comp_overlap_pct = communication_overlap_ratio

    # ── comm_kernel_breakdown (per collective type, per process group) ────
    nccl_breakdown = defaultdict(lambda: {"count": 0, "duration_us": 0.0, "total_bytes": 0})
    nccl_pg_breakdown = defaultdict(lambda: defaultdict(lambda: {"count": 0, "duration_us": 0.0, "total_bytes": 0}))

    for e in nccl_kernel_events:
        args = e.get("args", {})
        coll_name = args.get("Collective name", classify_nccl_op(e.get("name", "")))
        op_type = classify_nccl_op(e.get("name", ""), coll_name)
        dur = e.get("dur", 0)

        # Calculate data size
        in_nelems = args.get("In msg nelems", 0)
        out_nelems = args.get("Out msg nelems", 0)
        dt = args.get("dtype", "BFloat16")
        elem_bytes = dtype_bytes(dt)

        # For AllGather: data moved = out_nelems * elem_bytes (output is gathered)
        # For ReduceScatter: data moved = in_nelems * elem_bytes (input is scattered)
        # For AllReduce: data moved = in_nelems * elem_bytes * 2 (reduce + broadcast)
        # For SendRecv: data moved = max(in_nelems, out_nelems) * elem_bytes
        if op_type == "AllGather":
            data_bytes = out_nelems * elem_bytes
        elif op_type == "ReduceScatter":
            data_bytes = in_nelems * elem_bytes
        elif op_type == "AllReduce":
            data_bytes = in_nelems * elem_bytes * 2
        elif op_type == "SendRecv":
            data_bytes = max(in_nelems, out_nelems) * elem_bytes
        else:
            data_bytes = max(in_nelems, out_nelems) * elem_bytes

        nccl_breakdown[op_type]["count"] += 1
        nccl_breakdown[op_type]["duration_us"] += dur
        nccl_breakdown[op_type]["total_bytes"] += data_bytes

        pg_name = args.get("Process Group Description", args.get("Process Group Name", "unknown"))
        nccl_pg_breakdown[pg_name][op_type]["count"] += 1
        nccl_pg_breakdown[pg_name][op_type]["duration_us"] += dur
        nccl_pg_breakdown[pg_name][op_type]["total_bytes"] += data_bytes

    comm_kernel_breakdown = {}
    for op, val in sorted(nccl_breakdown.items(), key=lambda x: -x[1]["duration_us"]):
        bw = (val["total_bytes"] / (val["duration_us"] * 1e-6)) / 1e9 if val["duration_us"] > 0 else 0
        comm_kernel_breakdown[op] = {
            "count": val["count"],
            "duration_us": round(val["duration_us"], 2),
            "total_bytes": val["total_bytes"],
            "avg_bandwidth_GBps": round(bw, 2),
            "percentage_of_total_kernel_time": round(val["duration_us"] / total_kernel_dur_sum * 100, 4) if total_kernel_dur_sum > 0 else 0
        }

    # Per process group breakdown
    comm_kernel_breakdown_per_pg = {}
    for pg, ops in nccl_pg_breakdown.items():
        comm_kernel_breakdown_per_pg[pg] = {}
        for op, val in sorted(ops.items(), key=lambda x: -x[1]["duration_us"]):
            bw = (val["total_bytes"] / (val["duration_us"] * 1e-6)) / 1e9 if val["duration_us"] > 0 else 0
            comm_kernel_breakdown_per_pg[pg][op] = {
                "count": val["count"],
                "duration_us": round(val["duration_us"], 2),
                "total_bytes": val["total_bytes"],
                "avg_bandwidth_GBps": round(bw, 2)
            }

    # ── estimated_bandwidth (aggregate) ──────────────────────────────────
    total_nccl_bytes = sum(v["total_bytes"] for v in nccl_breakdown.values())
    estimated_bandwidth_GBps = (total_nccl_bytes / (total_nccl_dur_sum * 1e-6)) / 1e9 if total_nccl_dur_sum > 0 else 0

    # ── communication_overhead (fraction of total kernel time in NCCL) ───
    communication_overhead = communication_ratio / 100.0  # as fraction

    # ── traffic_window: time intervals between NCCL calls per parallelism ─
    # Group NCCL events by process group, sort by timestamp, compute gaps
    nccl_events_by_pg = defaultdict(list)
    for e in nccl_kernel_events:
        args = e.get("args", {})
        pg = args.get("Process Group Description", args.get("Process Group Name", "unknown"))
        nccl_events_by_pg[pg].append(e)

    traffic_window = {}
    for pg, evts in nccl_events_by_pg.items():
        evts_sorted = sorted(evts, key=lambda x: x["ts"])
        gaps = []
        for i in range(1, len(evts_sorted)):
            prev_end = evts_sorted[i-1]["ts"] + evts_sorted[i-1].get("dur", 0)
            curr_start = evts_sorted[i]["ts"]
            gap = curr_start - prev_end
            if gap > 0:
                gaps.append(gap)
        traffic_window[pg] = {
            "num_calls": len(evts_sorted),
            "num_gaps": len(gaps),
            "mean_inter_call_gap_us": round(sum(gaps) / len(gaps), 2) if gaps else 0,
            "min_gap_us": round(min(gaps), 2) if gaps else 0,
            "max_gap_us": round(max(gaps), 2) if gaps else 0,
            "total_span_us": round(evts_sorted[-1]["ts"] + evts_sorted[-1].get("dur", 0) - evts_sorted[0]["ts"], 2) if evts_sorted else 0
        }

    # ── traffic_distribution ─────────────────────────────────────────────
    traffic_distribution = {}
    total_nccl_bytes_all = sum(v["total_bytes"] for v in nccl_breakdown.values())
    for pg, ops in nccl_pg_breakdown.items():
        pg_bytes = sum(v["total_bytes"] for v in ops.values())
        pg_dur = sum(v["duration_us"] for v in ops.values())
        pg_count = sum(v["count"] for v in ops.values())
        traffic_distribution[pg] = {
            "total_bytes": pg_bytes,
            "total_duration_us": round(pg_dur, 2),
            "call_count": pg_count,
            "bytes_fraction": round(pg_bytes / total_nccl_bytes_all, 4) if total_nccl_bytes_all > 0 else 0,
            "time_fraction": round(pg_dur / total_nccl_dur_sum, 4) if total_nccl_dur_sum > 0 else 0
        }

    # ── bandwidth_utilization ────────────────────────────────────────────
    # A100-SXM4-40GB NVLink: ~600 GB/s bidirectional (300 GB/s per direction)
    # A100 SXM4 on Perlmutter: 4 NVLink connections, ~200 GB/s aggregate per direction for NCCL
    # We'll use theoretical peak for A100 SXM4 NVLink: 600 GB/s bidirectional
    # For NCCL ring allreduce over NVLink, bus bandwidth = 2*(n-1)/n * data_size / time
    peak_nvlink_bw_GBps = 600.0  # Theoretical bidirectional NVLink BW for A100 SXM4

    # Calculate bus bandwidth for each collective type
    bandwidth_utilization = {}
    for op, val in nccl_breakdown.items():
        if val["duration_us"] > 0 and val["total_bytes"] > 0:
            # For ring-based collectives, the bus bandwidth formula applies
            # busBW = algoBW * factor
            # AllReduce: factor = 2*(n-1)/n, AllGather: (n-1)/n, ReduceScatter: (n-1)/n
            raw_bw = (val["total_bytes"] / (val["duration_us"] * 1e-6)) / 1e9
            bandwidth_utilization[op] = {
                "algo_bandwidth_GBps": round(raw_bw, 2),
                "utilization_fraction": round(raw_bw / peak_nvlink_bw_GBps, 4) if peak_nvlink_bw_GBps > 0 else None,
                "note": f"Relative to {peak_nvlink_bw_GBps} GB/s theoretical NVLink BW"
            }

    # Overall bandwidth utilization
    overall_bw_util = estimated_bandwidth_GBps / peak_nvlink_bw_GBps if peak_nvlink_bw_GBps > 0 else None

    # ── bubble_size_pipeline ─────────────────────────────────────────────
    # Detect pipeline bubbles from SendRecv idle gaps
    sendrecv_events = [e for e in nccl_kernel_events
                       if classify_nccl_op(e.get("name", ""), e.get("args", {}).get("Collective name", "")) == "SendRecv"]

    # Pipeline bubble: idle time on GPU between compute phases that's attributable to PP
    # Heuristic: Look at the SendRecv events and measure gaps where no compute is happening
    pp_events = [e for e in nccl_kernel_events
                 if e.get("args", {}).get("Process Group Description", "") == "mesh_pp"]

    if pp_events:
        pp_sorted = sorted(pp_events, key=lambda x: x["ts"])
        pp_total_dur = sum(e.get("dur", 0) for e in pp_events)
        pp_span = pp_sorted[-1]["ts"] + pp_sorted[-1].get("dur", 0) - pp_sorted[0]["ts"]
        bubble_size_pipeline = {
            "pp_communication_time_us": round(pp_total_dur, 2),
            "pp_span_us": round(pp_span, 2),
            "note": "Pipeline bubble estimation requires multi-rank data; showing PP communication stats from this rank"
        }
    else:
        bubble_size_pipeline = None

    # ── iteration_wall_clock_time ────────────────────────────────────────
    iteration_wall_clock_time_s = iteration_wall_clock_time_us / 1e6 if iteration_wall_clock_time_us else None

    # ── SM utilization (weighted by kernel duration) ─────────────────────
    weighted_sm = 0.0
    total_weight = 0.0
    for e in kernel_events:
        args = e.get("args", {})
        dur = e.get("dur", 0)
        grid = args.get("grid", [1, 1, 1])
        total_blocks = 1
        for g in grid:
            total_blocks *= g
        if num_sms and dur > 0:
            sm_util = min(total_blocks / num_sms, 1.0)
            weighted_sm += sm_util * dur
            total_weight += dur

    sm_utilization_weighted = (weighted_sm / total_weight * 100) if total_weight > 0 else None

    # ── MFU (Model FLOPs Utilization) ────────────────────────────────────
    # Cannot compute without model config (hidden_size, num_layers, seq_len, etc.)
    mfu = None  # Requires model configuration and token count

    # ── Throughput (tokens/sec) ──────────────────────────────────────────
    throughput_tokens_sec = None  # Requires token count from training config

    # ── TTFT / TPOT ──────────────────────────────────────────────────────
    ttft = None  # Training trace, not inference
    tpot = None  # Training trace, not inference

    # ── straggler ────────────────────────────────────────────────────────
    straggler = None  # Requires multi-rank data

    # ── token_to_expert_assignment ───────────────────────────────────────
    token_to_expert_assignment = None  # Not MoE model (LLaMA 3.1 8B is dense)

    # ── Process group info for context ───────────────────────────────────
    pg_info = {}
    for pg in pg_config:
        pg_info[pg.get("pg_desc", pg.get("pg_name", "unknown"))] = {
            "pg_name": pg.get("pg_name"),
            "pg_size": pg.get("pg_size"),
            "ranks": pg.get("ranks")
        }

    # ── Assemble output ─────────────────────────────────────────────────
    results = {
        "_metadata": {
            "trace_file": TRACE_FILE,
            "rank": rank,
            "world_size": world_size,
            "gpu": gpu_name,
            "num_sms": num_sms,
            "process_groups": pg_info,
            "iteration_id": profiler_steps[0].get("name", "unknown") if profiler_steps else "unknown"
        },

        # Core metrics
        "coll_call_num": coll_call_num,
        "total_communication_time": {
            "value_us": round(total_communication_time_us, 2),
            "value_s": round(total_communication_time_us / 1e6, 6),
            "merged_value_us": round(total_nccl_time_merged, 2),
            "note": "Sum of all NCCL kernel durations (value_us) and merged/non-overlapping time (merged_value_us)"
        },
        "total_kernel_time": {
            "value_us": round(total_kernel_time_us, 2),
            "value_s": round(total_kernel_time_us / 1e6, 6),
            "merged_value_us": round(total_kernel_time_merged, 2),
            "note": "Sum of all kernel durations (value_us) and merged/non-overlapping time (merged_value_us)"
        },
        "communication_ratio": {
            "value_percent": round(communication_ratio, 4),
            "note": "Percentage of total kernel duration sum spent in NCCL kernels"
        },
        "communication_fraction": {
            "value": round(communication_ratio / 100, 6),
            "note": "Fraction of total kernel duration sum spent in NCCL kernels"
        },
        "communication_overhead": {
            "value": round(communication_overhead, 6),
            "note": "Fraction of total GPU kernel time spent in NCCL communication kernels"
        },
        "break_down_steps": break_down_steps,
        "moe_fraction": {
            "value_percent": round(moe_fraction, 4),
            "note": "LLaMA 3.1 8B is a dense model; no MoE kernels detected"
        },
        "dominant_kernel_concentration": {
            "value_percent": round(dominant_kernel_concentration, 4),
            "dominant_kernel": dominant_kernel_name[:120],
            "duration_us": round(dominant_kernel_dur, 2)
        },
        "aggregate_gpu_utilization": {
            "value_percent": round(aggregate_gpu_utilization, 4),
            "wall_clock_us": round(wall_clock_for_util, 2),
            "kernel_active_us": round(total_kernel_time_merged, 2),
            "note": "Fraction of iteration wall clock where at least one GPU kernel was active"
        },
        "mean_sm_coverage": {
            "value_percent": round(mean_sm_coverage, 4) if mean_sm_coverage else None,
            "note": "Average SM coverage across all kernels based on grid dimensions"
        },
        "sm": {
            "weighted_utilization_percent": round(sm_utilization_weighted, 4) if sm_utilization_weighted else None,
            "note": "Duration-weighted SM utilization based on grid blocks vs available SMs"
        },
        "memory_transfer_overhead": {
            "value_percent": round(memory_transfer_overhead, 4),
            "note": "Percentage of total kernel time in memcpy/memset/copy operations"
        },
        "average_memory_bandwidth": {
            "value_GBps": average_memory_bandwidth,
            "note": "Cannot compute from Kineto trace alone; requires NSight Compute memory counters"
        },
        "compute_bound_fraction": {
            "value_percent": round(compute_bound_fraction, 4),
            "note": "Heuristic: GEMM + attention kernels classified as compute-bound"
        },
        "memory_bound_fraction": {
            "value_percent": round(memory_bound_fraction, 4),
            "note": "Heuristic: elementwise, normalization, softmax, reduction, memory transfer classified as memory-bound"
        },
        "load_imbalance_ratio": {
            "value": load_imbalance_ratio,
            "note": "Requires multi-rank traces to compare GPU times across ranks"
        },
        "communication_overlap_ratio": {
            "value_percent": round(communication_overlap_ratio, 4),
            "overlap_duration_us": round(comm_comp_overlap_dur, 2),
            "note": "Percentage of NCCL communication time that overlaps with compute kernels"
        },
        "comm_comp_overlap": {
            "value_percent": round(comm_comp_overlap_pct, 4),
            "note": "Same as communication_overlap_ratio"
        },
        "comm_kernel_breakdown_tpu_group_4": comm_kernel_breakdown,
        "comm_kernel_breakdown_per_process_group": comm_kernel_breakdown_per_pg,
        "estimated_bandwidth": {
            "value_GBps": round(estimated_bandwidth_GBps, 2),
            "note": "Aggregate NCCL bandwidth: total bytes transferred / total NCCL kernel time"
        },
        "bandwidth_utilization": {
            "overall_fraction": round(overall_bw_util, 4) if overall_bw_util else None,
            "per_collective": bandwidth_utilization,
            "peak_reference_GBps": peak_nvlink_bw_GBps,
            "note": "Fraction of observed bandwidth vs theoretical A100 SXM4 NVLink bandwidth"
        },
        "traffic_window": traffic_window,
        "traffic_distribution": traffic_distribution,
        "bubble_size_pipeline": bubble_size_pipeline,
        "iteration_wall_clock_time": {
            "value_us": round(iteration_wall_clock_time_us, 2) if iteration_wall_clock_time_us else None,
            "value_s": round(iteration_wall_clock_time_s, 6) if iteration_wall_clock_time_s else None,
            "gpu_side_us": round(gpu_iteration_dur_us, 2) if gpu_iteration_dur_us else None
        },
        "mfu": {
            "value": mfu,
            "note": "Requires model configuration (hidden size, layers, seq len) and token count; not available in trace"
        },
        "throughput_tokens_sec": {
            "value": throughput_tokens_sec,
            "note": "Requires token count from training configuration; not available in trace"
        },
        "straggler": {
            "value": straggler,
            "note": "Requires multi-rank traces to detect straggler ranks"
        },
        "token_to_expert_assignment": {
            "value": token_to_expert_assignment,
            "note": "LLaMA 3.1 8B is a dense model; no MoE expert assignment applicable"
        },
        "TTFT": {
            "value": ttft,
            "note": "This is a training trace, not inference; TTFT not applicable"
        },
        "TPOT": {
            "value": tpot,
            "note": "This is a training trace, not inference; TPOT not applicable"
        }
    }

    # Write output
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {OUTPUT_FILE}")
    print(f"\n=== Summary ===")
    print(f"  NCCL calls: {coll_call_num}")
    print(f"  Total kernel time (sum): {total_kernel_dur_sum/1e6:.3f}s")
    print(f"  Total NCCL time (sum): {total_nccl_dur_sum/1e6:.3f}s")
    print(f"  Communication ratio: {communication_ratio:.2f}%")
    print(f"  Comm-compute overlap: {communication_overlap_ratio:.2f}%")
    print(f"  GPU utilization: {aggregate_gpu_utilization:.2f}%")
    print(f"  Estimated NCCL BW: {estimated_bandwidth_GBps:.2f} GB/s")
    print(f"  Iteration wall clock: {iteration_wall_clock_time_s:.3f}s" if iteration_wall_clock_time_s else "  Iteration wall clock: N/A")
    print(f"  Dominant kernel: {dominant_kernel_name[:80]} ({dominant_kernel_concentration:.2f}%)")


if __name__ == "__main__":
    main()
