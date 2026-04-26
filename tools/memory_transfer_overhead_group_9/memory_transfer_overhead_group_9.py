"""
Metric: memory_transfer_overhead
Description: Percentage of step/trace time spent exclusively on memory transfers —
             i.e., memcpy/DMA intervals that do NOT overlap with any compute kernel.
             Concurrent transfers are unioned before subtraction so overlapping
             copies are not double-counted.
Unit: Percentage (%)
Returns: Float between 0-100, or -1 if data unavailable
"""

import sqlite3
import sys
import os


# ── Interval arithmetic helpers ───────────────────────────────────────────────

def _merge_intervals(intervals):
    """Return sorted, non-overlapping union of (start, end) pairs."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _subtract_intervals(base, subtract):
    """
    Remove portions of *base* intervals covered by *subtract* intervals.
    Both inputs must already be sorted and merged (non-overlapping).
    """
    result = []
    si = 0
    n = len(subtract)
    for bs, be in base:
        # Advance lower bound: skip subtract intervals that end before this base interval
        while si < n and subtract[si][1] <= bs:
            si += 1
        cur = bs
        j = si
        while j < n and subtract[j][0] < be:
            ss, se = subtract[j]
            if ss > cur:
                result.append((cur, ss))
            cur = max(cur, se)
            j += 1
        if cur < be:
            result.append((cur, be))
    return result


def _sum_intervals(intervals):
    return sum(e - s for s, e in intervals)


# ── NSYS path ─────────────────────────────────────────────────────────────────

def find_sqlite_file(path):
    """Find SQLite file in directory or return path if it's already a .sqlite file"""
    path = os.path.abspath(path)
    if os.path.isfile(path) and path.endswith('.sqlite'):
        return path
    if os.path.isdir(path):
        sqlite_files = [f for f in os.listdir(path) if f.endswith('.sqlite')]
        if len(sqlite_files) == 0:
            return None
        non_profiling = [f for f in sqlite_files if 'profiling' not in f.lower()]
        if non_profiling:
            return os.path.abspath(os.path.join(path, non_profiling[0]))
        return os.path.abspath(os.path.join(path, sqlite_files[0]))
    return None


def calculate_metric(path):
    """
    NSYS/SQLite: time in merged memcpy intervals NOT covered by any compute kernel,
    divided by total trace span.
    """
    sqlite_path = find_sqlite_file(path)
    if sqlite_path is None:
        print(f"Error: No .sqlite file found in {path}", file=sys.stderr)
        return -1

    try:
        with sqlite3.connect(sqlite_path) as conn:
            cur = conn.cursor()

            cur.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_MEMCPY'
            """)
            if cur.fetchone() is None:
                return -1

            cur.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_KERNEL'
            """)
            if cur.fetchone() is None:
                return -1

            # Memcpy intervals (manageable row count)
            cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE end > start")
            memcpy_raw = cur.fetchall()
            if not memcpy_raw:
                return -1

            # Compute kernel intervals — fetch sorted to minimise merge work
            cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end > start ORDER BY start")
            kernel_raw = cur.fetchall()

        memcpy_merged = _merge_intervals(memcpy_raw)
        kernel_merged = _merge_intervals(kernel_raw)

        pure_memcpy = _subtract_intervals(memcpy_merged, kernel_merged)
        pure_memcpy_time = _sum_intervals(pure_memcpy)

        all_starts = [s for s, _ in memcpy_merged] + ([kernel_merged[0][0]] if kernel_merged else [])
        all_ends   = [e for _, e in memcpy_merged] + ([kernel_merged[-1][1]] if kernel_merged else [])
        trace_duration = max(all_ends) - min(all_starts)

        if trace_duration == 0:
            return -1

        return float((pure_memcpy_time / trace_duration) * 100)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return -1


# ── JSON / Kineto path ────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trace_metric_utils import (
    load_yaml, get_trace_types,
    _MEMCPY_RE, _COMM_RE,
)
from json_sampling import select_json_files


def _load_json_events(path):
    """Load traceEvents from a (possibly truncated) PyTorch-profiler JSON file."""
    import json
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            idx = content.find('"traceEvents"')
            if idx == -1:
                return []
            bracket = content.find("[", idx)
            if bracket == -1:
                return []
            partial = content[bracket:]
            for suffix in ("]}",  "]}}}"):
                try:
                    data = json.loads(partial + suffix)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                return []
            if isinstance(data, list):
                return data
        except Exception:
            return []
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return []


def _calc_json(directory: str) -> float:
    """
    JSON/Kineto: per-rank non-overlapped memcpy time over rank kernel span.

    For each rank file:
      - Collect memcpy kernel intervals and compute (non-memcpy, non-comm) intervals.
      - Union-merge each set; subtract compute from memcpy union.
      - Accumulate pure memcpy time and total kernel span.
    Return summed_pure_memcpy / summed_span * 100.
    """
    json_files = select_json_files(directory)
    if not json_files:
        return -1.0

    total_pure_memcpy = 0.0
    total_span = 0.0

    for path in json_files:
        events = _load_json_events(path)
        memcpy_ivs = []
        compute_ivs = []
        span_start = None
        span_end = None

        for e in events:
            if not isinstance(e, dict) or e.get("ph") != "X":
                continue
            if e.get("cat") not in ("kernel", "Op", "gpu_op", "device_op", None):
                continue
            ts = e.get("ts")
            dur = e.get("dur")
            if ts is None or dur is None or float(dur) <= 0:
                continue
            ts, dur = float(ts), float(dur)
            iv = (ts, ts + dur)

            if span_start is None or ts < span_start:
                span_start = ts
            if span_end is None or iv[1] > span_end:
                span_end = iv[1]

            name = e.get("name", "")
            if _MEMCPY_RE.search(name):
                memcpy_ivs.append(iv)
            elif not _COMM_RE.search(name):
                compute_ivs.append(iv)

        if span_start is None or span_end is None or not memcpy_ivs:
            continue
        span = span_end - span_start
        if span <= 0:
            continue

        memcpy_merged = _merge_intervals(memcpy_ivs)
        compute_merged = _merge_intervals(compute_ivs)
        pure = _subtract_intervals(memcpy_merged, compute_merged)
        total_pure_memcpy += _sum_intervals(pure)
        total_span += span

    if total_span <= 0:
        return -1.0
    return round((total_pure_memcpy / total_span) * 100.0, 4)


# ── TPU path ──────────────────────────────────────────────────────────────────

def _calc_json_tpu(directory: str) -> float:
    """
    TPU: non-overlapped DMA time over total step execution time.

    XLA pipelines DMA with compute aggressively, so raw merged DMA time
    significantly overstates true overhead.  This function subtracts any DMA
    time that is concurrent with a leaf device-compute op (fusion, matmul, etc.),
    leaving only the DMA time during which no compute is executing.

    DMA intervals:     copy-start.N / copy-done.N device_offset_ps pairs (ps).
    Compute intervals: device ops with device_offset_ps + device_duration_ps,
                       on TPU device PIDs, excluding jit_* container events
                       (which span the entire model run) and dependency-wait.
    Denominator:       total '$core.py:331 step' dur (µs) converted to ps.
    """
    import re
    COPY_START_RE = re.compile(r"copy-start\.([\d]+)")
    COPY_DONE_RE  = re.compile(r"copy-done\.([\d]+)")
    COPY_ANY_RE   = re.compile(r"copy-(?:start|done)\.")
    JIT_RE        = re.compile(r"^jit_")
    STEP_EVENT    = "$core.py:331 step"

    json_files = select_json_files(directory)
    if not json_files:
        return -1.0

    dma_intervals: list = []
    compute_intervals: list = []
    total_step_us = 0.0

    for path in json_files:
        events = _load_json_events(path)

        # Identify TPU device PIDs from metadata in this file
        tpu_pids: set = set()
        for e in events:
            if isinstance(e, dict) and e.get("ph") == "M" and e.get("name") == "process_name":
                if "/device:TPU:" in str((e.get("args") or {}).get("name", "")):
                    tpu_pids.add(e["pid"])

        pending: dict = {}

        for e in events:
            if not isinstance(e, dict) or e.get("ph") != "X":
                continue
            name = str(e.get("name", ""))
            args  = e.get("args") or {}

            # Host-side step timing (no pid filter needed)
            if name == STEP_EVENT:
                dur = e.get("dur")
                if dur is not None:
                    total_step_us += float(dur)
                continue

            if e.get("pid") not in tpu_pids:
                continue

            dev_ps = args.get("device_offset_ps")
            dev_dur_ps = args.get("device_duration_ps")

            # DMA: track async copy-start / copy-done pairs via device_offset_ps
            if dev_ps is not None:
                m = COPY_START_RE.search(name)
                if m:
                    pending[m.group(1)] = float(dev_ps)
                    continue
                m = COPY_DONE_RE.search(name)
                if m and m.group(1) in pending:
                    t0 = pending.pop(m.group(1))
                    t1 = float(dev_ps)
                    if t1 > t0:
                        dma_intervals.append((t0, t1))
                    continue

            # Compute: leaf device ops (exclude copy events and jit_* containers)
            if dev_ps is not None and dev_dur_ps is not None:
                if COPY_ANY_RE.search(name) or JIT_RE.match(name) or "dependency-wait" in name:
                    continue
                dur = float(dev_dur_ps)
                if dur > 0:
                    compute_intervals.append((float(dev_ps), float(dev_ps) + dur))

    if total_step_us <= 0 or not dma_intervals:
        return -1.0

    dma_merged     = _merge_intervals(dma_intervals)
    compute_merged = _merge_intervals(compute_intervals)
    pure_dma       = _subtract_intervals(dma_merged, compute_merged)
    merged_ps      = _sum_intervals(pure_dma)

    total_step_ps = total_step_us * 1e6
    return round((merged_ps / total_step_ps) * 100.0, 4)


# ── Dispatcher ────────────────────────────────────────────────────────────────

def metric_cal(directory: str) -> float:
    trace_types = get_trace_types(load_yaml(directory))
    if "nsys" in trace_types:
        return calculate_metric(directory)
    if "json_tpu" in trace_types:
        return _calc_json_tpu(directory)
    if "json" in trace_types:
        return _calc_json(directory)
    print(f"[memory_transfer_overhead] Unsupported trace types {trace_types}", file=sys.stderr)
    return -1.0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python memory_transfer_overhead_group_9.py <trace_directory_or_sqlite_file>")
        sys.exit(1)
    print(metric_cal(sys.argv[1]))
