#!/usr/bin/env python3
"""
Compute per-iteration straggler delay across ranks from PyTorch Profiler JSON traces
(Chrome trace format: traceEvents with ts/dur).

Input files example:
  rank0_trace.json, rank1_trace.json, ... rank15_trace.json

Outputs:
  - CSV: one row per iteration with straggler delay and per-rank iteration end times (ms)
  - JSON summary: overall stats
"""

import argparse
import csv
import glob
import json
import os
import re
from statistics import mean, median

# Matches "ProfilerStep#12" or "ProfilerStep# 12" etc.
STEP_ID_RE = re.compile(r"ProfilerStep#\s*(\d+)", re.IGNORECASE)


def parse_rank_from_filename(path: str) -> int:
    """Parse rank id from filenames like rank0_trace.json (case-insensitive)."""
    base = os.path.basename(path)
    m = re.search(r"rank\s*(\d+)", base, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse rank id from filename: {base}")
    return int(m.group(1))


def load_trace_events(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    # keep only complete events ("X") that have ts
    out = []
    for e in events:
        if e.get("ph") != "X":
            continue
        if "ts" not in e:
            continue
        out.append(e)
    return out


def infer_time_unit_scale_to_ms(events) -> float:
    """
    Infer whether ts/dur are in microseconds or nanoseconds and return a scale factor
    that converts (ts or dur) into milliseconds.

    Common:
      - PyTorch Profiler often uses microseconds => ms = us / 1000
      - Some traces may be nanoseconds => ms = ns / 1e6
    Heuristic:
      Check typical 'dur' magnitude (median) among non-trivial events.
    """
    durs = []
    for e in events:
        dur = e.get("dur", 0)
        if isinstance(dur, (int, float)) and dur > 0:
            durs.append(float(dur))
    if not durs:
        # default to us->ms
        return 1.0 / 1000.0

    d_med = median(durs)

    # Heuristic thresholds:
    # If median dur is huge (>= 1e6), it's likely nanoseconds (1e6 ns = 1 ms)
    # If median dur is in the thousands to hundreds of thousands, likely microseconds
    if d_med >= 1e6:
        # ns -> ms
        return 1.0 / 1_000_000.0
    else:
        # us -> ms
        return 1.0 / 1000.0


def find_step_markers(events, marker_substr: str):
    """
    Find iteration markers. Default expects event names containing 'ProfilerStep#<id>'.
    Returns dict step_id -> (start_ts, end_ts) in raw time units.
    """
    markers = {}
    for e in events:
        name = str(e.get("name", ""))
        if marker_substr and marker_substr.lower() not in name.lower():
            continue

        m = STEP_ID_RE.search(name)
        if not m:
            continue

        sid = int(m.group(1))
        ts = float(e["ts"])
        dur = float(e.get("dur", 0.0))
        te = ts + dur

        if sid not in markers:
            markers[sid] = [ts, te]
        else:
            markers[sid][0] = min(markers[sid][0], ts)
            markers[sid][1] = max(markers[sid][1], te)

    # step_id -> (start, end)
    return {sid: (v[0], v[1]) for sid, v in markers.items()}


def build_iteration_windows_from_markers(step_markers: dict):
    """
    If we have >=2 steps: use start times as boundaries: [start_i, start_{i+1})
    If we have exactly 1 step: fall back to that step's own [start, end] window.
    Returns list of tuples: (iteration_id, win_start, win_end) in raw time units.
    """
    if not step_markers:
        return []

    # sort by step id
    steps = sorted([(sid, s, e) for sid, (s, e) in step_markers.items()], key=lambda x: x[0])

    # Fallback: only one step recorded
    if len(steps) == 1:
        sid, s, e = steps[0]
        if e <= s:
            return []
        return [(sid, s, e)]

    # Normal case: >= 2 steps
    windows = []
    for i in range(len(steps) - 1):
        sid, s, _e = steps[i]
        _sid_next, s_next, _e_next = steps[i + 1]
        windows.append((sid, s, s_next))
    return windows



def iteration_end_time(events, win_start, win_end):
    """
    For a given window [win_start, win_end), compute the latest (ts+dur) among events
    that overlap the window. Returns raw time units.
    """
    latest = None
    for e in events:
        ts = float(e["ts"])
        dur = float(e.get("dur", 0.0))
        te = ts + dur

        # overlap with [win_start, win_end)
        if te <= win_start or ts >= win_end:
            continue
        if latest is None or te > latest:
            latest = te
    return latest


def summarize_straggler(delays_ms):
    if not delays_ms:
        return {}
    delays_sorted = sorted(delays_ms)
    p50 = delays_sorted[len(delays_sorted) // 2]
    p90 = delays_sorted[int(0.9 * (len(delays_sorted) - 1))]
    p99 = delays_sorted[int(0.99 * (len(delays_sorted) - 1))]
    return {
        "count": len(delays_ms),
        "mean_ms": mean(delays_ms),
        "median_ms": median(delays_ms),
        "p50_ms": p50,
        "p90_ms": p90,
        "p99_ms": p99,
        "min_ms": delays_sorted[0],
        "max_ms": delays_sorted[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing rank traces")
    ap.add_argument("--pattern", default="rank*_trace.json", help='Glob pattern (default: "rank*_trace.json")')
    ap.add_argument("--marker", default="ProfilerStep#", help="Marker substring used to detect iterations")
    ap.add_argument("--out_csv", default="straggler_delay.csv", help="Output CSV filename")
    ap.add_argument("--out_summary", default="summary.json", help="Output summary JSON filename")
    ap.add_argument("--ref_rank", type=int, default=None, help="Reference rank for iteration boundaries (default: smallest rank)")
    ap.add_argument("--strict", action="store_true", help="Drop iterations where any rank has missing end time")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        raise SystemExit(f"No files matched: {os.path.join(args.dir, args.pattern)}")

    # Load events per rank
    rank_events = {}
    for p in paths:
        r = parse_rank_from_filename(p)
        rank_events[r] = load_trace_events(p)

    ranks = sorted(rank_events.keys())
    print("Detected ranks:", ranks)

    # Pick reference rank
    ref_rank = args.ref_rank if args.ref_rank is not None else ranks[0]
    if ref_rank not in rank_events:
        raise SystemExit(f"ref_rank={ref_rank} not found. Available ranks: {ranks}")

    # Infer time scale to ms from reference rank events
    scale_to_ms = infer_time_unit_scale_to_ms(rank_events[ref_rank])
    unit_guess = "ns" if abs(scale_to_ms - (1.0 / 1_000_000.0)) < 1e-12 else "us"
    print(f"Inferred time unit: {unit_guess} (scale to ms = {scale_to_ms})")

    # Build iteration windows from ref rank markers
    markers = find_step_markers(rank_events[ref_rank], args.marker)
    windows = build_iteration_windows_from_markers(markers)
    if not windows:
        raise SystemExit(
            "Could not build iteration windows.\n"
            "Try changing --marker to match your trace step events.\n"
            "Example: --marker ProfilerStep#"
        )

    # Compute per-iteration per-rank end times (ms relative to window start)
    rows = []
    delays = []

    for (it, win_s, win_e) in windows:
        per_rank_end_ms = {}
        missing = False

        for r in ranks:
            et = iteration_end_time(rank_events[r], win_s, win_e)
            if et is None:
                missing = True
                per_rank_end_ms[r] = None
            else:
                per_rank_end_ms[r] = (et - win_s) * scale_to_ms  # convert to ms and make relative

        if missing and args.strict:
            continue

        # Only compute straggler if we have at least 2 ranks with values
        valid = {r: v for r, v in per_rank_end_ms.items() if v is not None}
        if len(valid) < 2:
            continue

        fastest_rank = min(valid, key=valid.get)
        slowest_rank = max(valid, key=valid.get)
        straggler_ms = valid[slowest_rank] - valid[fastest_rank]
        delays.append(straggler_ms)

        row = {
            "iteration": it,
            "window_len_ms": (win_e - win_s) * scale_to_ms,
            "straggler_delay_ms": straggler_ms,
            "fastest_rank": fastest_rank,
            "slowest_rank": slowest_rank,
        }
        for r in ranks:
            row[f"rank{r}_iter_end_ms"] = per_rank_end_ms[r]
        rows.append(row)

    if not rows:
        raise SystemExit("No iterations produced. Check marker and trace content.")

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    summary = {
        "input_dir": os.path.abspath(args.dir),
        "pattern": args.pattern,
        "marker": args.marker,
        "ref_rank": ref_rank,
        "ranks": ranks,
        "time_unit_inferred": unit_guess,
        "iterations_written": len(rows),
        "straggler_delay_stats": summarize_straggler(delays),
    }
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(rows)} iterations -> {args.out_csv}")
    print(f"Wrote summary -> {args.out_summary}")
    print("Straggler delay stats:", summary["straggler_delay_stats"])


if __name__ == "__main__":
    main()
