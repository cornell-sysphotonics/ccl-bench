"""
straggler_pct.py

For a given parallelism group (comm_stream), computes the straggler time
as a percentage of total communication time:

    straggle_pct = mean over all collective events of:
        100 x (max_comm - min_comm) / max_comm

Iterations are processed in parallel. Ranks are walked in lock-step.

Usage
-----
    python straggler_pct.py <input_dir> <comm_stream> [--num-workers N] [--output out.pkl]
"""

import argparse
import gc
import json
import multiprocessing
import os
import pickle
import re
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trace_metric_utils import load_yaml, get_trace_types

def parse_comm_events(input_path, comm_stream_to_analyze):
    with open(input_path) as f:
        data = json.load(f)

    metadata = data.get("distributedInfo", {})
    pg_config = {
        pg["pg_name"]: {"desc": pg["pg_desc"], "ranks": pg["ranks"], "streams": set()}
        for pg in metadata["pg_config"]
    }

    events = data.get("traceEvents", [])
    events.sort(key=lambda e: e.get("ts", 0))

    profiler_ts = next(
        (
            e["ts"]
            for e in events
            if "Profiler" in e.get("name", "") and e.get("cat") == "gpu_user_annotation"
        ),
        None,
    )

    for e in events:
        if "ncclDevKernel" not in e.get("name", ""):
            continue
        pg_id = e.get("args", {}).get("Process Group Name", "N/A")
        if pg_id != "N/A" and pg_id in pg_config:
            pg_config[pg_id]["streams"].add("Stream_" + str(e["args"]["stream"]))

    # Always populate pg_groups with everything for fallback lookup
    pg_groups = {}
    target_streams = set()
    default_streams = set()
    for pg in pg_config.values():
        desc  = pg["desc"]
        ranks = pg["ranks"]
        pg_groups[desc] = ranks
        if desc == comm_stream_to_analyze and len(ranks) > 1:
            target_streams = pg["streams"]
        elif desc == "default_pg" and len(ranks) > 1:
            default_streams = pg["streams"]

    # Map streams to desc: prefer exact match, fall back to default_pg
    stream_to_desc = {}
    if target_streams:
        for s in target_streams:
            stream_to_desc[s] = comm_stream_to_analyze
    else:
        for s in default_streams:
            stream_to_desc[s] = "default_pg"

    raw = []
    for e in events:
        if e.get("cat") not in ("kernel", "gpu_memcpy"):
            continue
        if profiler_ts is not None and e["ts"] <= profiler_ts:
            continue
        if "ncclDevKernel" not in e.get("name", ""):
            continue
        stream = "Stream_" + str(e["args"]["stream"])
        if stream in stream_to_desc:
            raw.append((e["ts"], e["dur"], e["name"]))

    raw.sort(key=lambda x: x[0])
    comm_events = [(dur, name) for _, dur, name in raw]

    del events, data, raw
    gc.collect()
    return comm_events, pg_groups


def straggler_pct_for_group(rank_comms):
    """
    Lock-stepped walk over all ranks simultaneously via zip.
    rank_comms: {rank -> [(dur, name), ...]}

    Returns the mean per-collective straggler percentage, where each
    collective contributes:
        100 x (max_comm - min_comm) / max_comm
    """
    per_collective_pcts = []

    for event_row in zip(*rank_comms.values()):
        names = [ev[1] for ev in event_row]
        assert len(set(names)) == 1, f"Collective mismatch across ranks: {names}"
        durs     = [ev[0] for ev in event_row]
        max_comm = max(durs)
        min_comm = min(durs)
        if max_comm > 0:
            per_collective_pcts.append(100.0 * (max_comm - min_comm) / max_comm)

    if not per_collective_pcts:
        return {
            "straggle_pct": 0.0,
            "num_events":   0,
        }

    return {
        "straggle_pct": sum(per_collective_pcts) / len(per_collective_pcts),
        "num_events":   len(per_collective_pcts),
    }


def process_iteration(subdir_path, comm_stream_to_analyze):
    subdir     = os.path.basename(subdir_path)
    file_paths = sorted(os.path.join(subdir_path, f) for f in os.listdir(subdir_path))
    print(f"[{subdir}] parsing {len(file_paths)} rank(s)...")

    rank_comms = {}
    all_groups = {}

    for fp in file_paths:
        m = re.search(r"rank(\d+)_trace\.json", os.path.basename(fp))
        if not m:
            continue
        rank = int(m.group(1))
        comm_events, pg_groups = parse_comm_events(fp, comm_stream_to_analyze)
        rank_comms[rank] = comm_events
        all_groups.update(pg_groups)

    # Primary: exact match with participant count check
    group_ranks = all_groups.get(comm_stream_to_analyze)
    if group_ranks is not None and len(group_ranks) <= 1:
        print(
            f"  [{subdir}] WARNING: '{comm_stream_to_analyze}' has only "
            f"{len(group_ranks)} participant(s) — skipping primary match"
        )
        group_ranks = None

    # Fallback: default_pg, requiring >1 participant and actual nccl traffic
    using_fallback = False
    if group_ranks is None:
        fallback_ranks = all_groups.get("default_pg")
        has_traffic = fallback_ranks and len(fallback_ranks) > 1 and any(
            len(rank_comms.get(r, [])) > 0
            for r in fallback_ranks if r in rank_comms
        )
        if has_traffic:
            print(
                f"  [{subdir}] WARNING: no valid group for '{comm_stream_to_analyze}' "
                f"— falling back to default_pg (ranks {fallback_ranks})"
            )
            group_ranks    = fallback_ranks
            using_fallback = True
        else:
            raise RuntimeError(
                f"[{subdir}] No valid group found for '{comm_stream_to_analyze}' "
                f"and no usable default_pg"
            )

    rank_comms = {r: rank_comms[r] for r in group_ranks if r in rank_comms}

    lengths = {r: len(v) for r, v in rank_comms.items()}
    if len(set(lengths.values())) != 1:
        print(f"  WARNING: unequal event counts {lengths} — trimming to min")
        min_len = min(lengths.values())
        rank_comms = {r: v[:min_len] for r, v in rank_comms.items()}

    stats = straggler_pct_for_group(rank_comms)
    label = f"{comm_stream_to_analyze} (via default_pg)" if using_fallback else comm_stream_to_analyze
    print(
        f"  [{subdir}] [{label}] straggle={stats['straggle_pct']:.2f}%  "
        f"over {stats['num_events']} collectives"
    )
    return subdir, stats


def _calc_json(directory, comm_stream):
    subdir_paths = sorted(
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    )

    worker = partial(process_iteration, comm_stream_to_analyze=comm_stream)

    all_results = {}
    if True:  # single-worker path; multiprocessing left to CLI entrypoint
        for sp in subdir_paths:
            try:
                subdir, stats = worker(sp)
                all_results[subdir] = stats
            except RuntimeError as e:
                print(f"  [skip] {e}", file=sys.stderr)

    if not all_results:
        print(f"[straggler_pct/json] No usable data in {directory}", file=sys.stderr)
        return -1.0

    # Average the per-iteration per-collective means, weighted by event count
    total_weighted = sum(s["straggle_pct"] * s["num_events"] for s in all_results.values())
    total_events   = sum(s["num_events"] for s in all_results.values())
    overall_pct    = total_weighted / total_events if total_events > 0 else 0.0

    print("\n" + "=" * 55)
    print(f"OVERALL  --  {comm_stream}")
    print(f"  straggle : {overall_pct:.2f}%")
    print(f"  iters    : {len(all_results)}")
    print("=" * 55)

    return overall_pct


def metric_cal(directory: str) -> float:
    """
    Average per-collective straggler percentage for the given parallelism group.
    Only supported for nccl (json) backends.
    Returns percent in [0, 100], or -1 if unavailable.
    """
    yaml_data   = load_yaml(directory)
    trace_types = get_trace_types(yaml_data)

    if "json" in trace_types:
        return _calc_json(directory + "/n4-tp-llama8b-a100", "default_pg")

    print(
        f"[straggler_pct] No supported trace type in {trace_types} "
        f"(only 'json'/nccl is supported)",
        file=sys.stderr,
    )
    return -1.0


def main():
    parser = argparse.ArgumentParser(
        description="Straggler time as %% of total comm time for a parallelism group."
    )
    parser.add_argument("input_dir",   help="Directory of iteration sub-directories")
    parser.add_argument("comm_stream", help="Parallelism group to analyze (e.g. mesh_tp)")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--output", default=None, help="Optional .pkl output path")
    args = parser.parse_args()
    args.num_workers = 22
    subdir_paths = sorted(
        os.path.join(args.input_dir, d)
        for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    )

    worker = partial(process_iteration, comm_stream_to_analyze=args.comm_stream)

    all_results = {}
    def collect(it):
        for subdir, stats in it:
            all_results[subdir] = stats

    if args.num_workers == 1:
        collect(map(worker, subdir_paths))
    else:
        with multiprocessing.Pool(processes=args.num_workers, maxtasksperchild=1) as pool:
            collect(pool.imap_unordered(worker, subdir_paths))


    if not all_results:
        print("[straggler_pct] No results collected.", file=sys.stderr)
        sys.exit(1)

    total_weighted = sum(s["straggle_pct"] * s["num_events"] for s in all_results.values())
    total_events   = sum(s["num_events"] for s in all_results.values())
    overall_pct    = total_weighted / total_events if total_events > 0 else 0.0

    print("\n" + "=" * 55)
    print(f"OVERALL  --  {args.comm_stream}")
    print(f"  straggle : {overall_pct:.2f}%")
    print(f"  iters    : {len(all_results)}")
    print("=" * 55)

    if args.output:
        with open(args.output, "wb") as f:
            pickle.dump({
                "per_iteration": all_results,
                "overall_pct":   overall_pct,
            }, f)
        print(f"Saved to {args.output}")

    print(overall_pct)


if __name__ == "__main__":
    main()