#!/usr/bin/env python3
"""
Generate Chakra ET files from a single TPU XLA Chrome trace.

This is a TPU-specific companion to gen_chakra_et.py. It selects one clustered
jit_train_step iteration from a MaxText/JAX trace and emits one Chakra ET per TPU
rank. In kernels mode, non-collective HLO/device events become COMP_NODEs and
XLA collective events become COMM_COLL_NODEs. When the trace does not carry
communication byte counts, --fallback-comm-size is used.
"""

import argparse
import json
import re
from pathlib import Path

from chakra.schema.protobuf.et_def_pb2 import (
    ALL_GATHER,
    ALL_REDUCE,
    ALL_TO_ALL,
    BROADCAST,
    COMM_COLL_NODE,
    COMP_NODE,
    REDUCE_SCATTER,
    GlobalMetadata,
)
from chakra.schema.protobuf.et_def_pb2 import AttributeProto as ChakraAttr
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message

COMM_TYPES = [
    ("reduce-scatter", REDUCE_SCATTER),
    ("reducescatter", REDUCE_SCATTER),
    ("reduce_scatter", REDUCE_SCATTER),
    ("all-gather", ALL_GATHER),
    ("allgather", ALL_GATHER),
    ("all_gather", ALL_GATHER),
    ("all-to-all", ALL_TO_ALL),
    ("alltoall", ALL_TO_ALL),
    ("all_to_all", ALL_TO_ALL),
    ("all-reduce", ALL_REDUCE),
    ("allreduce", ALL_REDUCE),
    ("all_reduce", ALL_REDUCE),
    ("collective-permute", ALL_TO_ALL),
    ("ragged-all-to-all", ALL_TO_ALL),
]
RANK_PREFIX_RE = re.compile(r"^\[(\d+)\]\s+")


def load_events(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    events = data.get("traceEvents", data)
    return events if isinstance(events, list) else []


def duration_us(event: dict) -> float:
    args = event.get("args", {})
    if "device_duration_ps" in args:
        return float(args["device_duration_ps"]) / 1e6
    return float(event.get("dur", 0) or 0)


def event_end_us(event: dict) -> float:
    return float(event.get("ts", 0) or 0) + duration_us(event)


def comm_type(event: dict) -> int | None:
    args = event.get("args", {})
    haystack = " ".join(
        str(value).lower()
        for value in (
            event.get("name", ""),
            event.get("cat", ""),
            args.get("hlo_category", ""),
            args.get("hlo_op", ""),
            args.get("long_name", ""),
        )
    )
    for needle, ctype in COMM_TYPES:
        if needle in haystack:
            return ctype
    return None


def comm_size_bytes(event: dict, fallback: int) -> int:
    args = event.get("args", {})
    for key in (
        "bytes",
        "Bytes",
        "size_bytes",
        "message_size",
        "comm_size",
        "send_bytes",
        "recv_bytes",
    ):
        if key in args:
            try:
                value = int(float(args[key]))
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
    return fallback


def event_rank(event: dict, ranks: int) -> int | None:
    match = RANK_PREFIX_RE.match(str(event.get("name", "")))
    if match:
        rank = int(match.group(1))
        if 0 <= rank < ranks:
            return rank
    args = event.get("args", {})
    for key in ("global_device_id", "chip_id", "device", "device_id", "replica", "core"):
        if key in args:
            text = str(args[key])
            match = re.search(r"\d+", text)
            if match:
                rank = int(match.group(0))
                if 0 <= rank < ranks:
                    return rank
    return None


def cluster_step_windows(
    events: list[dict],
    marker: str = "jit_train_step",
    start_gap_us: float = 100000.0,
    min_step_duration_us: float = 100000.0,
) -> list[tuple[float, float]]:
    step_events = sorted(
        [
            event for event in events
            if event.get("ph") == "X"
            and marker in str(event.get("name", ""))
            and float(event.get("dur", 0) or 0) >= min_step_duration_us
        ],
        key=lambda event: float(event.get("ts", 0) or 0),
    )
    clusters: list[list[dict]] = []
    for event in step_events:
        start = float(event.get("ts", 0) or 0)
        if not clusters or start - max(float(e.get("ts", 0) or 0) for e in clusters[-1]) > start_gap_us:
            clusters.append([event])
        else:
            clusters[-1].append(event)
    return [
        (
            min(float(event.get("ts", 0) or 0) for event in cluster),
            max(event_end_us(event) for event in cluster),
        )
        for cluster in clusters
    ]


def make_compute_node(node_id: int, dur_us: float) -> ChakraNode:
    node = ChakraNode()
    node.id = node_id
    node.name = f"compute_gap_{node_id}"
    node.type = COMP_NODE
    node.duration_micros = max(1, int(dur_us))
    node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    node.attr.append(ChakraAttr(name="num_ops", int64_val=0))
    return node


def make_kernel_node(node_id: int, event: dict) -> ChakraNode:
    node = make_compute_node(node_id, duration_us(event))
    category = event.get("args", {}).get("hlo_category", "kernel")
    node.name = f"kernel_{str(category).replace(' ', '_')}_{node_id}"
    return node


def make_comm_node(node_id: int, event: dict, fallback_size: int) -> ChakraNode:
    ctype = comm_type(event)
    if ctype is None:
        raise ValueError("not a communication event")
    node = ChakraNode()
    node.id = node_id
    node.name = f"{str(event.get('name', 'xla_collective')).split()[0]}_{node_id}"
    node.type = COMM_COLL_NODE
    node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    node.attr.append(ChakraAttr(name="comm_type", int64_val=ctype))
    node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size_bytes(event, fallback_size)))
    return node


def write_rank_et(
    output_dir: Path,
    rank: int,
    events: list[dict],
    fallback_size: int,
    compute_model: str,
) -> int:
    nodes: list[ChakraNode] = []
    node_id = 0
    prev_end = None
    prev_node_id = None
    for event in sorted(events, key=lambda e: (float(e.get("ts", 0) or 0), duration_us(e))):
        start = float(event.get("ts", 0) or 0)
        if prev_end is not None and start > prev_end:
            gap = make_compute_node(node_id, start - prev_end)
            if prev_node_id is not None:
                gap.ctrl_deps.append(prev_node_id)
            nodes.append(gap)
            prev_node_id = node_id
            node_id += 1
        if comm_type(event) is None:
            if compute_model != "kernels":
                continue
            node = make_kernel_node(node_id, event)
        else:
            node = make_comm_node(node_id, event, fallback_size)
        if prev_node_id is not None:
            node.ctrl_deps.append(prev_node_id)
        nodes.append(node)
        prev_node_id = node_id
        node_id += 1
        prev_end = max(prev_end or start, event_end_us(event))

    et_path = output_dir / f"chakra_trace.{rank}.et"
    with et_path.open("wb") as f:
        encode_message(f, GlobalMetadata(version="0.0.4"))
        for node in nodes:
            encode_message(f, node)
    n_comm = sum(1 for node in nodes if node.type == COMM_COLL_NODE)
    n_comp = sum(1 for node in nodes if node.type == COMP_NODE)
    print(f"  rank {rank}: {n_comm} XLA collective nodes, {n_comp} compute/gap nodes -> {et_path.name}")
    return n_comm


def is_tpu_kernel_event(event: dict) -> bool:
    if event.get("ph") != "X" or duration_us(event) <= 0:
        return False
    args = event.get("args", {})
    if "jit_train_step" in str(event.get("name", "")):
        return False
    if "CommonPjRtLoadedExecutable::Execute" in str(event.get("name", "")):
        return False
    return "device_duration_ps" in args and "hlo_category" in args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--iteration-index", type=int, default=1,
                        help="0-based clustered iteration index")
    parser.add_argument("--iteration-marker", default="jit_train_step",
                        help="event-name substring used to identify iteration "
                             "windows (default: jit_train_step)")
    parser.add_argument("--min-iteration-duration-us", type=float, default=100000.0,
                        help="minimum marker event duration in us")
    parser.add_argument("--compute-model", choices=["gap", "kernels"], default="kernels",
                        help="gap emits only collective nodes plus measured gaps; "
                             "kernels also emits non-collective HLO/device events")
    parser.add_argument("--fallback-comm-size", type=int, default=256 * 1024 * 1024,
                        help="bytes to use when XLA trace events do not expose message size")
    args = parser.parse_args()

    trace_path = Path(args.trace_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(trace_path)
    windows = cluster_step_windows(
        events,
        marker=args.iteration_marker,
        min_step_duration_us=args.min_iteration_duration_us,
    )
    if not windows:
        raise SystemExit(f"No {args.iteration_marker} iteration windows found")
    if args.iteration_index >= len(windows):
        raise SystemExit(
            f"Requested iteration {args.iteration_index}, but only found {len(windows)} windows"
        )
    start, end = windows[args.iteration_index]
    print(
        f"[gen_tpu_xla_chakra_et] Selected iteration {args.iteration_index}: "
        f"{start:.3f}us..{end:.3f}us marker={args.iteration_marker}"
    )

    selected_events = [
        event for event in events
        if start <= float(event.get("ts", 0) or 0) < end
        and (
            comm_type(event) is not None
            or (args.compute_model == "kernels" and is_tpu_kernel_event(event))
        )
    ]
    if not selected_events:
        raise SystemExit("No XLA device events found in selected iteration")

    events_by_rank = {rank: [] for rank in range(args.ranks)}
    unranked = []
    for event in selected_events:
        rank = event_rank(event, args.ranks)
        if rank is None:
            if comm_type(event) is not None:
                unranked.append(event)
        else:
            events_by_rank[rank].append(event)
    if unranked:
        # Some TPU traces only expose a single logical XLA timeline. Mirror it to
        # every rank so AstraSim can model collective synchronization across chips.
        for rank in range(args.ranks):
            events_by_rank[rank].extend(unranked)

    total = 0
    for rank in range(args.ranks):
        if not events_by_rank[rank]:
            print(f"  rank {rank}: no XLA collective events")
            continue
        total += write_rank_et(
            output_dir, rank, events_by_rank[rank],
            args.fallback_comm_size, args.compute_model
        )
    print(f"[gen_tpu_xla_chakra_et] Done. Total collective nodes: {total}")


if __name__ == "__main__":
    main()
