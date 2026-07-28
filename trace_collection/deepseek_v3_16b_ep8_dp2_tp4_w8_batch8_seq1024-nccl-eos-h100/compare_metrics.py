#!/usr/bin/env python3
"""Compare representative CCL-Bench metrics for matched A100 and H100 WL5."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


A100_NAME = "deepseek_v3_16b_ep8_dp2_tp4_w8_batch8_seq1024-nccl-perlmutter"
PROFILED_LOG_STEP = 4
LOCAL_BATCH_SIZE = 4
SEQUENCE_LENGTH = 1024
NON_DATA_PARALLEL_SIZE = 4  # TP × PP × CP = 4 × 1 × 1
METRICS = (
    ("avg_step_time", "Average step time", "s", "lower"),
    ("mfu", "MFU", "%", "higher"),
    ("mean_sm_coverage", "Mean SM coverage", "%", "higher"),
    ("dominant_kernel_concentration", "Dominant kernel", "%", "lower"),
    ("moe_fraction", "MoE kernel fraction", "%", "context"),
    ("average_memory_bandwidth", "Memory bandwidth", "GB/s", "higher"),
    ("memory_transfer_overhead", "Memory-transfer overhead", "%", "lower"),
    ("communication_fraction", "Communication fraction", "%", "lower"),
    ("total_communication_time", "Communication time", "s", "lower"),
    ("compute_comm_overlap", "Compute/communication overlap", "%", "higher"),
    ("load_imbalance_ratio", "Load imbalance", "ratio", "lower"),
)


def load_a100(repo: Path) -> dict[str, float | None]:
    with (repo / "website" / "benchmark_data.json").open(encoding="utf-8") as handle:
        rows = json.load(handle)["rows"]
    for row in rows:
        if row["metadata"]["workload_name"] == A100_NAME:
            return row["metrics"]
    raise RuntimeError(f"A100 baseline row not found: {A100_NAME}")


def prepare_trace_view(trace_root: Path, view: Path) -> Path:
    """Put the workload card and all rank traces in one analyzer directory."""
    yaml_files = sorted(trace_root.rglob("*.yaml"))
    rank_files = sorted(trace_root.rglob("rank*_trace.json"))
    if not yaml_files:
        raise RuntimeError(f"No workload YAML found below {trace_root}")
    if not rank_files:
        raise RuntimeError(f"No rank trace JSON found below {trace_root}")

    os.symlink(yaml_files[0], view / yaml_files[0].name)
    for trace_file in rank_files:
        destination = view / trace_file.name
        if destination.exists():
            raise RuntimeError(f"Duplicate rank trace name: {trace_file.name}")
        os.symlink(trace_file, destination)
    return view


def load_log_step_time(trace_root: Path) -> float:
    """Derive profiled-step time from TorchTitan's per-device TPS log."""
    pattern = re.compile(
        rf"step:\s*{PROFILED_LOG_STEP}\b.*?\btps:\s*([\d,]+)"
    )
    step_times = []
    tokens_per_non_dp_rank = (
        LOCAL_BATCH_SIZE * SEQUENCE_LENGTH / NON_DATA_PARALLEL_SIZE
    )
    for log_file in sorted(trace_root.rglob("node-rank-*.log")):
        for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                tps = float(match.group(1).replace(",", ""))
                step_times.append(tokens_per_non_dp_rank / tps)

    if not step_times:
        raise RuntimeError(
            f"No TorchTitan TPS entries for step {PROFILED_LOG_STEP} below "
            f"{trace_root}"
        )
    return statistics.mean(step_times)


def calculate(repo: Path, trace: Path, metric: str) -> float | None:
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "main.py"),
            "--trace",
            str(trace),
            "--metric",
            metric,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        value = float(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None
    return None if value < 0 else value


def fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.5g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--hopper-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    a100 = load_a100(args.repo)
    lines = [
        "# Matched WL5 DeepSeek-V3-16B EP8 architecture comparison",
        "",
        "Fixed workload/execution settings: TorchTitan commit 4a57715, BF16, "
        "NCCL, 2 nodes × 4 GPUs, TP4/DP-shard2/EP8/PP1, batch 8, sequence "
        "1024, six steps. Hardware platform: A100/Perlmutter → H100/Eos.",
        "",
        "H100 step time is derived from TorchTitan's step-4 TPS log across all "
        "eight ranks; the remaining H100 metrics are calculated from Kineto traces.",
        "",
        "| Metric | A100 | H100 | H100 / A100 | Unit | Preferred |",
        "|---|---:|---:|---:|---|---|",
    ]
    h100_log_step_time = load_log_step_time(args.hopper_trace)
    result_json: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix="ccl-bench-metrics-") as temp_dir:
        trace_view = prepare_trace_view(args.hopper_trace, Path(temp_dir))
        for key, label, unit, direction in METRICS:
            print(f"Calculating {key}...", flush=True)
            old = a100.get(key)
            new = (
                h100_log_step_time
                if key == "avg_step_time"
                else calculate(args.repo, trace_view, key)
            )
            ratio = None if old in (None, 0) or new is None else new / old
            lines.append(
                f"| {label} | {fmt(old)} | {fmt(new)} | {fmt(ratio)} | "
                f"{unit} | {direction} |"
            )
            result_json[key] = {
                "a100": old,
                "h100": new,
                "h100_over_a100": ratio,
            }
            if key == "avg_step_time":
                result_json[key]["h100_source"] = (
                    "TorchTitan step-4 TPS log, mean across 8 ranks"
                )

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(result_json, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
