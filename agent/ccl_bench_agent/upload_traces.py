#!/usr/bin/env python3
"""
CCL-Bench trace uploader — standalone script.

Reads the populated workload card (with a `runs:` section written by the
agent loop) and uploads each successful run's traces + the card itself to
the CCL-Bench upload server.

API:
    POST {server}/api/upload   multipart/form-data
        files         — *.json trace files
        group_name    — run label (e.g. iter003_tp4dp2pp1)
        description   — config + metrics summary
        workload_card — full workload card as a JSON string

Usage:
    # Upload all successful runs recorded in a workload card:
    python upload_traces.py --card workload_card.yaml

    # Upload a single trace directory without a card:
    python upload_traces.py --trace-dir /tmp/ccl_bench_traces/run1

    # Override server (default: $CCL_UPLOAD_URL or http://localhost:5000):
    python upload_traces.py --card workload_card.yaml --server http://host:5000

    # Dry-run (print what would be uploaded, no network calls):
    python upload_traces.py --card workload_card.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests
import yaml


DEFAULT_SERVER = os.environ.get("CCL_UPLOAD_URL", "http://localhost:5000")


# ── Core upload ────────────────────────────────────────────────────────────────

def _collect_traces(trace_dir: Path) -> list[Path]:
    return sorted(trace_dir.glob("*.json"))


def _card_to_json(card: dict) -> str:
    return json.dumps(card, default=str)


def upload_one(
    server: str,
    card: dict,
    trace_dir: Path,
    group_name: str,
    description: str = "",
    timeout: int = 120,
    dry_run: bool = False,
) -> dict:
    """
    Upload traces + workload card for one run.
    Returns the server response dict, or a synthetic dict on dry-run/skip.
    """
    traces = _collect_traces(trace_dir)
    if not traces:
        return {"status": "skipped", "reason": f"no *.json files in {trace_dir}"}

    if dry_run:
        return {
            "status": "dry-run",
            "group_name": group_name,
            "files": [t.name for t in traces],
            "description": description,
        }

    file_handles = []
    try:
        files_field = [
            ("files", (p.name, (fh := open(p, "rb")), "application/json"))
            for p in traces
            for _ in [file_handles.append(fh)]  # register handle for cleanup
        ]
        resp = requests.post(
            f"{server.rstrip('/')}/api/upload",
            files=files_field,
            data={
                "group_name":    group_name,
                "description":   description,
                "workload_card": _card_to_json(card),
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    finally:
        for fh in file_handles:
            fh.close()


# ── Multi-run upload from workload card ────────────────────────────────────────

def upload_card_runs(
    server: str,
    card_path: Path,
    *,
    only_successful: bool = True,
    dry_run: bool = False,
) -> list[dict]:
    """
    Upload every run recorded in the workload card's `runs:` section.
    Returns a list of result dicts (one per run attempted).
    """
    with open(card_path) as f:
        card = yaml.safe_load(f)

    runs = card.get("runs") or []
    if not runs:
        print("No runs found in workload card.")
        return []

    results = []
    for i, run in enumerate(runs):
        status = run.get("status", "")
        if only_successful and status != "success":
            print(f"  iter {i:03d}: skipping ({status})")
            continue

        trace_dir_str = run.get("trace_dir")
        if not trace_dir_str:
            print(f"  iter {i:03d}: skipping (no trace_dir)")
            continue

        trace_dir = Path(trace_dir_str)
        if not trace_dir.is_dir():
            print(f"  iter {i:03d}: skipping (trace_dir not found: {trace_dir})")
            continue

        config  = run.get("config", {})
        metrics = run.get("metrics", {})
        score   = run.get("score", "?")

        config_str  = "_".join(f"{k}{v}" for k, v in sorted(config.items()))
        group_name  = f"iter{i:03d}_{config_str}"
        metric_str  = "  ".join(f"{k}={v:.4g}" for k, v in metrics.items())
        description = (
            f"iter={i}  "
            + "  ".join(f"{k}={v}" for k, v in sorted(config.items()))
            + (f"  {metric_str}" if metric_str else "")
            + f"  score={score}"
        )

        traces = _collect_traces(trace_dir)
        print(f"  iter {i:03d}: {len(traces)} trace(s)  group={group_name}")

        result = upload_one(
            server=server,
            card=card,
            trace_dir=trace_dir,
            group_name=group_name,
            description=description,
            dry_run=dry_run,
        )
        result["iteration"] = i
        results.append(result)

        tag = result.get("status", "?")
        if tag == "ok":
            print(f"           → uploaded  ({result.get('group_name')})")
        else:
            print(f"           → {tag}: {result.get('reason') or result.get('files')}")

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload CCL-Bench traces to the upload server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--server", default=DEFAULT_SERVER,
        help="Upload server base URL (default: $CCL_UPLOAD_URL or http://localhost:5000)",
    )
    parser.add_argument(
        "--card",
        help="Path to workload_card.yaml (uploads all runs recorded in the card)",
    )
    parser.add_argument(
        "--trace-dir",
        help="Upload a single trace directory (requires --card or standalone)",
    )
    parser.add_argument(
        "--group", default="",
        help="Group name for --trace-dir upload (defaults to directory name)",
    )
    parser.add_argument(
        "--desc", default="",
        help="Description for --trace-dir upload",
    )
    parser.add_argument(
        "--include-failed", action="store_true",
        help="Also upload runs with non-success status (default: successful only)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be uploaded without making any network calls",
    )
    args = parser.parse_args()

    if not args.card and not args.trace_dir:
        parser.error("Provide --card, --trace-dir, or both.")

    if args.dry_run:
        print("[dry-run] No files will be uploaded.\n")

    # ── Mode 1: upload all runs from a workload card ───────────────────────────
    if args.card and not args.trace_dir:
        card_path = Path(args.card)
        print(f"Card:   {card_path}")
        print(f"Server: {args.server}\n")
        results = upload_card_runs(
            server=args.server,
            card_path=card_path,
            only_successful=not args.include_failed,
            dry_run=args.dry_run,
        )
        ok = sum(1 for r in results if r.get("status") == "ok")
        print(f"\nDone: {ok}/{len(results)} uploaded.")
        return

    # ── Mode 2: single trace directory (with optional card) ───────────────────
    trace_dir = Path(args.trace_dir)
    card: dict = {}
    if args.card:
        with open(args.card) as f:
            card = yaml.safe_load(f) or {}

    group = args.group or trace_dir.name
    traces = _collect_traces(trace_dir)
    print(f"Trace dir: {trace_dir}  ({len(traces)} file(s))")
    print(f"Group:     {group}")
    print(f"Server:    {args.server}\n")

    result = upload_one(
        server=args.server,
        card=card,
        trace_dir=trace_dir,
        group_name=group,
        description=args.desc,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    if result.get("status") not in ("ok", "dry-run", "skipped"):
        sys.exit(1)


if __name__ == "__main__":
    main()
