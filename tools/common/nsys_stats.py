"""Utilities for running `nsys stats` and parsing CSV output."""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Iterable


def _find_rep_files(trace_dir: Path) -> list[Path]:
    reps = list(trace_dir.rglob("*.nsys-rep"))
    if reps:
        return reps
    return list(trace_dir.rglob("*.qdrep"))


def _sanitize_reports(reports: Iterable[str]) -> list[str]:
    cleaned = []
    for report in reports:
        report_name = report.strip()
        if not report_name or not report_name.replace("_", "").isalnum():
            raise ValueError(f"Invalid report name: {report!r}")
        cleaned.append(report_name)
    return cleaned


async def _run_nsys_async(cmd: list[str]) -> None:
    nsys_path = shutil.which(cmd[0])
    if nsys_path is None:
        raise FileNotFoundError("`nsys` executable not found in PATH")

    cmd[0] = nsys_path
    proc = await asyncio.create_subprocess_exec(*cmd)
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        returncode = proc.returncode or 1
        raise subprocess.CalledProcessError(returncode, cmd, stderr)


def _execute_nsys(cmd: list[str]) -> None:
    asyncio.run(_run_nsys_async(cmd))


DEFAULT_REPORTS = ("cuda_api_sum", "cuda_gpu_kern_sum")


def run_nsys_stats(
    rep_path: str | Path, reports: Iterable[str] = DEFAULT_REPORTS
) -> dict[str, Any]:
    rep_path = Path(rep_path)
    if not rep_path.exists():
        raise FileNotFoundError(f"Nsight report not found: {rep_path}")

    reports_list = _sanitize_reports(reports)
    reports_arg = ",".join(reports_list)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_prefix = Path(tmpdir) / "nsys_stats"
        cmd = [
            "nsys",
            "stats",
            "--report",
            reports_arg,
            "--format",
            "csv",
            "--output",
            str(output_prefix),
            str(rep_path),
        ]
        _execute_nsys(cmd)

        results: dict[str, Any] = {}
        for report in reports_list:
            csv_path = Path(f"{output_prefix}.{report}.csv")
            if not csv_path.exists():
                continue
            with csv_path.open() as fh:
                reader = csv.DictReader(fh)
                results[report] = list(reader)
        return results


def run_nsys_stats_in_dir(
    trace_dir: str, reports: Iterable[str] = DEFAULT_REPORTS
) -> dict[str, Any]:
    base = Path(trace_dir)
    reps = _find_rep_files(base)
    if not reps:
        raise FileNotFoundError(f"No .nsys-rep/.qdrep files found under {trace_dir}")

    aggregated: dict[str, Any] = {}
    for rep in reps:
        try:
            rep_result = run_nsys_stats(rep, reports=reports)
        except subprocess.CalledProcessError:
            # Skip reps that cannot be processed (e.g., missing CUDA data)
            continue
        for key, value in rep_result.items():
            aggregated.setdefault(key, []).extend(value)
    return aggregated
