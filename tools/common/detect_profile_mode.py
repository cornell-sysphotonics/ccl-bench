"""Profile mode detection helpers.

Centralized heuristics to decide whether a trace directory corresponds to
Torch profiler traces (Kineto / Torch ET) or Nsight Systems reports.
"""

from __future__ import annotations

from pathlib import Path


_TORCH_PATTERNS = (
    "trace.json",
    "events.json",
    "local.rank*.json",
    "execution_trace_*.json",
)

_NSYS_PATTERNS = (
    "*.nsys-rep",
    "*.qdrep",
    "report*.sqlite",
)


def _glob_any(base: Path, patterns: tuple[str, ...]) -> bool:
    return any(any(base.rglob(pattern)) for pattern in patterns)


def detect_profile_mode(trace_dir: str) -> str:
    """Detect whether the trace directory is torch or nsys.

    Raises:
        FileNotFoundError: if the directory does not exist.
        RuntimeError: if no known trace artifacts are found.
    """
    base = Path(trace_dir)
    if not base.exists():
        raise FileNotFoundError(f"Trace directory not found: {trace_dir}")

    if _glob_any(base, _NSYS_PATTERNS):
        return "nsys"

    if _glob_any(base, _TORCH_PATTERNS):
        return "torch"

    raise RuntimeError(
        "Unable to detect profile mode; expected torch traces (*.json) or Nsight Systems reports (*.nsys-rep|*.qdrep|report*.sqlite)."
    )
