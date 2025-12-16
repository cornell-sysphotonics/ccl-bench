"""Profile mode detection helpers.

Centralized heuristics to decide whether a trace directory corresponds to
Torch profiler traces (Kineto / Torch ET) or Nsight Systems reports.
"""

from __future__ import annotations

from pathlib import Path


_TORCH_PATTERNS = (
    "trace.json",
    "*trace.json",
    "*_trace.json",
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


def list_torch_trace_dirs(trace_dir: str) -> list[Path]:
    """Return all directories that contain torch trace JSON files.

    Searches the base directory and ``profile_traces/iteration_*`` children.
    Sorted by iteration index when available (ascending).
    """
    base = Path(trace_dir)
    if not base.exists():
        return []

    candidate_dirs: list[Path] = []

    def has_traces(path: Path) -> bool:
        return any(path.glob("*trace.json")) or (path / "kineto_trace_0.json").exists()

    if has_traces(base):
        candidate_dirs.append(base)

    profile_root = base / "profile_traces"
    if profile_root.exists():
        iteration_dirs = [cand for cand in profile_root.glob("iteration_*") if has_traces(cand)]

        def iteration_key(p: Path) -> tuple[int, str]:
            try:
                return (int(p.name.split("_")[1]), p.name)
            except Exception:
                return (0, p.name)

        iteration_dirs.sort(key=iteration_key)
        candidate_dirs.extend(iteration_dirs)

    return candidate_dirs


def find_torch_trace_dir(trace_dir: str) -> Path | None:
    """Return the latest torch trace directory if present."""
    dirs = list_torch_trace_dirs(trace_dir)
    return dirs[-1] if dirs else None


def available_profile_modes(trace_dir: str) -> list[str]:
    """Return all detectable profile modes under the trace directory.

    Order is torch first, then nsys, so callers can prefer torch when both exist.
    """
    base = Path(trace_dir)
    if not base.exists():
        raise FileNotFoundError(f"Trace directory not found: {trace_dir}")

    modes: list[str] = []
    if _glob_any(base, _TORCH_PATTERNS):
        modes.append("torch")
    if _glob_any(base, _NSYS_PATTERNS):
        modes.append("nsys")

    if not modes:
        raise RuntimeError(
            "Unable to detect profile mode; expected torch traces (*.json) or Nsight Systems reports (*.nsys-rep|*.qdrep|report*.sqlite)."
        )

    return modes


def detect_profile_mode(trace_dir: str) -> str:
    """Detect whether the trace directory is torch or nsys.

    Raises:
        FileNotFoundError: if the directory does not exist.
        RuntimeError: if no known trace artifacts are found.
    """
    modes = available_profile_modes(trace_dir)
    # Preserve historical precedence: prefer nsys when present alone; if both, return nsys
    if "nsys" in modes:
        return "nsys"
    return modes[0]
