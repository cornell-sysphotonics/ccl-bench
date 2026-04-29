"""
Run result cache — avoids re-executing the same (workload, config) pair.

Cache key: stable JSON of (run_script, sorted config dict).
Cache file: run_cache.json in the same directory as agent.py.

Only the fields needed to reconstruct a record are stored (metrics, score,
status, error_msg, trace_dir). The policy field is intentionally excluded
so each iteration still logs the current policy version.
"""

import json
from pathlib import Path

_CACHE_FILE = Path(__file__).parent / "run_cache.json"

# In-memory copy; populated by load() at startup.
_cache: dict[str, dict] = {}


def _make_key(workload: dict, config: dict) -> str:
    return json.dumps(
        {"run_script": workload.get("run_script", ""), "config": config},
        sort_keys=True,
    )


def load() -> None:
    """Load cache from disk into memory. Call once at agent startup."""
    global _cache
    if _CACHE_FILE.exists():
        try:
            _cache = json.loads(_CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            _cache = {}
    else:
        _cache = {}


def get(workload: dict, config: dict) -> dict | None:
    """Return cached record for (workload, config), or None on miss."""
    return _cache.get(_make_key(workload, config))


def put(workload: dict, config: dict, record: dict) -> None:
    """Store record in cache and persist to disk."""
    key = _make_key(workload, config)
    # Store only the objective fields; exclude policy (varies per iteration).
    _cache[key] = {
        "config":    record.get("config", {}),
        "metrics":   record.get("metrics", {}),
        "score":     record.get("score", float("inf")),
        "status":    record.get("status", ""),
        "error_msg": record.get("error_msg"),
        "trace_dir": record.get("trace_dir"),
    }
    _CACHE_FILE.write_text(json.dumps(_cache, indent=2))
