"""
Shared utility: list JSON files in a trace directory with optional random sampling.

If the directory contains more than `max_files` JSON files, a random sample of
`max_files` is returned to bound per-tool parse time.  The seed is fixed so that
all metrics evaluated on the same directory use the same subset of rank files.
"""

import os
import random
import sys

_MAX_JSON_FILES = 16
_SAMPLE_SEED = 42


def select_json_files(directory: str, max_files: int = _MAX_JSON_FILES) -> list:
    """
    Return a sorted list of .json file paths inside *directory*.

    If the directory contains more than *max_files* JSON files, a deterministic
    random sample of *max_files* is returned instead of the full list.

    Args:
        directory: Trace directory path.
        max_files: Maximum number of files to return (default 16).

    Returns:
        List of absolute file paths, sorted by filename.
    """
    all_files = sorted(
        os.path.join(directory, fn)
        for fn in os.listdir(directory)
        if fn.endswith(".json")
    )
    if len(all_files) > max_files:
        rng = random.Random(_SAMPLE_SEED)
        sampled = sorted(rng.sample(all_files, max_files))
        names = [os.path.basename(f) for f in sampled]
        print(
            f"  [json_sampling] {len(all_files)} files found; "
            f"sampling {max_files}: {names}",
            file=sys.stderr,
        )
        return sampled
    return all_files
