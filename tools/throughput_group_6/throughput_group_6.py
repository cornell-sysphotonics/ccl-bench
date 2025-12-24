import json
from pathlib import Path
from typing import Dict


def _load_single_record(json_path: Path) -> dict:
    """Load exactly one JSON object from a JSONL file."""
    record = None
    num_records = 0

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num_records += 1
            if num_records == 1:
                record = json.loads(line)
            else:
                raise ValueError(f"Error: expected exactly 1 record, found {num_records}: {json_path}")

    if record is None:
        raise ValueError(f"Error: no record found in {json_path}")

    return record


def metric_cal(directory: str) -> float:
    """Extract throughput metrics from an SGLang benchmark JSONL file.

    Args:
        directory (str): The directory path containing the sglang benchmark JSONL file.

    Returns:
        float: The total tokens processed per second.
    """
    json_path = Path(directory) / "bench_results.jsonl"
    if not json_path.exists():
        raise FileNotFoundError(f"bench_results.jsonl not found at: {json_path}")

    record = _load_single_record(json_path)

    duration_s = float(record["duration"])          # seconds

    total_input_tokens = int(record.get("total_input_tokens", 0))
    total_output_tokens = int(record.get("total_output_tokens", 0))
    total_tokens = total_input_tokens + total_output_tokens

    if duration_s <= 0:
        raise ValueError(f"Invalid duration (<=0) in {json_path}: {duration_s}")

    # Throughput
    total_tokens_per_sec = total_tokens / duration_s

    return total_tokens_per_sec
