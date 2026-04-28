"""
update_history step — ADRS agent loop, step 5.

Appends the latest run record to the in-memory history and persists the
result to the workload card YAML so the populated card can be uploaded to
the CCL-Bench platform.

Interface:
    update_history(history, record, card, card_path)
    # history is modified in-place; card YAML is rewritten.
"""

from pathlib import Path

import yaml


def update_history(
    history: list[dict],
    record: dict,
    card: dict,
    card_path: Path,
) -> None:
    """Append record to history and persist to the workload card file."""
    history.append(record)

    if "runs" not in card or card["runs"] is None:
        card["runs"] = []
    card["runs"].append(record)

    with open(card_path, "w") as f:
        yaml.dump(card, f, allow_unicode=True, sort_keys=False)
