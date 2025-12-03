#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
from typing import Dict, Tuple

TX_RE = re.compile(r"Link (\d+): Data Tx: (\d+) KiB")
RX_RE = re.compile(r"Link (\d+): Data Rx: (\d+) KiB")


def parse_dump(path: Path) -> Dict[int, Dict[int, Tuple[int, int]]]:
    """Return {gpu: {link: (tx_kib, rx_kib)}} for an nvlink -gt d dump."""
    data: Dict[int, Dict[int, Tuple[int, int]]] = defaultdict(dict)
    pending: Dict[int, Tuple[int, int]] = {}
    gpu = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("GPU "):
            gpu = int(line.split(":")[0].split()[1])
            data.setdefault(gpu, {})
            pending.clear()
            continue
        if gpu is None:
            continue
        tx_match = TX_RE.search(line)
        if tx_match:
            link = int(tx_match.group(1))
            tx = int(tx_match.group(2))
            pending[link] = (tx, pending.get(link, (0, 0))[1])
            continue
        rx_match = RX_RE.search(line)
        if rx_match:
            link = int(rx_match.group(1))
            rx = int(rx_match.group(2))
            tx = pending.get(link, (0, 0))[0]
            data[gpu][link] = (tx, rx)
    return data


def kib_to_gib(value: int) -> float:
    return value / (1024**2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show per-GPU/per-link NVLink deltas between two nvidia-smi nvlink -gt d dumps.",
    )
    parser.add_argument(
        "before",
        nargs="?",
        default="before.txt",
        type=Path,
        help="nvlink -gt d output captured before the workload (default: before.txt)",
    )
    parser.add_argument(
        "after",
        nargs="?",
        default="after.txt",
        type=Path,
        help="nvlink -gt d output captured after the workload (default: after.txt)",
    )
    args = parser.parse_args()

    before = parse_dump(args.before)
    after = parse_dump(args.after)

    rows = []
    for gpu, links in after.items():
        for link, (tx_after, rx_after) in links.items():
            tx_before, rx_before = before.get(gpu, {}).get(link, (tx_after, rx_after))
            rows.append((gpu, link, tx_after - tx_before, rx_after - rx_before))

    if not rows:
        print("No overlapping GPU/link entries were found.")
        return

    header = f"{'GPU':<4} {'Link':<4} {'ΔTx (KiB)':>12} {'ΔRx (KiB)':>12} {'ΔTx (GiB)':>12} {'ΔRx (GiB)':>12}"
    print(header)
    print("-" * len(header))

    totals = defaultdict(lambda: [0, 0])
    for gpu, link, delta_tx, delta_rx in sorted(rows):
        totals[gpu][0] += delta_tx
        totals[gpu][1] += delta_rx
        print(
            f"{gpu:<4} {link:<4} {delta_tx:12,} {delta_rx:12,} "
            f"{kib_to_gib(delta_tx):12.3f} {kib_to_gib(delta_rx):12.3f}"
        )

    print("\nTotals per GPU (GiB)")
    for gpu in sorted(totals):
        total_tx, total_rx = totals[gpu]
        print(f"GPU {gpu}: ΔTx={kib_to_gib(total_tx):.3f} GiB, ΔRx={kib_to_gib(total_rx):.3f} GiB")


if __name__ == "__main__":
    main()

