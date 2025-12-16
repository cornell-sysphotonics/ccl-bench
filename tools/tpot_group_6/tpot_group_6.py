import json
import sys
import statistics
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt

def compute_tpots_ms(record):
    """Compute per-request TPOTs (ms) from sglang 'itls'."""
    itls = record["itls"]  # list[list[float]] in seconds
    tpots_ms = []
    for itl in itls:
        if not itl:
            continue
        tpot_ms = (sum(itl) / len(itl)) * 1000.0
        tpots_ms.append(tpot_ms)
    return tpots_ms


def plot_and_save(tpots, outfile):
    """Plot TPOT arrays and save figure."""
    plt.figure(figsize=(10, 6))

    # Unsorted subplot
    plt.subplot(2, 1, 1)
    plt.plot(tpots, marker=".", linestyle="-")
    plt.title(f"Unsorted TPOTs")
    plt.xlabel("Request index")
    plt.ylabel("TPOT (ms)")

    # Sorted subplot
    plt.subplot(2, 1, 2)
    plt.plot(sorted(tpots), marker=".", linestyle="-")
    plt.title(f"Sorted TPOTs")
    plt.xlabel("Request index")
    plt.ylabel("TPOT (ms)")

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    # print(f"Saved: {outfile}")


def metric_cal(directory: str) -> Dict[str, float]:
    """
    Extract TPOTs from a sglang benchmark JSONL file and plot them.

    Args:
        directory (str): The directory path containing the sglang benchmark JSONL file.

    Returns:
        Dict[str, float]: A dictionary containing the mean, median, standard deviation, and P99 TPOT values.
    """
    json_path = Path(directory) / "bench_results.jsonl"
    out_path = Path(directory) / "tpot.png"

    record = None
    num_records = 0

    # Load JSONL (require exactly 1 record)
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num_records += 1
            if num_records == 1:
                record = json.loads(line)
            else:
                print(f"Error: expected exactly 1 record, found {num_records}.")
                sys.exit(1)

    if record is None:
        print("Error: no record found.")
        sys.exit(1)


    # Load timings
    tpots_ms = compute_tpots_ms(record)

    # Stats check
    # if tpots_ms:
    #     mean_calc = statistics.mean(tpots_ms)
    #     std_calc = statistics.pstdev(tpots_ms)

    #     assert abs(record["mean_tpot_ms"] - mean_calc) < 1e-2, f"Expected {record['mean_tpot_ms']}, got {mean_calc}"
    #     assert abs(record["std_tpot_ms"] - std_calc) < 1e-2, f"Expected {record['std_tpot_ms']}, got {std_calc}"

    plot_and_save(tpots_ms, out_path)

    ret = {
        "mean_tpot_ms": record["mean_tpot_ms"],
        "median_tpot_ms": record["median_tpot_ms"],
        "std_tpot_ms": record["std_tpot_ms"],
        "p99_tpot_ms": record["p99_tpot_ms"],
    }

    return ret