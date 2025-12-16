import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from textwrap import dedent


def plot_and_save(ttfts, outfile):
    """Plot TTFT arrays and save figure."""
    plt.figure(figsize=(10, 6))

    # Unsorted subplot
    plt.subplot(2, 1, 1)
    plt.plot(ttfts, marker=".", linestyle="-")
    plt.title(f"Unsorted TTFTs")
    plt.xlabel("Request index")
    plt.ylabel("TTFT (ms)")

    # Sorted subplot
    plt.subplot(2, 1, 2)
    plt.plot(sorted(ttfts), marker=".", linestyle="-")
    plt.title(f"Sorted TTFTs")
    plt.xlabel("Request index")
    plt.ylabel("TTFT (ms)")

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")


def metric_cal(directory: str) -> str:
    """
    Extract TTFTs from a sglang benchmark JSONL file and plot them.

    Args:
        directory (str): The directory path containing the sglang benchmark JSONL file.

    Returns:
        str: A pretty-printed string with the statistics of TTFTs.
    """
    json_path = Path(directory) / "bench_results.jsonl"
    out_path = Path(directory) / "ttft.png"

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


    ttfts_ms = [t * 1000.0 for t in record["ttfts"]]
    plot_and_save(ttfts_ms, out_path)

    ret = dedent(f"""
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          {record['mean_ttft_ms']:.2f}
    Median TTFT (ms):                        {record['median_ttft_ms']:.2f}
    Std TTFT (ms):                           {record['std_ttft_ms']:.2f}
    P99 TTFT (ms):                           {record['p99_ttft_ms']:.2f}
    """)

    return ret