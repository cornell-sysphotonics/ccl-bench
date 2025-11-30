import json
import sys
import statistics
import matplotlib.pyplot as plt
from pathlib import Path


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


def plot_and_save(ttfts, tpots, outfile, title_prefix):
    """Plot TTFT/TPOT arrays and save figure."""
    plt.figure(figsize=(10, 6))

    # TTFT subplot
    plt.subplot(2, 1, 1)
    plt.plot(ttfts, marker=".", linestyle="-")
    plt.title(f"{title_prefix} TTFTs")
    plt.xlabel("Request index")
    plt.ylabel("TTFT (ms)")

    # TPOT subplot
    plt.subplot(2, 1, 2)
    plt.plot(tpots, marker=".", linestyle="-")
    plt.title(f"{title_prefix} TPOTs")
    plt.xlabel("Request index")
    plt.ylabel("TPOT (ms)")

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <sglang_benchmark.jsonl>")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    out_unsorted = json_path.with_suffix("").as_posix() + "_unsorted.png"
    out_sorted = json_path.with_suffix("").as_posix() + "_sorted.png"

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
    ttfts_ms = [t * 1000.0 for t in record["ttfts"]]
    tpots_ms = compute_tpots_ms(record)

    # Stats check
    if tpots_ms:
        mean_calc = statistics.mean(tpots_ms)
        std_calc = statistics.pstdev(tpots_ms)

        print("\nTPOT stats check (ms):")
        print(f"  file mean = {record['mean_tpot_ms']:.6f}")
        print(f"  calc mean = {mean_calc:.6f}")
        print(f"  file std  = {record['std_tpot_ms']:.6f}")
        print(f"  calc std  = {std_calc:.6f}")

    # Create unsorted figure
    plot_and_save(
        ttfts_ms,
        tpots_ms,
        out_unsorted,
        "Unsorted"
    )

    # Create sorted figure
    ttfts_sorted = sorted(ttfts_ms)
    tpots_sorted = sorted(tpots_ms)

    plot_and_save(
        ttfts_sorted,
        tpots_sorted,
        out_sorted,
        "Sorted"
    )


if __name__ == "__main__":
    main()
