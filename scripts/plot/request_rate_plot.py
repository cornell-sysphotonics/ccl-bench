#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Data
# ============================================================

batch_sizes = [4, 8]
request_rates = [0, 5, 10, 20, 50]

TOKENS_PER_REQUEST = 512  # <-- new: convert req/s -> tok/s

# Units (raw data):
# - throughput: req/s   (we'll convert to tok/s when plotting)
# - ttft, e2e, tpot: seconds
data = {
    4: {
        "throughput": [0.620, 0.671, 0.666, 0.682, 0.673],
        "ttft": [0.0826, 0.0535, 0.0767, 0.0349, 0.0351],
        "e2e": [6.2230, 5.6886, 5.7501, 5.4603, 5.5422],
        "tpot": [0.0147, 0.0122, 0.0124, 0.0120, 0.0122],
    },
    8: {
        "throughput": [1.144, 1.110, 1.149, 1.146, 1.134],
        "ttft": [0.0692, 0.0380, 0.0381, 0.0400, 0.0406],
        "e2e": [6.2158, 5.9287, 5.9765, 6.0235, 6.1106],
        "tpot": [0.0127, 0.0130, 0.0128, 0.0128, 0.0130],
    },
}

# ============================================================
# Plot helper
# ============================================================


def plot_metric(metric, ylabel, title, save_path=None):
    """
    metric: one of ['throughput', 'ttft', 'e2e', 'tpot']
    """
    x = np.arange(len(batch_sizes))
    width = 0.14

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for i, rr in enumerate(request_rates):
        values = [data[b][metric][i] for b in batch_sizes]

        # ---- convert throughput req/s -> tok/s ----
        if metric == "throughput":
            values = [v * TOKENS_PER_REQUEST for v in values]

        ax.bar(
            x + (i - (len(request_rates) - 1) / 2) * width,
            values,
            width,
            label=f"RR={rr}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Batch {b}" for b in batch_sizes])
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.legend(
        ncol=3,
        fontsize=9,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


# ============================================================
# Main
# ============================================================


def main():
    # 1. Throughput (token/s)
    plot_metric(
        metric="throughput",
        ylabel="Throughput (token/s)",
        title="Throughput vs Batch Size under Different Request Rates",
        save_path="throughput_vs_batch.png",
    )

    # 2. TTFT
    plot_metric(
        metric="ttft",
        ylabel="TTFT (s)",
        title="TTFT vs Batch Size under Different Request Rates",
        save_path="ttft_vs_batch.png",
    )

    # 3. End-to-End Latency
    plot_metric(
        metric="e2e",
        ylabel="End-to-End Latency (s)",
        title="End-to-End Latency vs Batch Size under Different Request Rates",
        save_path="e2e_vs_batch.png",
    )

    # 4. TPOT / ITL
    plot_metric(
        metric="tpot",
        ylabel="TPOT / ITL (s)",
        title="Inter-token Latency vs Batch Size under Different Request Rates",
        save_path="tpot_vs_batch.png",
    )

    print("All figures generated.")


if __name__ == "__main__":
    main()
