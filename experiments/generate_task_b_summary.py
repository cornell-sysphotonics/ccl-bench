import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Constants for throughput estimation (tokens/sec derived from TPOT)
    BATCH_SIZE = 16 # From config

    # Data collected from experiments
    data = [
        {
            "experiment": "E2.1_qwen_tp4",
            "config": "TP=4, PP=1",
            "ttft_ms": 142.6,
            "tpot_ms": 46.1,
            "bubble_ratio_pct": 0.0, # N/A for PP=1
            "comm_overhead_pct": 58.42,
            "sm_efficiency_pct": 63.85,
        },
        {
            "experiment": "E2.2_qwen_tp2_pp2",
            "config": "TP=2, PP=2",
            "ttft_ms": 290.3,
            "tpot_ms": 48.6,
            "bubble_ratio_pct": 22.91,
            "comm_overhead_pct": 19.84,
            "sm_efficiency_pct": 78.01,
        },
        {
            "experiment": "E2.3_qwen_pp4",
            "config": "TP=1, PP=4",
            "ttft_ms": 227.9,
            "tpot_ms": 60.4,
            "bubble_ratio_pct": 9.47,
            "comm_overhead_pct": 40.51,
            "sm_efficiency_pct": 91.83,
        }
    ]

    # Calculate derived metrics
    for d in data:
        # System Throughput (tokens/sec) ~= Batch_Size / TPOT (s)
        # TPOT is in ms, so / 1000
        tpot_sec = d['tpot_ms'] / 1000.0
        throughput = BATCH_SIZE / tpot_sec
        d['throughput_tokens_sec'] = throughput

    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    os.makedirs("experiments", exist_ok=True)
    
    # 1. Save CSV
    csv_path = "experiments/results_summary.csv"
    # Reorder columns
    cols = ["experiment", "config", "ttft_ms", "tpot_ms", "bubble_ratio_pct", "comm_overhead_pct", "sm_efficiency_pct", "throughput_tokens_sec"]
    df = df[cols]
    df.to_csv(csv_path, index=False)
    print(f"Saved summary to {csv_path}")
    print(df.to_string())

    # 2. Generate Plots (Individual Files)
    
    # Plot 1: TTFT
    plt.figure(figsize=(8, 6))
    bars = plt.bar(df['config'], df['ttft_ms'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('TTFT (ms)')
    plt.title('Time to First Token (Lower is Better)')
    plt.bar_label(bars, fmt='%.1f')
    plt.tight_layout()
    plt.savefig("experiments/plot_ttft.png", dpi=300)
    plt.close()
    print("Saved experiments/plot_ttft.png")
    
    # Plot 2: TPOT
    plt.figure(figsize=(8, 6))
    bars = plt.bar(df['config'], df['tpot_ms'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('TPOT (ms)')
    plt.title('Time Per Output Token (Lower is Better)')
    plt.bar_label(bars, fmt='%.1f')
    plt.tight_layout()
    plt.savefig("experiments/plot_tpot.png", dpi=300)
    plt.close()
    print("Saved experiments/plot_tpot.png")
    
    # Plot 3: Bubble Ratio
    plt.figure(figsize=(8, 6))
    # Filter out E2.1 for Bubble Ratio
    df_pp = df[df['bubble_ratio_pct'] > 0]
    if not df_pp.empty:
        bars = plt.bar(df_pp['config'], df_pp['bubble_ratio_pct'], color=['#ff7f0e', '#2ca02c'])
        plt.ylabel('Bubble Ratio (%)')
        plt.title('Pipeline Bubble Ratio (Lower is Better)')
        plt.bar_label(bars, fmt='%.2f')
    else:
        plt.text(0.5, 0.5, "No Pipeline Parallelism", ha='center', va='center')
    plt.tight_layout()
    plt.savefig("experiments/plot_bubble_ratio.png", dpi=300)
    plt.close()
    print("Saved experiments/plot_bubble_ratio.png")

    # Plot 4: Comm Overhead
    plt.figure(figsize=(8, 6))
    bars = plt.bar(df['config'], df['comm_overhead_pct'], color=['#d62728', '#9467bd', '#8c564b'])
    plt.ylabel('Comm Overhead (%)')
    plt.title('Communication Overhead (Lower is Better)')
    plt.bar_label(bars, fmt='%.2f')
    plt.tight_layout()
    plt.savefig("experiments/plot_comm_overhead.png", dpi=300)
    plt.close()
    print("Saved experiments/plot_comm_overhead.png")

    # Plot 5: SM Efficiency
    plt.figure(figsize=(8, 6))
    bars = plt.bar(df['config'], df['sm_efficiency_pct'], color=['#17becf', '#bcbd22', '#7f7f7f'])
    plt.ylabel('SM Efficiency (%)')
    plt.title('SM Efficiency (Higher is Better)')
    plt.bar_label(bars, fmt='%.2f')
    plt.tight_layout()
    plt.savefig("experiments/plot_sm_efficiency.png", dpi=300)
    plt.close()
    print("Saved experiments/plot_sm_efficiency.png")

if __name__ == "__main__":
    main()
