import re
import matplotlib.pyplot as plt


def load_expert_file(path):
    """
    读取一个 expert_calls_gpuX.txt 文件，返回 dict {expert_id: count}
    """
    expert_pattern = re.compile(r"expert_(\d+):\s+(\d+)")
    data = {}

    with open(path, "r") as f:
        for line in f:
            m = expert_pattern.search(line)
            if m:
                expert_id = int(m.group(1))
                count = int(m.group(2))
                data[expert_id] = count

    return data


def sort_by_usage(expert_dict):
    """
    将 {expert_id: count} 变成按 count 从大到小排序的 list
    返回 sorted_counts（长度64）
    """
    sorted_items = sorted(expert_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_counts = [c for _, c in sorted_items]
    return sorted_counts


def plot_two_sorted(file1, file2, label1, label2, output="expert_sorted_compare"):
    d1 = load_expert_file(file1)
    d2 = load_expert_file(file2)

    s1 = sort_by_usage(d1)
    s2 = sort_by_usage(d2)

    x = range(len(s1))

    # -------- Line Chart --------
    plt.figure(figsize=(10, 6))
    plt.plot(x, s1, marker="o", label=label1)
    plt.plot(x, s2, marker="o", label=label2)

    plt.xlabel("Expert Rank (sorted by call count)")
    plt.ylabel("Call Count")
    plt.title("Expert Usage Comparison (Sorted)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output}.png")
    print(f"Saved {output}.png")
    plt.close()

    # -------- Bar Chart --------
    plt.figure(figsize=(10, 6))
    plt.bar([i - 0.2 for i in x], s1, width=0.4, label=label1)
    plt.bar([i + 0.2 for i in x], s2, width=0.4, label=label2)

    plt.xlabel("Expert Rank (sorted by call count)")
    plt.ylabel("Call Count")
    plt.title("Expert Usage Comparison (Bar Chart)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output}_bar.png")
    print(f"Saved {output}_bar.png")
    plt.close()


if __name__ == "__main__":
    # 修改这里两个路径
    file1 = "benchmark-wikitext-naive.txt"
    file2 = "benchmark-c4.txt"

    plot_two_sorted(file1, file2, "wikitext", "c4")
