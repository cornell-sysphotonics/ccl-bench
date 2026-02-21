import pandas as pd
import re
import matplotlib.pyplot as plt


def parse_inference_log(file_path, label):
    """
    解析 vLLM 性能测试日志，提取所有核心指标
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"警告: 未找到文件 {file_path}")
        return []

    # 按 benchmark 命令分割数据块
    blocks = re.split(r"python bench\.py", content)
    data = []

    for block in blocks:
        if "Throughput" not in block:
            continue

        # 使用正则表达式匹配各项指标
        bs_match = re.search(r"--batch-size\s+(\d+)", block)
        tp_match = re.search(r"Throughput \(completed\):\s+([\d.]+)\s+req/s", block)
        ttft_match = re.search(r"TTFT:\s+mean=([\d.]+)s", block)
        e2e_match = re.search(r"E2E:\s+mean=([\d.]+)s", block)
        tpot_match = re.search(r"TPOT:\s+mean=([\d.]+)s", block)

        if bs_match and tp_match:
            data.append(
                {
                    "Config": label,
                    "Batch Size": int(bs_match.group(1)),
                    "Throughput (req/s)": float(tp_match.group(1)),
                    "TTFT (s)": float(ttft_match.group(1)) if ttft_match else None,
                    "E2E (s)": float(e2e_match.group(1)) if e2e_match else None,
                    "TPOT/ITL (s)": float(tpot_match.group(1)) if tpot_match else None,
                }
            )
    return data


# 1. 数据加载与整合
configs = [
    ("batch_default.txt", "Default"),
    ("batch_without_Cugraph.txt", "w/o Cugraph"),
    ("batch_without_chunked_prefill.txt", "w/o Chunked Prefill"),
]

all_data = []
for file, label in configs:
    all_data.extend(parse_inference_log(file, label))

df = pd.DataFrame(all_data)

# 2. 计算加速比 (Speedup) - 以 Default 为基准
# 吞吐量对比透视表
tp_pivot = df.pivot(index="Batch Size", columns="Config", values="Throughput (req/s)")
tp_pivot["Cugraph Speedup"] = tp_pivot["Default"] / tp_pivot["w/o Cugraph"]
tp_pivot["Chunked Prefill Speedup"] = (
    tp_pivot["Default"] / tp_pivot["w/o Chunked Prefill"]
)

print("\n--- 吞吐量与加速比分析 ---")
print(tp_pivot)

# 3. 绘图：吞吐量对比
plt.figure(figsize=(10, 6))
df.pivot(index="Batch Size", columns="Config", values="Throughput (req/s)").plot(
    kind="bar", ax=plt.gca()
)
plt.title("Throughput Comparison (req/s)")
plt.ylabel("Requests Per Second")
plt.xlabel("Batch Size")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend(title="Optimization")
plt.tight_layout()
plt.savefig("throughput_comparison.png")

# 4. 绘图：延迟指标 (Latency Metrics)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = [
    ("TTFT (s)", "Mean TTFT (Time to First Token)"),
    ("TPOT/ITL (s)", "Mean TPOT (Inter-token Latency)"),
    ("E2E (s)", "Mean E2E (End-to-End Latency)"),
]

for i, (col, title) in enumerate(metrics):
    df.pivot(index="Batch Size", columns="Config", values=col).plot(
        kind="bar", ax=axes[i]
    )
    axes[i].set_title(title)
    axes[i].set_ylabel("Time (seconds)")
    axes[i].grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("inference_latency_comparison.png")
print(
    "\n[系统信息] 图表已保存为 'throughput_comparison.png' 和 'inference_latency_comparison.png'"
)

# 5. 生成 LaTeX 表格代码
print("\n--- LaTeX 表格代码 ---")
latex_str = df.sort_values(["Batch Size", "Config"]).to_latex(
    index=False,
    float_format="%.3f",
    caption="Full Inference Metrics: Comparing Optimizations across Batch Sizes",
    label="tab:inference_full",
)
print(latex_str)
