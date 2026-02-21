import pandas as pd
import re
import matplotlib.pyplot as plt


def parse_inference_log(file_path, label):
    with open(file_path, "r") as f:
        content = f.read()
    blocks = re.split(r"python bench\.py", content)
    data = []
    for block in blocks:
        if "Throughput" not in block:
            continue
        bs = re.search(r"--batch-size\s+(\d+)", block)
        tp = re.search(r"Throughput \(completed\):\s+([\d.]+)\s+req/s", block)
        ttft = re.search(r"TTFT:\s+mean=([\d.]+)s", block)
        e2e = re.search(r"E2E:\s+mean=([\d.]+)s", block)
        tpot = re.search(r"TPOT:\s+mean=([\d.]+)s", block)

        if bs and tp and ttft and e2e and tpot:
            # 核心计算：估算每秒生成的 Token 总数
            tokens_per_req = (float(e2e.group(1)) - float(ttft.group(1))) / float(
                tpot.group(1)
            )
            token_tp = float(tp.group(1)) * tokens_per_req
            data.append(
                {
                    "Config": label,
                    "Batch Size": int(bs.group(1)),
                    "Throughput (tokens/s)": token_tp,
                }
            )
    return data


# 加载数据并绘图
res = []
res.extend(parse_inference_log("batch_default.txt", "Default"))
res.extend(parse_inference_log("batch_without_Cugraph.txt", "w/o Cugraph"))
res.extend(
    parse_inference_log("batch_without_chunked_prefill.txt", "w/o Chunked Prefill")
)
df = pd.DataFrame(res)

plt.figure(figsize=(10, 6))
df.pivot(index="Batch Size", columns="Config", values="Throughput (tokens/s)").plot(
    kind="bar", ax=plt.gca()
)
plt.title("Token Throughput Comparison (tokens/s)")
plt.ylabel("Tokens Per Second")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig("throughput_tokens_plot.png")
