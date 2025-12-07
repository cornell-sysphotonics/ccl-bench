import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================

# 1. 文件路径 (对应你训练代码里生成的路径)
TRACE_FILES = {
    'NCCL': 'trace/kineto_trace_nccl_rank0.json',
    'MSCCLPP': 'trace/kineto_trace_mscclpp_rank0.json'
}

OUTPUT_DIR = 'images'
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# 2. 关键词配置 (根据你的 Trace 内容微调)
# NCCL: 通常是 GPU 上的 kernel
KEYWORDS = {
    'NCCL': ['ncclDevKernel', 'nccl:all_reduce'], 
    # MSCCLPP: 你的 hook 里调用的是 manager.run_all_reduce
    'MSCCLPP': ['run_all_reduce', 'mscclpp_kernel'] 
}

# ================= 解析逻辑 =================

def load_events(file_path):
    if not os.path.exists(file_path): 
        print(f"Warning: File not found {file_path}")
        return []
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else data.get('traceEvents', [])
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def extract_step_ops(events, keywords, method_name):
    """
    提取一个训练 Step 内的所有通信操作
    """
    # 1. 过滤出通信事件
    comm_events = []
    for e in events:
        name = e.get('name', '')
        # 只要匹配任一关键词
        if 'dur' in e and any(k in name for k in keywords):
            comm_events.append(e)
            
    # 2. 按时间排序 (非常重要，对应 DDP 的 Bucket 顺序)
    comm_events.sort(key=lambda x: x['ts'])
    
    print(f"[{method_name}] Found {len(comm_events)} communication events.")
    
    # 3. 转换为 DataFrame 格式
    extracted_data = []
    for i, e in enumerate(comm_events):
        extracted_data.append({
            'Method': method_name,
            'Bucket ID': i + 1,        # 第几个桶
            'Latency (us)': e['dur'],  # 耗时
            'Start Time (us)': e['ts']
        })
        
    return pd.DataFrame(extracted_data)

# ================= 绘图逻辑 =================

def main():
    dfs = []
    
    # --- 1. 加载并解析数据 ---
    for method, filepath in TRACE_FILES.items():
        events = load_events(filepath)
        df_method = extract_step_ops(events, KEYWORDS[method], method)
        dfs.append(df_method)
    
    if not dfs:
        print("No data found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    
    if df.empty:
        print("No communication events matched the keywords.")
        return

    # --- 2. 检查数据对齐 ---
    # 理想情况下，NCCL 和 MSCCLPP 的 Bucket 数量应该一样
    counts = df['Method'].value_counts()
    print("\n--- Event Counts per Method ---")
    print(counts)
    
    # --- Plot 1: 每个 Bucket 的延迟对比 (Bar Plot) ---
    # 这是最直观的图：展示每一个梯度包的传输耗时
    plt.figure(figsize=(12, 6))
    
    sns.barplot(
        data=df, 
        x='Bucket ID', 
        y='Latency (us)', 
        hue='Method',
        palette={'NCCL': '#7f7f7f', 'MSCCLPP': '#d62728'}
    )
    
    plt.title('Per-Bucket Communication Latency (Lower is Better)')
    plt.ylabel('Latency (microseconds)')
    plt.xlabel('DDP Bucket Sequence ID')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    save_path = os.path.join(OUTPUT_DIR, '01_training_bucket_latency.png')
    plt.savefig(save_path)
    print(f"Saved: {save_path}")

    # --- Plot 2: 累计通信时间 (Total Overhead) ---
    # 计算这一步训练中，通信总共卡了多久
    total_time = df.groupby('Method')['Latency (us)'].sum().reset_index()
    total_time['Latency (ms)'] = total_time['Latency (us)'] / 1000.0
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        data=total_time, 
        x='Method', 
        y='Latency (ms)', 
        palette={'NCCL': '#7f7f7f', 'MSCCLPP': '#d62728'}
    )
    
    plt.title('Total Communication Overhead per Step')
    plt.ylabel('Total Latency (milliseconds)')
    
    # 在柱子上标数值
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f} ms', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    save_path = os.path.join(OUTPUT_DIR, '02_total_step_overhead.png')
    plt.savefig(save_path)
    print(f"Saved: {save_path}")

    # --- Plot 3: 简单的散点分布 (Strip Plot) ---
    plt.figure(figsize=(8, 6))
    sns.stripplot(
        data=df, 
        x='Method', 
        y='Latency (us)', 
        jitter=True, 
        size=8,
        palette={'NCCL': '#7f7f7f', 'MSCCLPP': '#d62728'}
    )
    plt.title('Latency Distribution of All Packets')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    save_path = os.path.join(OUTPUT_DIR, '03_latency_distribution.png')
    plt.savefig(save_path)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()