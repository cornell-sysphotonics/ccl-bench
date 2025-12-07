import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import glob

# ================= 配置区域 =================
# 这里填入你的 trace 文件路径
TRACE_FILES = {
    'NCCL': 'trace/kineto_trace_nccl_rank0.json',
    'MSCCLPP': 'trace/kineto_trace_mscclpp_rank0.json'
}

# 输出目录
OUTPUT_DIR = 'images'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 设置绘图风格
sns.set_theme(style="whitegrid", context="talk")

# ================= 解析函数 =================
def load_trace_events(file_path):
    """读取 Trace JSON 文件并返回 events 列表"""
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return []
    
    print(f"Loading {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Kineto trace 格式通常在 'traceEvents' 键下，但也可能直接是 list
            if isinstance(data, dict):
                return data.get('traceEvents', [])
            elif isinstance(data, list):
                return data
            else:
                return []
    except Exception as e:
        print(f"[Error] Failed to load json: {e}")
        return []

def extract_durations(events, keywords, exclude_keywords=None):
    """
    从 events 中提取匹配关键词的事件 duration (单位: us)
    keywords: 列表，只要 name 包含其中任意一个即可
    """
    durations = []
    names = []
    
    for e in events:
        # 必须包含 'dur' (duration) 字段
        if 'dur' not in e:
            continue
            
        name = e.get('name', '')
        cat = e.get('cat', '')
        
        # 排除项
        if exclude_keywords:
            if any(ex in name for ex in exclude_keywords):
                continue

        # 匹配关键词
        # 逻辑：如果是 GPU kernel 或 关键标注
        is_match = False
        for k in keywords:
            if k in name:
                is_match = True
                break
        
        if is_match:
            durations.append(e['dur']) # trace 中的 dur 通常是 microseconds (us)
            names.append(name)
            
    return durations, names

# ================= 主逻辑 =================

def main():
    all_data = []

    # 1. 定义针对不同 Trace 的过滤规则 (基于你提供的 log)
    # NCCL: 关注 GPU 上的 ncclDevKernel 或者标注 nccl:all_reduce
    nccl_filters = ['ncclDevKernel_AllReduce', 'nccl:all_reduce']
    
    # MSCCLPP: 关注 python 层的 run_all_reduce 或者底层的 mscclpp 关键字
    # 注意：如果 MSCCLPP 是基于 CUDA kernel 实现的，可能有特定的 kernel name
    # 如果没有特定的 GPU kernel name，我们对比 Python 层的开销
    mscclpp_filters = ['run_all_reduce', 'mscclpp_hook', 'mscclpp'] 

    # 2. 读取并提取数据
    
    # --- 处理 NCCL ---
    nccl_events = load_trace_events(TRACE_FILES['NCCL'])
    nccl_durs, nccl_names = extract_durations(nccl_events, nccl_filters)
    print(f"NCCL: Found {len(nccl_durs)} events matching {nccl_filters}")
    
    for d in nccl_durs:
        all_data.append({'Method': 'NCCL', 'Duration (us)': d})

    # --- 处理 MSCCLPP ---
    mscclpp_events = load_trace_events(TRACE_FILES['MSCCLPP'])
    mscclpp_durs, mscclpp_names = extract_durations(mscclpp_events, mscclpp_filters)
    print(f"MSCCLPP: Found {len(mscclpp_durs)} events matching {mscclpp_filters}")

    # 如果 MSCCLPP 主要是 Python 调用，可能包含了很多极短的 overhead，或者极长的等待
    # 我们可以打印一些具体的 name 看看捕获了什么
    if len(mscclpp_names) > 0:
        print(f"Sample MSCCLPP events: {list(set(mscclpp_names))[:5]}")

    for d in mscclpp_durs:
        all_data.append({'Method': 'MSCCLPP', 'Duration (us)': d})

    # 3. 转换为 DataFrame
    if not all_data:
        print("No matching events found. Please check keywords.")
        return

    df = pd.DataFrame(all_data)
    
    # 去除极端的 Outliers (可选，为了绘图好看)
    # df = df[df['Duration (us)'] < df['Duration (us)'].quantile(0.99)]

    # ================= 绘图 =================
    
    # --- Plot 1: Boxplot (分布对比) ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Method', y='Duration (us)', palette={'NCCL': '#7f7f7f', 'MSCCLPP': '#d62728'}, showfliers=False)
    plt.title('Kernel/Op Duration Distribution (Lower is Better)')
    plt.ylabel('Duration (microseconds)')
    
    save_path = os.path.join(OUTPUT_DIR, 'trace_duration_boxplot.png')
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

    # --- Plot 2: Violin Plot (密度分布) ---
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='Method', y='Duration (us)', palette={'NCCL': '#7f7f7f', 'MSCCLPP': '#d62728'}, inner='quartile')
    plt.title('Kernel/Op Duration Density')
    
    save_path = os.path.join(OUTPUT_DIR, 'trace_duration_violin.png')
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

    # --- 统计输出 ---
    print("\n--- Summary Statistics (us) ---")
    print(df.groupby('Method')['Duration (us)'].describe())

if __name__ == "__main__":
    main()