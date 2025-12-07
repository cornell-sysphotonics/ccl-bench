import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import os

# ================= 配置区域 =================
TRACE_FILES = {
    'NCCL': 'trace/kineto_trace_nccl_rank0.json',
    'MSCCLPP': 'trace/kineto_trace_mscclpp_rank0.json'
}
OUTPUT_DIR = 'images'

# ================= 解析逻辑 =================
def load_events(file_path):
    if not os.path.exists(file_path): return []
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else data.get('traceEvents', [])
    except: return []

def get_timeline_data(events, keyword, label_prefix):
    """提取时间轴数据：开始时间 (ts) 和 持续时间 (dur)"""
    timeline = []
    # 找到第一个相关事件的时间作为基准零点
    start_offset = None
    
    # 按照时间排序
    sorted_events = sorted(events, key=lambda x: x.get('ts', 0))
    
    for e in sorted_events:
        name = e.get('name', '')
        if keyword in name and 'ts' in e and 'dur' in e:
            ts = e['ts']
            if start_offset is None:
                start_offset = ts
            
            # 归一化时间 (变成从 0ms 开始)
            rel_start = (ts - start_offset) / 1000.0 # 转换为 ms
            duration = e['dur'] / 1000.0             # 转换为 ms
            
            timeline.append({
                'Label': label_prefix,
                'Start': rel_start,
                'Duration': duration,
                'Name': name
            })
            
            # 为了图表清晰，只取前 50 个事件
            if len(timeline) >= 50:
                break
    return timeline

def count_memory_ops(events):
    """统计内存操作次数"""
    ops = {'Memcpy': 0, 'Memset': 0}
    for e in events:
        name = e.get('name', '').lower()
        if 'memcpy' in name: ops['Memcpy'] += 1
        if 'memset' in name: ops['Memset'] += 1
    return ops

# ================= 主程序 =================
def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # 1. Timeline 数据准备
    # NCCL 找 'nccl' 相关的 kernel/annotation
    nccl_events_raw = load_events(TRACE_FILES['NCCL'])
    nccl_timeline = get_timeline_data(nccl_events_raw, 'nccl', 'NCCL Stream')
    
    # MSCCLPP 找 'mscclpp' 或 'run_all_reduce'
    mscclpp_events_raw = load_events(TRACE_FILES['MSCCLPP'])
    # 注意：这里关键词用你 trace 里出现过的，如 'mscclpp' 或 'run_all_reduce'
    mscclpp_timeline = get_timeline_data(mscclpp_events_raw, 'mscclpp', 'MSCCLPP Stream')

    all_timeline = nccl_timeline + mscclpp_timeline
    df_timeline = pd.DataFrame(all_timeline)

    # 2. 绘制甘特图 (Gantt Chart)
    if not df_timeline.empty:
        plt.figure(figsize=(15, 6))
        
        # 使用 broken_barh 绘制
        # 这是一个技巧：Y轴是 Method，X轴是时间
        for i, method in enumerate(['NCCL Stream', 'MSCCLPP Stream']):
            subset = df_timeline[df_timeline['Label'] == method]
            if subset.empty: continue
            
            # data format: [(start, width), (start, width)...]
            bars = list(zip(subset['Start'], subset['Duration']))
            color = '#7f7f7f' if 'NCCL' in method else '#d62728'
            
            plt.broken_barh(bars, (i*10 + 5, 8), facecolors=color, edgecolor='none', label=method)

        plt.xlabel('Time (milliseconds)')
        plt.yticks([9, 19], ['NCCL', 'MSCCLPP'])
        plt.title('Execution Timeline (First 50 Events comparison)')
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        
        # 手动添加图例
        patches = [mpatches.Patch(color='#7f7f7f', label='NCCL Ops'),
                   mpatches.Patch(color='#d62728', label='MSCCLPP Ops')]
        plt.legend(handles=patches)

        save_path = os.path.join(OUTPUT_DIR, '03_execution_timeline.png')
        plt.savefig(save_path)
        print(f"Generated Timeline: {save_path}")
    else:
        print("No timeline data found. Check keywords.")

    # 3. 统计内存操作 (Memory Ops Analysis)
    nccl_mem = count_memory_ops(nccl_events_raw)
    mscclpp_mem = count_memory_ops(mscclpp_events_raw)
    
    print("\n--- Memory Operation Counts (Full Trace) ---")
    print(f"NCCL Trace:    Memcpy={nccl_mem['Memcpy']}, Memset={nccl_mem['Memset']}")
    print(f"MSCCLPP Trace: Memcpy={mscclpp_mem['Memcpy']}, Memset={mscclpp_mem['Memset']}")
    
    # 简单的柱状图对比
    mem_df = pd.DataFrame([
        {'Method': 'NCCL', 'Type': 'Memcpy', 'Count': nccl_mem['Memcpy']},
        {'Method': 'MSCCLPP', 'Type': 'Memcpy', 'Count': mscclpp_mem['Memcpy']},
        {'Method': 'NCCL', 'Type': 'Memset', 'Count': nccl_mem['Memset']},
        {'Method': 'MSCCLPP', 'Type': 'Memset', 'Count': mscclpp_mem['Memset']}
    ])
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=mem_df, x='Type', y='Count', hue='Method', palette={'NCCL': '#7f7f7f', 'MSCCLPP': '#d62728'})
    plt.title('Memory Operations Count (Lower is usually better for latency)')
    plt.ylabel('Count')
    
    save_path = os.path.join(OUTPUT_DIR, '04_memory_ops_count.png')
    plt.savefig(save_path)
    print(f"Generated Memory Ops Plot: {save_path}")

if __name__ == "__main__":
    main()