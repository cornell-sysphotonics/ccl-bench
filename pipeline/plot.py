import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_trace(filename, label):
    print(f"Loading {label} trace from {filename}...")
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return []

    durations = []
    
    # 遍历所有事件
    for event in data['traceEvents']:
        if 'dur' not in event:
            continue
            
        name = event.get('name', '')
        cat = event.get('cat', '')
        
        # === 针对你 inspect 结果的精准匹配 ===
        
        # 1. NCCL 匹配规则
        # 根据 inspect: Cat: user_annotation, Name: nccl:all_reduce
        if label == 'NCCL':
            if cat == 'user_annotation' and 'nccl:all_reduce' in name:
                durations.append(event['dur'])

        # 2. MSCCL++ 匹配规则
        # 根据 inspect: Cat: python_function, Name: mscclpp_manager.py(68): run_all_reduce
        # 或者 Name: train_llama_mscclpp.py(97): mscclpp_hook
        elif label == 'MSCCL++':
            # 我们统计 run_all_reduce 的耗时，这代表了 MSCCL++ 通信逻辑的 CPU 开销
            if cat == 'python_function' and 'run_all_reduce' in name:
                durations.append(event['dur'])

    print(f"  -> Found {len(durations)} events for {label}.")
    if len(durations) > 0:
        avg_dur = sum(durations) / len(durations)
        print(f"  -> Average Latency: {avg_dur:.2f} us")
    
    return durations

# 请确保文件名路径正确，如果不确定可以用绝对路径
file_nccl = 'trace/kineto_trace_nccl_rank0.json'
file_mscclpp = 'trace/kineto_trace_mscclpp_rank0.json'

# 1. 提取数据
nccl_times = analyze_trace(file_nccl, 'NCCL')
mscclpp_times = analyze_trace(file_mscclpp, 'MSCCL++')

# 2. 绘图
if len(nccl_times) > 0 and len(mscclpp_times) > 0:
    plt.figure(figsize=(10, 6))
    
    # 创建箱线图
    # showfliers=False 会隐藏异常值，让对比更清晰
    plt.boxplot([nccl_times, mscclpp_times], labels=['NCCL', 'MSCCL++'], showfliers=False)
    
    plt.ylabel('CPU Overhead / Latency (microseconds)')
    plt.title('Communication Overhead Comparison: NCCL vs MSCCL++')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 保存图片
    output_img = 'latency_comparison.png'
    plt.savefig(output_img)
    print(f"\nSuccess! Plot saved to {output_img}")
    
    # 打印简报
    print("\n--- Summary ---")
    print(f"NCCL Median: {np.median(nccl_times):.2f} us")
    print(f"MSCCL++ Median: {np.median(mscclpp_times):.2f} us")
    
else:
    print("\nError: Could not find matching events in one or both files.")
    print("Please double check the 'inspect.py' output to ensure 'nccl:all_reduce' and 'run_all_reduce' exist.")