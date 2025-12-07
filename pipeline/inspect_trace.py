import json

def inspect_trace_names(filename):
    print(f"--- Inspecting: {filename} ---")
    with open(filename, 'r') as f:
        data = json.load(f)
    
    kernel_names = set()
    categories = set()
    
    for event in data['traceEvents']:
        # 只要有 duration (耗时) 的通常都是执行事件
        if 'dur' in event:
            name = event.get('name', '')
            cat = event.get('cat', '')
            
            # 记录下所有的 category 以便排查
            categories.add(cat)

            # 我们重点关注包含 'nccl', 'allreduce', 'msccl' 的名字
            # 或者 category 是 'kernel' 或 'cuda' 的
            if 'kernel' in cat.lower() or 'cuda' in cat.lower():
                kernel_names.add(name)
            elif 'nccl' in name.lower() or 'msccl' in name.lower() or 'all_reduce' in name.lower():
                # 有些通信算子可能标记在 CPU 侧，category 不是 kernel
                print(f"[Potential Match] Cat: {cat}, Name: {name}")

    print("\n--- Found GPU Kernels (Top 20 examples) ---")
    for i, name in enumerate(list(kernel_names)[:20]):
        print(name)
        
    print(f"\n--- Categories Found: {categories} ---")

# 运行侦探
inspect_trace_names('trace/kineto_trace_nccl_rank0.json')
inspect_trace_names('trace/kineto_trace_mscclpp_rank0.json')