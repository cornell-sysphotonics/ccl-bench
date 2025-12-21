#!/usr/bin/env python3
"""
Kernel Time Breakdown Visualization Script

Based on Stream ID classification (same logic as comm_time_breakdown.py),
visualize the time breakdown of different kernel types:
- DP (Data Parallel): AllGather, ReduceScatter, AllReduce for ZeRO-3
  - DP_param: Parameter gathering (AllGather)
  - DP_grad: Gradient scattering (ReduceScatter)
  - DP_sync: Synchronization (AllReduce)
- TP (Tensor Parallel): Small AllReduce for tensor parallelism
- PP (Pipeline Parallel): Send/Recv operations
- EP (Expert Parallel): AllToAll for MoE
- Compute: Non-NCCL kernels (GEMM, activation, etc.)
- Idle: GPU idle time

Usage:
    python visualize_kernel_breakdown.py <trace.sqlite> [--output <output.png>]
"""

import os
import sys
import sqlite3
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Import analysis function from comm_time_breakdown
try:
    from comm_time_breakdown import analyze_timeline_by_parallelism, detect_stream_roles
except ImportError:
    # If running from different directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from comm_time_breakdown import analyze_timeline_by_parallelism, detect_stream_roles

# ============================================================================
# Color scheme - Light pastel colors for PPT compatibility
# ============================================================================
# Light Blue / Light Grey / Light Green palette
CATEGORY_COLORS = {
    'compute': '#81C784',      # Light Green - compute kernels
    'DP_param': '#81D4FA',     # Light Blue - DP AllGather (parameter fetch)
    'DP_grad': '#4FC3F7',      # Sky Blue - DP ReduceScatter (gradient sync)
    'DP_sync': '#29B6F6',      # Light Blue - DP AllReduce (sync)
    'DP': '#4DD0E1',           # Cyan - DP (combined)
    'TP': '#B39DDB',           # Light Purple - Tensor Parallel
    'PP': '#80CBC4',           # Light Teal - Pipeline Parallel
    'EP': '#9FA8DA',           # Light Indigo - Expert Parallel
    'OTHER': '#B0BEC5',        # Light Grey - Other communication
    'idle': '#ECEFF1',         # Very Light Grey - Idle time
}

# Color scheme for high-level categories (Communication / Compute / Memory)
# Light Blue / Light Grey / Light Green
HIGHLEVEL_COLORS = {
    'Communication': '#64B5F6',  # Light Blue - NCCL, collective ops
    'Compute': '#81C784',        # Light Green - GEMM, activation, etc.
    'Memory': '#B0BEC5',         # Light Grey - memcpy, memset
    'Other': '#CFD8DC',          # Very Light Grey - other kernels
}

# Alternative distinct colors for Top 10 chart (pastel theme)
TOP10_COLORS = [
    '#64B5F6',  # Light Blue
    '#81C784',  # Light Green
    '#B0BEC5',  # Light Grey
    '#4FC3F7',  # Sky Blue
    '#A5D6A7',  # Pale Green
    '#90A4AE',  # Grey
    '#4DD0E1',  # Cyan
    '#C5E1A5',  # Lime Green
    '#B0BEC5',  # Light Grey
    '#80DEEA',  # Light Cyan
]

# ============================================================================
# Kernel Classification: Map kernel names to categories
# ============================================================================
def classify_kernel_highlevel(kernel_name):
    """
    Classify a kernel into high-level category: Communication / Compute / Memory / Other
    """
    name_lower = kernel_name.lower()
    
    # Communication kernels (NCCL)
    if 'nccl' in name_lower:
        return 'Communication'
    
    # Memory operations
    if any(x in name_lower for x in ['memcpy', 'memset', 'copy', 'cudamemcpy']):
        return 'Memory'
    
    # Compute kernels
    compute_patterns = [
        'gemm', 'gemv', 'matmul', 'dot',           # Matrix operations
        'conv', 'winograd',                         # Convolution
        'softmax', 'layernorm', 'batchnorm',        # Normalization
        'relu', 'gelu', 'silu', 'sigmoid', 'tanh',  # Activation
        'elementwise', 'vectorized',                # Element-wise ops
        'reduce', 'sum', 'mean',                    # Reduction ops
        'attention', 'flash',                       # Attention
        'adam', 'sgd', 'optimizer',                 # Optimizer
        'embedding', 'lookup',                      # Embedding
        'dropout', 'mask',                          # Regularization
        'cast', 'convert',                          # Type conversion
        'transpose', 'permute',                     # Tensor ops
        'fused', 'kernel',                          # Generic compute
    ]
    
    if any(p in name_lower for p in compute_patterns):
        return 'Compute'
    
    # Default to Compute for unknown GPU kernels
    return 'Compute'


def get_kernel_short_name(kernel_name):
    """
    Extract a readable short name from the full kernel name.
    
    Examples:
    - "ncclKernel_AllGather_RING_LL_Sum_int8_t" -> "NCCL AllGather Ring"
    - "ampere_bf16_s16816gemm_bf16_256x64_ldg8_..." -> "GEMM BF16"
    - "vectorized_elementwise_kernel" -> "Elementwise"
    """
    name_lower = kernel_name.lower()
    
    # NCCL kernels - extract operation type and algorithm
    if 'nccl' in name_lower:
        # Determine operation type
        if 'allgather' in name_lower:
            op = 'AllGather'
        elif 'reducescatter' in name_lower:
            op = 'ReduceScatter'
        elif 'allreduce' in name_lower:
            op = 'AllReduce'
        elif 'broadcast' in name_lower:
            op = 'Broadcast'
        elif 'send' in name_lower or 'recv' in name_lower:
            op = 'Send/Recv'
        else:
            op = 'NCCL Other'
        
        # Determine algorithm
        if 'ring' in name_lower:
            algo = 'Ring'
        elif 'tree' in name_lower:
            algo = 'Tree'
        elif 'collnet' in name_lower:
            algo = 'CollNet'
        else:
            algo = ''
        
        return f"NCCL {op} {algo}".strip()
    
    # GEMM kernels
    if 'gemm' in name_lower or 'gemv' in name_lower:
        # Determine precision
        if 'bf16' in name_lower:
            prec = 'BF16'
        elif 'fp16' in name_lower or 'f16' in name_lower:
            prec = 'FP16'
        elif 'fp32' in name_lower or 'f32' in name_lower:
            prec = 'FP32'
        elif 'int8' in name_lower or 'i8' in name_lower:
            prec = 'INT8'
        else:
            prec = ''
        return f"GEMM {prec}".strip()
    
    # Memory operations
    if 'memcpy' in name_lower:
        if 'htod' in name_lower or 'h2d' in name_lower:
            return 'Memcpy H2D'
        elif 'dtoh' in name_lower or 'd2h' in name_lower:
            return 'Memcpy D2H'
        elif 'dtod' in name_lower or 'd2d' in name_lower:
            return 'Memcpy D2D'
        return 'Memcpy'
    
    if 'memset' in name_lower:
        return 'Memset'
    
    # Normalization
    if 'layernorm' in name_lower:
        return 'LayerNorm'
    if 'batchnorm' in name_lower:
        return 'BatchNorm'
    if 'rmsnorm' in name_lower:
        return 'RMSNorm'
    
    # Activation
    if 'softmax' in name_lower:
        return 'Softmax'
    if 'gelu' in name_lower:
        return 'GELU'
    if 'silu' in name_lower or 'swish' in name_lower:
        return 'SiLU'
    if 'relu' in name_lower:
        return 'ReLU'
    
    # Attention
    if 'flash' in name_lower and 'attention' in name_lower:
        return 'FlashAttention'
    if 'attention' in name_lower:
        return 'Attention'
    
    # Optimizer
    if 'adam' in name_lower:
        return 'Adam Optimizer'
    
    # Element-wise
    if 'elementwise' in name_lower or 'vectorized' in name_lower:
        return 'Elementwise'
    
    # Reduction
    if 'reduce' in name_lower:
        return 'Reduce'
    
    # Embedding
    if 'embedding' in name_lower:
        return 'Embedding'
    
    # Cast/Convert
    if 'cast' in name_lower or 'convert' in name_lower:
        return 'TypeCast'
    
    # Fused kernels
    if 'fused' in name_lower:
        return 'Fused Kernel'
    
    # Transpose
    if 'transpose' in name_lower or 'permute' in name_lower:
        return 'Transpose'
    
    # Default: try to extract meaningful part
    # Remove common prefixes/suffixes
    name = kernel_name
    for prefix in ['void ', 'ampere_', 'volta_', 'turing_', 'hopper_']:
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
    
    # Truncate long names
    if len(name) > 30:
        name = name[:27] + '...'
    
    return name


def analyze_kernel_types(sqlite_file):
    """
    Analyze all kernels and group by type with high-level classification.
    
    Returns:
        dict: {
            'kernel_stats': [{name, short_name, category, total_time, count}, ...],
            'category_totals': {Communication: time, Compute: time, Memory: time},
            'total_time': total kernel time
        }
    """
    if not os.path.exists(sqlite_file):
        print(f"Error: SQLite file not found: {sqlite_file}")
        return None
    
    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()
    
    # Query all kernels with their duration
    query = """
    SELECT s.value as name, 
           SUM(k.end - k.start) as total_time,
           COUNT(*) as count
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.shortName = s.id
    GROUP BY s.value
    ORDER BY total_time DESC
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("Error: No kernels found")
        return None
    
    # Process each kernel type
    kernel_stats = []
    category_totals = defaultdict(float)
    total_time = 0
    
    for name, time_ns, count in rows:
        short_name = get_kernel_short_name(name)
        category = classify_kernel_highlevel(name)
        
        kernel_stats.append({
            'name': name,
            'short_name': short_name,
            'category': category,
            'total_time': time_ns,
            'count': count
        })
        
        category_totals[category] += time_ns
        total_time += time_ns
    
    # Merge similar short names
    merged_stats = defaultdict(lambda: {'total_time': 0, 'count': 0, 'category': None})
    for k in kernel_stats:
        key = k['short_name']
        merged_stats[key]['total_time'] += k['total_time']
        merged_stats[key]['count'] += k['count']
        merged_stats[key]['category'] = k['category']
    
    # Convert to list and sort
    merged_list = [
        {
            'short_name': name,
            'category': stats['category'],
            'total_time': stats['total_time'],
            'count': stats['count']
        }
        for name, stats in merged_stats.items()
    ]
    merged_list.sort(key=lambda x: x['total_time'], reverse=True)
    
    return {
        'kernel_stats': merged_list,
        'category_totals': dict(category_totals),
        'total_time': total_time
    }

# ============================================================================
# Timeline Analysis (uses comm_time_breakdown.py logic)
# ============================================================================
def analyze_kernel_timeline(sqlite_file):
    """
    Analyze all kernels and compute wall-clock time for each category.
    
    Uses the same logic as comm_time_breakdown.py for consistency.
    
    Returns:
        dict: {
            'category_times': {category: wall_clock_time_ns},
            'dp_detail_times': {DP_param/DP_grad/DP_sync: time_ns},
            'total_time': total_wall_clock_time_ns,
            'stream_roles': detected stream roles,
            'kernel_counts': {category: count}
        }
    """
    # Use the .nsys-rep file path for analyze_timeline_by_parallelism
    nsys_rep_file = sqlite_file.replace('.sqlite', '.nsys-rep')
    
    # Call the analysis function from comm_time_breakdown
    result = analyze_timeline_by_parallelism(nsys_rep_file)
    
    if result is None:
        print(f"Error: Failed to analyze timeline for {sqlite_file}")
        return None
    
    # Convert result format for visualization
    # comm_time_breakdown returns times in ms, convert to ns for consistency
    category_times = {
        'compute': result['compute_time_ms'] * 1e6,
        'DP': result['comm_DP_time_ms'] * 1e6,
        'TP': result['comm_TP_time_ms'] * 1e6,
        'PP': result['comm_PP_time_ms'] * 1e6,
        'EP': result['comm_EP_time_ms'] * 1e6,
        'OTHER': result['comm_OTHER_time_ms'] * 1e6,
    }
    
    dp_detail_times = {}
    if result.get('dp_detail'):
        dp = result['dp_detail']
        dp_detail_times = {
            'DP_param': dp.get('DP_param_time_ms', 0) * 1e6,
            'DP_grad': dp.get('DP_grad_time_ms', 0) * 1e6,
            'DP_sync': dp.get('DP_sync_time_ms', 0) * 1e6,
        }
    
    # Get stream roles - need to convert string keys back to int
    stream_roles = {}
    if result.get('stream_roles'):
        for stream_id_str, role in result['stream_roles'].items():
            try:
                stream_id = int(stream_id_str)
                stream_roles[stream_id] = {
                    'role': role,
                    'stats': {'total': 0}  # Placeholder
                }
            except ValueError:
                pass
    
    return {
        'category_times': category_times,
        'dp_detail_times': dp_detail_times,
        'idle_time': result['idle_time_ms'] * 1e6,
        'total_time': result['total_time_ms'] * 1e6,
        'stream_roles': stream_roles,
        'kernel_counts': result.get('category_counts', {}),
        'dp_detail_counts': {}  # Not tracked in this format
    }


# ============================================================================
# Visualization Functions
# ============================================================================
def plot_pie_chart(data, output_path, title="Kernel Time Breakdown"):
    """
    Create a pie chart showing the breakdown of kernel types.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # === Left: Main categories ===
    categories = []
    times = []
    colors = []
    
    # Add categories in order
    for cat in ['compute', 'DP', 'TP', 'PP', 'EP', 'OTHER', 'idle']:
        if cat == 'idle':
            time_val = data['idle_time']
        else:
            time_val = data['category_times'].get(cat, 0)
        
        if time_val > 0:
            categories.append(cat)
            times.append(time_val)
            colors.append(CATEGORY_COLORS.get(cat, '#CCCCCC'))
    
    total = sum(times)
    percentages = [t / total * 100 for t in times]
    
    # Create pie chart
    wedges, texts, autotexts = ax1.pie(
        percentages, 
        labels=categories,
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 2 else '',
        pctdistance=0.75,
        startangle=90,
        explode=[0.02] * len(categories)
    )
    
    # Style
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax1.set_title("Overall Kernel Time Distribution", fontsize=14, fontweight='bold')
    
    # === Right: DP Detail breakdown (if available) ===
    dp_details = data.get('dp_detail_times', {})
    if dp_details:
        dp_cats = []
        dp_times = []
        dp_colors = []
        
        for cat in ['DP_param', 'DP_grad', 'DP_sync']:
            time_val = dp_details.get(cat, 0)
            if time_val > 0:
                dp_cats.append(cat)
                dp_times.append(time_val)
                dp_colors.append(CATEGORY_COLORS.get(cat, '#CCCCCC'))
        
        if dp_times:
            dp_total = sum(dp_times)
            dp_percentages = [t / dp_total * 100 for t in dp_times]
            
            wedges2, texts2, autotexts2 = ax2.pie(
                dp_percentages,
                labels=dp_cats,
                colors=dp_colors,
                autopct='%1.1f%%',
                pctdistance=0.75,
                startangle=90,
                explode=[0.02] * len(dp_cats)
            )
            
            for autotext in autotexts2:
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
            
            ax2.set_title("DP Communication Breakdown\n(AllGather / ReduceScatter / AllReduce)", 
                         fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, "No DP detail data", ha='center', va='center', fontsize=12)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
    else:
        ax2.text(0.5, 0.5, "No DP detail data", ha='center', va='center', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    pie_path = output_path.replace('.png', '_pie.png')
    plt.savefig(pie_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved pie chart: {pie_path}")
    plt.close()
    
    return pie_path


def plot_bar_chart(data, output_path, title="Kernel Time Breakdown"):
    """
    Create a horizontal bar chart showing time breakdown.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    categories = []
    times_ms = []
    colors = []
    
    # Main categories
    for cat in ['compute', 'DP', 'TP', 'PP', 'EP', 'OTHER', 'idle']:
        if cat == 'idle':
            time_val = data['idle_time']
        else:
            time_val = data['category_times'].get(cat, 0)
        
        if time_val > 0:
            categories.append(cat)
            times_ms.append(time_val / 1e6)  # ns to ms
            colors.append(CATEGORY_COLORS.get(cat, '#CCCCCC'))
    
    total_ms = sum(times_ms)
    percentages = [t / total_ms * 100 for t in times_ms]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, times_ms, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        width = bar.get_width()
        ax.text(width + total_ms * 0.01, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}% ({times_ms[i]:.1f} ms)',
                ha='left', va='center', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(times_ms) * 1.3)
    
    # Add grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    bar_path = output_path.replace('.png', '_bar.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved bar chart: {bar_path}")
    plt.close()
    
    return bar_path


def plot_stacked_bar(data, output_path, title="Kernel Time Breakdown"):
    """
    Create a stacked bar chart showing the composition.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Prepare data - use DP detail if available
    categories = []
    times = []
    colors = []
    
    # Add compute first
    compute_time = data['category_times'].get('compute', 0)
    if compute_time > 0:
        categories.append('compute')
        times.append(compute_time)
        colors.append(CATEGORY_COLORS['compute'])
    
    # Add DP details if available, otherwise use combined DP
    dp_details = data.get('dp_detail_times', {})
    if dp_details:
        for cat in ['DP_param', 'DP_grad', 'DP_sync']:
            time_val = dp_details.get(cat, 0)
            if time_val > 0:
                categories.append(cat)
                times.append(time_val)
                colors.append(CATEGORY_COLORS.get(cat, '#CCCCCC'))
    else:
        dp_time = data['category_times'].get('DP', 0)
        if dp_time > 0:
            categories.append('DP')
            times.append(dp_time)
            colors.append(CATEGORY_COLORS['DP'])
    
    # Add other categories
    for cat in ['TP', 'PP', 'EP', 'OTHER']:
        time_val = data['category_times'].get(cat, 0)
        if time_val > 0:
            categories.append(cat)
            times.append(time_val)
            colors.append(CATEGORY_COLORS.get(cat, '#CCCCCC'))
    
    # Add idle
    idle_time = data['idle_time']
    if idle_time > 0:
        categories.append('idle')
        times.append(idle_time)
        colors.append(CATEGORY_COLORS['idle'])
    
    total = sum(times)
    percentages = [t / total * 100 for t in times]
    
    # Create stacked horizontal bar
    left = 0
    for i, (cat, time_val, color, pct) in enumerate(zip(categories, times, colors, percentages)):
        width = time_val / total
        ax.barh(0, width, left=left, color=color, edgecolor='white', linewidth=1, height=0.6)
        
        # Add label if segment is wide enough
        if pct > 3:
            ax.text(left + width/2, 0, f'{cat}\n{pct:.1f}%',
                   ha='center', va='center', fontsize=9, fontweight='bold')
        
        left += width
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Time Proportion', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend
    legend_patches = [mpatches.Patch(color=colors[i], label=f'{categories[i]} ({percentages[i]:.1f}%)')
                     for i in range(len(categories))]
    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.15),
             ncol=min(5, len(categories)), fontsize=10)
    
    plt.tight_layout()
    
    stacked_path = output_path.replace('.png', '_stacked.png')
    plt.savefig(stacked_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved stacked bar: {stacked_path}")
    plt.close()
    
    return stacked_path


def get_stream_stats_from_sqlite(sqlite_file):
    """
    Get stream statistics directly from SQLite for visualization.
    Uses the same logic as comm_time_breakdown.detect_stream_roles.
    """
    if not os.path.exists(sqlite_file):
        return {}
    
    try:
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        
        # Get stream roles using the imported function
        stream_roles = detect_stream_roles(cursor)
        conn.close()
        
        return stream_roles
    except Exception as e:
        print(f"  Error getting stream stats: {e}")
        return {}


def plot_stream_breakdown(data, output_path, title="Stream Role Breakdown", sqlite_file=None):
    """
    Create a visualization showing the role of each stream.
    """
    # Try to get stream roles from data first
    stream_roles = data.get('stream_roles', {})
    
    # If stream_roles doesn't have stats, try to get from SQLite
    if stream_roles and sqlite_file:
        full_stream_roles = get_stream_stats_from_sqlite(sqlite_file)
        if full_stream_roles:
            stream_roles = full_stream_roles
    
    if not stream_roles:
        print("  No stream role data available")
        return None
    
    fig, ax = plt.subplots(figsize=(12, max(4, len(stream_roles) * 0.5)))
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('white')
    
    # Prepare data
    streams = []
    roles = []
    kernel_counts = []
    colors = []
    
    for stream_id in sorted(stream_roles.keys()):
        info = stream_roles[stream_id]
        role = info['role'] if isinstance(info, dict) else info
        stats = info.get('stats', {}) if isinstance(info, dict) else {}
        total = stats.get('total', 0)
        
        streams.append(f"Stream {stream_id}")
        roles.append(role)
        kernel_counts.append(total)
        colors.append(CATEGORY_COLORS.get(role, '#B0BEC5'))
    
    # Filter out streams with 0 kernels
    filtered_data = [(s, r, c, col) for s, r, c, col in zip(streams, roles, kernel_counts, colors) if c > 0]
    if not filtered_data:
        print("  No streams with NCCL kernels found")
        return None
    
    streams, roles, kernel_counts, colors = zip(*filtered_data)
    
    # Create horizontal bar
    y_pos = np.arange(len(streams))
    bars = ax.barh(y_pos, kernel_counts, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add role labels
    max_count = max(kernel_counts) if kernel_counts else 1
    for i, (bar, role, count) in enumerate(zip(bars, roles, kernel_counts)):
        ax.text(bar.get_width() + max_count * 0.02, bar.get_y() + bar.get_height()/2,
                f'{role} ({count} kernels)',
                ha='left', va='center', fontsize=10, color='#37474F')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(streams, fontsize=11, color='#37474F')
    ax.set_xlabel('Number of NCCL Kernels', fontsize=12, color='#455A64', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', color='#37474F')
    ax.set_xlim(0, max_count * 1.5)
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#CFD8DC')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CFD8DC')
    ax.spines['bottom'].set_color('#CFD8DC')
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color=CATEGORY_COLORS['DP_param'], label='DP_param (AllGather)'),
        mpatches.Patch(color=CATEGORY_COLORS['DP_grad'], label='DP_grad (ReduceScatter)'),
        mpatches.Patch(color=CATEGORY_COLORS['DP_sync'], label='DP_sync (AllReduce)'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    stream_path = output_path.replace('.png', '_streams.png')
    plt.savefig(stream_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved stream breakdown: {stream_path}")
    plt.close()
    
    return stream_path


def plot_top10_kernels(data, output_path, title="Top 10 Kernel Time Breakdown"):
    """
    Create a horizontal bar chart showing Top 10 most time-consuming kernel types.
    Uses blue theme with different shades for categories.
    """
    kernel_stats = data['kernel_stats'][:10]  # Top 10
    total_time = data['total_time']
    
    if not kernel_stats:
        print("  No kernel data for Top 10 chart")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set white background with light blue accent
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('white')
    
    # Prepare data
    names = [k['short_name'] for k in kernel_stats]
    times_ms = [k['total_time'] / 1e6 for k in kernel_stats]
    categories = [k['category'] for k in kernel_stats]
    counts = [k['count'] for k in kernel_stats]
    percentages = [k['total_time'] / total_time * 100 for k in kernel_stats]
    
    # Light pastel colors: Light Blue / Light Green / Light Grey
    colors = []
    edge_colors = []
    hatches = []
    for cat in categories:
        if cat == 'Communication':
            colors.append('#64B5F6')      # Light Blue
            edge_colors.append('#42A5F5')
            hatches.append('')
        elif cat == 'Compute':
            colors.append('#81C784')      # Light Green
            edge_colors.append('#66BB6A')
            hatches.append('')
        elif cat == 'Memory':
            colors.append('#B0BEC5')      # Light Grey
            edge_colors.append('#90A4AE')
            hatches.append('')
        else:
            colors.append('#CFD8DC')      # Very Light Grey
            edge_colors.append('#B0BEC5')
            hatches.append('')
    
    # Reverse for horizontal bar (top item at top)
    names = names[::-1]
    times_ms = times_ms[::-1]
    categories = categories[::-1]
    counts = counts[::-1]
    colors = colors[::-1]
    edge_colors = edge_colors[::-1]
    percentages = percentages[::-1]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, times_ms, color=colors, edgecolor=edge_colors, 
                   linewidth=1.5, height=0.7)
    
    # Add rounded corners effect with shadow
    for bar in bars:
        bar.set_alpha(0.9)
    
    # Add labels with better formatting
    max_time = max(times_ms)
    for i, (bar, pct, count, cat) in enumerate(zip(bars, percentages, counts, categories)):
        width = bar.get_width()
        # Category tag with color coding
        cat_short = {'Communication': 'Comm', 'Compute': 'Comp', 'Memory': 'Mem', 'Other': 'Other'}
        cat_label = cat_short.get(cat, cat)
        label = f'{pct:.1f}%  •  {count:,} calls  •  {cat_label}'
        ax.text(width + max_time * 0.015, bar.get_y() + bar.get_height()/2,
                label, ha='left', va='center', fontsize=10, color='#37474F')
    
    # Style
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11, fontweight='bold', color='#37474F')
    ax.set_xlabel('Total Time (ms)', fontsize=12, color='#455A64', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', color='#37474F', pad=15)
    ax.set_xlim(0, max_time * 1.45)
    
    # Add grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#CFD8DC')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CFD8DC')
    ax.spines['bottom'].set_color('#CFD8DC')
    
    # Add legend with pastel theme
    legend_patches = [
        mpatches.Patch(color='#64B5F6', label='Communication (NCCL)', edgecolor='#42A5F5', linewidth=1.5),
        mpatches.Patch(color='#81C784', label='Compute (GEMM, etc.)', edgecolor='#66BB6A', linewidth=1.5),
        mpatches.Patch(color='#B0BEC5', label='Memory (Memcpy, etc.)', edgecolor='#90A4AE', linewidth=1.5),
    ]
    legend = ax.legend(handles=legend_patches, loc='lower right', fontsize=10,
                       framealpha=0.95, edgecolor='#CFD8DC')
    legend.get_frame().set_facecolor('white')
    
    plt.tight_layout()
    
    top10_path = output_path.replace('.png', '_top10.png')
    plt.savefig(top10_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved Top 10 chart: {top10_path}")
    plt.close()
    
    return top10_path


def plot_category_breakdown(data, output_path, title="Category Time Breakdown"):
    """
    Create a pie chart showing Communication / Compute / Memory breakdown.
    """
    category_totals = data['category_totals']
    total_time = data['total_time']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data
    categories = []
    times = []
    colors = []
    
    for cat in ['Communication', 'Compute', 'Memory', 'Other']:
        time_val = category_totals.get(cat, 0)
        if time_val > 0:
            categories.append(cat)
            times.append(time_val)
            colors.append(HIGHLEVEL_COLORS.get(cat, '#CCCCCC'))
    
    percentages = [t / total_time * 100 for t in times]
    
    # Create pie chart with labels
    def make_autopct(values):
        def autopct(pct):
            total = sum(values)
            val = pct * total / 100.0
            return f'{pct:.1f}%\n({val/1e6:.1f} ms)'
        return autopct
    
    wedges, texts, autotexts = ax.pie(
        times,
        labels=categories,
        colors=colors,
        autopct=make_autopct(times),
        pctdistance=0.75,
        startangle=90,
        explode=[0.03] * len(categories)
    )
    
    # Style
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_fontsize(10)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    cat_path = output_path.replace('.png', '_category.png')
    plt.savefig(cat_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved category chart: {cat_path}")
    plt.close()
    
    return cat_path


def plot_combined_top10(data, output_path, title="Kernel Time Analysis"):
    """
    Create a combined visualization with:
    - Left: Category breakdown pie chart
    - Right: Top 10 kernels bar chart
    Blue theme for PPT compatibility.
    """
    kernel_stats = data['kernel_stats'][:10]
    category_totals = data['category_totals']
    total_time = data['total_time']
    
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor('white')
    
    # === Left subplot: Category pie chart ===
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_facecolor('#F8FAFC')
    
    categories = []
    times = []
    colors = []
    
    # Light pastel colors for categories
    cat_colors_pastel = {
        'Communication': '#64B5F6',  # Light Blue
        'Compute': '#81C784',        # Light Green
        'Memory': '#B0BEC5',         # Light Grey
        'Other': '#CFD8DC',          # Very Light Grey
    }
    
    for cat in ['Communication', 'Compute', 'Memory', 'Other']:
        time_val = category_totals.get(cat, 0)
        if time_val > 0:
            categories.append(cat)
            times.append(time_val)
            colors.append(cat_colors_pastel.get(cat, '#CFD8DC'))
    
    def make_autopct(values):
        def autopct(pct):
            total = sum(values)
            val = pct * total / 100.0
            return f'{pct:.1f}%\n({val/1e6:.1f} ms)'
        return autopct
    
    wedges, texts, autotexts = ax1.pie(
        times,
        labels=categories,
        colors=colors,
        autopct=make_autopct(times),
        pctdistance=0.7,
        startangle=90,
        explode=[0.03] * len(categories),
        wedgeprops=dict(linewidth=2, edgecolor='white')
    )
    
    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('bold')
        text.set_color('#37474F')
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color('#37474F')
        autotext.set_fontweight('bold')
    
    ax1.set_title("Category Breakdown\n(Communication / Compute / Memory)", 
                  fontsize=12, fontweight='bold', color='#37474F')
    
    # === Right subplot: Top 10 bar chart ===
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor('#F8FAFC')
    
    names = [k['short_name'] for k in kernel_stats]
    times_ms = [k['total_time'] / 1e6 for k in kernel_stats]
    cats = [k['category'] for k in kernel_stats]
    counts = [k['count'] for k in kernel_stats]
    bar_colors = [cat_colors_pastel.get(cat, '#CFD8DC') for cat in cats]
    percentages = [k['total_time'] / total_time * 100 for k in kernel_stats]
    
    # Reverse for top-to-bottom display
    names = names[::-1]
    times_ms = times_ms[::-1]
    cats = cats[::-1]
    counts = counts[::-1]
    bar_colors = bar_colors[::-1]
    percentages = percentages[::-1]
    
    y_pos = np.arange(len(names))
    bars = ax2.barh(y_pos, times_ms, color=bar_colors, edgecolor='white', linewidth=1.5, height=0.7)
    
    max_time = max(times_ms)
    for i, (bar, pct, count) in enumerate(zip(bars, percentages, counts)):
        width = bar.get_width()
        label = f'{pct:.1f}% ({count:,})'
        ax2.text(width + max_time * 0.02, bar.get_y() + bar.get_height()/2,
                label, ha='left', va='center', fontsize=9, color='#37474F')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=10, fontweight='bold', color='#37474F')
    ax2.set_xlabel('Time (ms)', fontsize=11, color='#455A64', fontweight='bold')
    ax2.set_title("Top 10 Time-Consuming Kernels", fontsize=12, fontweight='bold', color='#37474F')
    ax2.set_xlim(0, max_time * 1.4)
    ax2.xaxis.grid(True, linestyle='--', alpha=0.5, color='#CFD8DC')
    ax2.set_axisbelow(True)
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#CFD8DC')
    ax2.spines['bottom'].set_color('#CFD8DC')
    
    # Add legend with pastel theme
    legend_patches = [
        mpatches.Patch(color='#64B5F6', label='Communication'),
        mpatches.Patch(color='#81C784', label='Compute'),
        mpatches.Patch(color='#B0BEC5', label='Memory'),
    ]
    legend = ax2.legend(handles=legend_patches, loc='lower right', fontsize=9,
                       framealpha=0.95, edgecolor='#CFD8DC')
    legend.get_frame().set_facecolor('white')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02, color='#37474F')
    plt.tight_layout()
    
    combined_path = output_path.replace('.png', '_combined.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved combined chart: {combined_path}")
    plt.close()
    
    return combined_path


def print_top10_summary(data):
    """
    Print Top 10 kernel summary.
    """
    print("\n" + "=" * 80)
    print("TOP 10 KERNEL TIME BREAKDOWN")
    print("=" * 80)
    
    kernel_stats = data['kernel_stats'][:10]
    total_time = data['total_time']
    
    print(f"\n{'Rank':<5} {'Kernel Type':<25} {'Category':<15} {'Time (ms)':<12} {'%':<8} {'Calls':<10}")
    print("-" * 80)
    
    for i, k in enumerate(kernel_stats, 1):
        time_ms = k['total_time'] / 1e6
        pct = k['total_time'] / total_time * 100
        print(f"{i:<5} {k['short_name']:<25} {k['category']:<15} {time_ms:<12.2f} {pct:<8.1f} {k['count']:<10,}")
    
    # Category summary
    print("\n" + "-" * 80)
    print("CATEGORY SUMMARY:")
    print("-" * 80)
    
    category_totals = data['category_totals']
    for cat in ['Communication', 'Compute', 'Memory', 'Other']:
        time_ns = category_totals.get(cat, 0)
        if time_ns > 0:
            time_ms = time_ns / 1e6
            pct = time_ns / total_time * 100
            print(f"  {cat:<15}: {time_ms:>10.2f} ms ({pct:>5.1f}%)")
    
    print("=" * 80)


def print_summary(data):
    """
    Print a text summary of the analysis.
    """
    print("\n" + "=" * 60)
    print("KERNEL TIME BREAKDOWN SUMMARY")
    print("=" * 60)
    
    total_ns = data['total_time']
    total_ms = total_ns / 1e6
    
    print(f"\nTotal wall-clock time: {total_ms:.2f} ms")
    print(f"\n{'Category':<15} {'Time (ms)':<12} {'Percentage':<10} {'Kernels':<10}")
    print("-" * 50)
    
    # Print main categories
    for cat in ['compute', 'DP', 'TP', 'PP', 'EP', 'OTHER', 'idle']:
        if cat == 'idle':
            time_ns = data['idle_time']
        else:
            time_ns = data['category_times'].get(cat, 0)
        
        if time_ns > 0:
            time_ms = time_ns / 1e6
            pct = time_ns / total_ns * 100
            count = data['kernel_counts'].get(cat, 0) if cat != 'idle' else '-'
            print(f"{cat:<15} {time_ms:<12.2f} {pct:<10.1f}% {count}")
    
    # Print DP detail if available
    dp_details = data.get('dp_detail_times', {})
    if dp_details:
        print(f"\n{'DP Detail':<15} {'Time (ms)':<12} {'Percentage':<10} {'Kernels':<10}")
        print("-" * 50)
        dp_total = sum(dp_details.values())
        for cat in ['DP_param', 'DP_grad', 'DP_sync']:
            time_ns = dp_details.get(cat, 0)
            if time_ns > 0:
                time_ms = time_ns / 1e6
                pct = time_ns / dp_total * 100
                count = data['dp_detail_counts'].get(cat, 0)
                print(f"  {cat:<13} {time_ms:<12.2f} {pct:<10.1f}% {count}")
    
    print("\n" + "=" * 60)


# ============================================================================
# Multi-file Analysis Support
# ============================================================================
def collect_sqlite_files(path):
    """
    Collect SQLite files from a path (file or directory).
    """
    sqlite_files = []
    
    if os.path.isfile(path):
        # Single file
        if path.endswith('.sqlite'):
            sqlite_files.append(path)
        elif path.endswith('.nsys-rep'):
            sqlite_file = path.replace('.nsys-rep', '.sqlite')
            if os.path.exists(sqlite_file):
                sqlite_files.append(sqlite_file)
    elif os.path.isdir(path):
        # Directory - find all sqlite files
        for f in os.listdir(path):
            if f.endswith('.sqlite'):
                sqlite_files.append(os.path.join(path, f))
    
    return sorted(sqlite_files)


def merge_kernel_data(data_list):
    """
    Merge kernel analysis data from multiple traces.
    Aggregates time and counts across all ranks.
    """
    merged_stats = defaultdict(lambda: {'total_time': 0, 'count': 0, 'category': None})
    category_totals = defaultdict(float)
    total_time = 0
    
    for data in data_list:
        if data is None:
            continue
        
        for k in data['kernel_stats']:
            key = k['short_name']
            merged_stats[key]['total_time'] += k['total_time']
            merged_stats[key]['count'] += k['count']
            merged_stats[key]['category'] = k['category']
        
        for cat, time_val in data['category_totals'].items():
            category_totals[cat] += time_val
        
        total_time += data['total_time']
    
    # Convert to list and sort
    merged_list = [
        {
            'short_name': name,
            'category': stats['category'],
            'total_time': stats['total_time'],
            'count': stats['count']
        }
        for name, stats in merged_stats.items()
    ]
    merged_list.sort(key=lambda x: x['total_time'], reverse=True)
    
    return {
        'kernel_stats': merged_list,
        'category_totals': dict(category_totals),
        'total_time': total_time
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Visualize kernel time breakdown from Nsys trace(s)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single trace file
  python visualize_kernel_breakdown.py trace.sqlite -o output.png

  # Multiple trace files (2-node 8-GPU)
  python visualize_kernel_breakdown.py nsys_2node/ -o output.png

  # With custom title
  python visualize_kernel_breakdown.py trace.sqlite -o output.png -t "LLaMA 8B Training"
        """
    )
    parser.add_argument('trace', help='Path to .sqlite file or directory containing multiple traces')
    parser.add_argument('--output', '-o', default='kernel_breakdown.png',
                       help='Output file path (default: kernel_breakdown.png)')
    parser.add_argument('--title', '-t', default=None,
                       help='Chart title')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of top kernels to show (default: 10)')
    
    args = parser.parse_args()
    
    # Collect SQLite files
    sqlite_files = collect_sqlite_files(args.trace)
    
    if not sqlite_files:
        print(f"Error: No SQLite files found in: {args.trace}")
        sys.exit(1)
    
    print(f"\n{'=' * 60}")
    print(f"Found {len(sqlite_files)} trace file(s):")
    for f in sqlite_files:
        print(f"  - {os.path.basename(f)}")
    print(f"{'=' * 60}")
    
    # Generate title
    if args.title:
        title = args.title
    elif len(sqlite_files) == 1:
        title = os.path.basename(sqlite_files[0]).replace('.sqlite', '')
    else:
        # Use parent directory name for multi-file
        title = os.path.basename(os.path.dirname(sqlite_files[0])) + f" ({len(sqlite_files)} ranks)"
    
    # ===== Single file analysis =====
    if len(sqlite_files) == 1:
        sqlite_file = sqlite_files[0]
        print(f"\nAnalyzing single trace: {sqlite_file}")
        
        # Stream-based timeline analysis
        print("\n--- Stream-based Timeline Analysis ---")
        timeline_data = analyze_kernel_timeline(sqlite_file)
        if timeline_data:
            print_summary(timeline_data)
            print("\nGenerating stream-based visualizations...")
            plot_pie_chart(timeline_data, args.output, f"Parallelism Breakdown: {title}")
            plot_stacked_bar(timeline_data, args.output, f"Time Composition: {title}")
            plot_stream_breakdown(timeline_data, args.output, f"Stream Roles: {title}", sqlite_file=sqlite_file)
        
        # Kernel type breakdown
        print("\n--- Kernel Type Analysis ---")
        kernel_data = analyze_kernel_types(sqlite_file)
        if kernel_data:
            print_top10_summary(kernel_data)
            print("\nGenerating kernel type visualizations...")
            plot_top10_kernels(kernel_data, args.output, f"Top {args.top} Kernels: {title}")
            plot_category_breakdown(kernel_data, args.output, f"Category Breakdown: {title}")
            plot_combined_top10(kernel_data, args.output, f"Kernel Analysis: {title}")
    
    # ===== Multi-file merged analysis =====
    else:
        print(f"\nMerging analysis from {len(sqlite_files)} traces...")
        
        # Collect kernel data from all files
        all_kernel_data = []
        for sqlite_file in sqlite_files:
            print(f"  Analyzing: {os.path.basename(sqlite_file)}")
            data = analyze_kernel_types(sqlite_file)
            if data:
                all_kernel_data.append(data)
        
        # Merge results
        if all_kernel_data:
            merged_data = merge_kernel_data(all_kernel_data)
            
            print("\n--- Merged Kernel Type Analysis ---")
            print_top10_summary(merged_data)
            
            print("\nGenerating merged visualizations...")
            plot_top10_kernels(merged_data, args.output, f"Top {args.top} Kernels: {title}")
            plot_category_breakdown(merged_data, args.output, f"Category Breakdown: {title}")
            plot_combined_top10(merged_data, args.output, f"Kernel Analysis: {title}")
        
        # Also analyze first file for stream roles (usually representative)
        print("\n--- Stream Role Analysis (from first rank) ---")
        timeline_data = analyze_kernel_timeline(sqlite_files[0])
        if timeline_data:
            plot_stream_breakdown(timeline_data, args.output, f"Stream Roles: {title}", sqlite_file=sqlite_files[0])
    
    # Print summary
    print("\n" + "=" * 60)
    print("DONE! Generated charts:")
    print("=" * 60)
    base_path = args.output.replace('.png', '')
    print(f"  - {base_path}_top10.png     (Top 10 kernels by time)")
    print(f"  - {base_path}_category.png  (Comm/Compute/Memory pie)")
    print(f"  - {base_path}_combined.png  (Combined analysis)")
    print("=" * 60)


if __name__ == '__main__':
    main()

