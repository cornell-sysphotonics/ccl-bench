#!/usr/bin/env python3
"""
Compare CUDA Kernel Summary CSVs from Nsight Systems for EPLB Analysis

Usage:
    python3 analyzeEPLB.py EPLBOFF.csv EPLBON.csv

CSV format expected (exported from Nsight Systems CUDA GPU Kernel Summary):
    Time,Total Time,Instances,Avg,Med,Min,Max,StdDev,Name
    48.3%,102.664 s,267576,383.682 μs,346.878 μs,2.592 μs,239.809 ms,2.472 ms,fused_moe_kernel
"""

import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path


def parse_time_value(val: str) -> float:
    if pd.isna(val) or val == '' or val == '-':
        return 0.0
    
    val = str(val).strip()

    if '%' in val:
        return float(val.replace('%', ''))
    
    match = re.match(r'([\d.]+)\s*([a-zμµ]+)?', val, re.IGNORECASE)
    if not match:
        try:
            return float(val)
        except:
            return 0.0
    
    number = float(match.group(1))
    unit = match.group(2).lower() if match.group(2) else ''
    
    unit = unit.replace('µ', 'μ')
    
    multipliers = {
        's': 1e6,
        'ms': 1e3,
        'μs': 1,
        'us': 1,
        'ns': 1e-3,
        '': 1
    }
    
    return number * multipliers.get(unit, 1)


def parse_csv(filepath: str) -> pd.DataFrame:
    for encoding in ['utf-8', 'utf-16', 'latin-1']:
        for sep in [',', '\t', ';']:
            try:
                df = pd.read_csv(filepath, encoding=encoding, sep=sep)
                if len(df.columns) >= 5:
                    break
            except:
                continue
        else:
            continue
        break
    
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    column_map = {
        'Time (%)': 'Time',
        'Time%': 'Time', 
        'Total Time (ns)': 'Total Time',
        'Avg (ns)': 'Avg',
        'Min (ns)': 'Min',
        'Max (ns)': 'Max',
        'Std Dev': 'StdDev',
        'Standard Deviation': 'StdDev',
        'Kernel Name': 'Name',
        'Function Name': 'Name',
    }
    df.rename(columns=column_map, inplace=True)
    
    print(f"Loaded {filepath}: {len(df)} kernels")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def extract_metrics(df: pd.DataFrame, label: str = "") -> dict:
    results = {'label': label}
    
    time_cols = ['Total Time', 'Avg', 'Med', 'Min', 'Max', 'StdDev']
    for col in time_cols:
        if col in df.columns:
            df[f'{col}_us'] = df[col].apply(parse_time_value)
    
    if 'Time' in df.columns:
        df['Time_pct'] = df['Time'].apply(lambda x: parse_time_value(str(x).replace('%', '')) if '%' in str(x) else float(x) if pd.notna(x) else 0)
    
    moe_mask = df['Name'].str.lower().str.contains('moe|expert', na=False)
    moe_df = df[moe_mask]
    
    if len(moe_df) > 0:
        results['moe_kernels'] = moe_df['Name'].tolist()
        results['moe_total_time_ms'] = moe_df['Total Time_us'].sum() / 1000
        results['moe_time_pct'] = moe_df['Time_pct'].sum() if 'Time_pct' in moe_df.columns else 0
        results['moe_instances'] = moe_df['Instances'].sum() if 'Instances' in moe_df.columns else 0
        
        fused_moe = moe_df[moe_df['Name'].str.contains('fused_moe', case=False, na=False)]
        if len(fused_moe) > 0:
            row = fused_moe.iloc[0]
            results['fused_moe_total_ms'] = row['Total Time_us'] / 1000
            results['fused_moe_avg_us'] = row['Avg_us']
            results['fused_moe_min_us'] = row['Min_us']
            results['fused_moe_max_us'] = row['Max_us']
            results['fused_moe_std_us'] = row['StdDev_us'] if 'StdDev_us' in row else 0
            results['fused_moe_instances'] = row['Instances'] if 'Instances' in row else 0
            
            if results['fused_moe_min_us'] > 0:
                results['fused_moe_max_min_ratio'] = results['fused_moe_max_us'] / results['fused_moe_min_us']
            else:
                results['fused_moe_max_min_ratio'] = float('inf')
            
            if results['fused_moe_avg_us'] > 0:
                results['fused_moe_cv'] = results['fused_moe_std_us'] / results['fused_moe_avg_us']
            else:
                results['fused_moe_cv'] = 0
            
            if 'Med_us' in row and row['Med_us'] > 0:
                results['fused_moe_med_us'] = row['Med_us']
                results['fused_moe_med_mean_ratio'] = row['Med_us'] / results['fused_moe_avg_us']
    
    nccl_mask = df['Name'].str.lower().str.contains('nccl', na=False)
    nccl_df = df[nccl_mask]
    
    if len(nccl_df) > 0:
        results['nccl_kernels'] = nccl_df['Name'].tolist()
        results['nccl_total_time_ms'] = nccl_df['Total Time_us'].sum() / 1000
        results['nccl_time_pct'] = nccl_df['Time_pct'].sum() if 'Time_pct' in nccl_df.columns else 0
        results['nccl_instances'] = nccl_df['Instances'].sum() if 'Instances' in nccl_df.columns else 0
        
        alltoall = nccl_df[nccl_df['Name'].str.lower().str.contains('alltoall', na=False)]
        if len(alltoall) > 0:
            results['alltoall_total_ms'] = alltoall['Total Time_us'].sum() / 1000
            results['alltoall_instances'] = alltoall['Instances'].sum() if 'Instances' in alltoall.columns else 0
            results['alltoall_avg_us'] = alltoall['Avg_us'].mean()
            results['alltoall_max_us'] = alltoall['Max_us'].max()
            results['alltoall_min_us'] = alltoall['Min_us'].min()
            if results['alltoall_min_us'] > 0:
                results['alltoall_max_min_ratio'] = results['alltoall_max_us'] / results['alltoall_min_us']
        
        allreduce = nccl_df[nccl_df['Name'].str.lower().str.contains('allreduce', na=False)]
        if len(allreduce) > 0:
            results['allreduce_total_ms'] = allreduce['Total Time_us'].sum() / 1000
            results['allreduce_instances'] = allreduce['Instances'].sum() if 'Instances' in allreduce.columns else 0
    
    attn_mask = df['Name'].str.lower().str.contains('attention|flash|attn', na=False)
    attn_df = df[attn_mask]
    
    if len(attn_df) > 0:
        results['attention_total_ms'] = attn_df['Total Time_us'].sum() / 1000
        results['attention_time_pct'] = attn_df['Time_pct'].sum() if 'Time_pct' in attn_df.columns else 0
    
    df_sorted = df.sort_values('Total Time_us', ascending=False)
    results['top5_kernels'] = []
    for _, row in df_sorted.head(5).iterrows():
        results['top5_kernels'].append({
            'name': row['Name'],
            'total_ms': row['Total Time_us'] / 1000,
            'pct': row['Time_pct'] if 'Time_pct' in row else 0,
            'instances': row['Instances'] if 'Instances' in row else 0
        })
    
    results['total_kernel_time_ms'] = df['Total Time_us'].sum() / 1000
    results['total_kernel_instances'] = df['Instances'].sum() if 'Instances' in df.columns else 0
    
    return results


def print_metrics(results: dict):
    print(f"\n{'='*60}")
    print(f"  {results.get('label', 'Trace Analysis')}")
    print('='*60)
    
    print(f"\n--- MoE/Expert Kernels ---")
    if 'moe_kernels' in results:
        print(f"Kernels: {', '.join(results['moe_kernels'][:3])}...")
        print(f"Total time: {results.get('moe_total_time_ms', 0):.2f} ms ({results.get('moe_time_pct', 0):.1f}%)")
    
    if 'fused_moe_avg_us' in results:
        print(f"\nfused_moe_kernel stats:")
        print(f"  Total:     {results['fused_moe_total_ms']:.2f} ms")
        print(f"  Instances: {results.get('fused_moe_instances', 'N/A'):,}")
        print(f"  Avg:       {results['fused_moe_avg_us']:.2f} μs")
        print(f"  Med:       {results.get('fused_moe_med_us', 0):.2f} μs")
        print(f"  Min:       {results['fused_moe_min_us']:.2f} μs")
        print(f"  Max:       {results['fused_moe_max_us']:.2f} μs")
        print(f"  StdDev:    {results.get('fused_moe_std_us', 0):.2f} μs")
        print(f"  CV (σ/μ):  {results.get('fused_moe_cv', 0):.4f}")
        print(f"  Max/Min:   {results.get('fused_moe_max_min_ratio', 0):.2f}x")
    
    print(f"\n--- NCCL Communication ---")
    if 'nccl_total_time_ms' in results:
        print(f"Total NCCL time: {results['nccl_total_time_ms']:.2f} ms ({results.get('nccl_time_pct', 0):.1f}%)")
        print(f"Total instances: {results.get('nccl_instances', 0):,}")
    
    if 'alltoall_total_ms' in results:
        print(f"\nAllToAll stats:")
        print(f"  Total:     {results['alltoall_total_ms']:.2f} ms")
        print(f"  Instances: {results.get('alltoall_instances', 0):,}")
        print(f"  Avg:       {results.get('alltoall_avg_us', 0):.2f} μs")
        print(f"  Max/Min:   {results.get('alltoall_max_min_ratio', 0):.2f}x")
    
    if 'allreduce_total_ms' in results:
        print(f"\nAllReduce: {results['allreduce_total_ms']:.2f} ms ({results.get('allreduce_instances', 0):,} calls)")
    
    print(f"\n--- Top 5 Kernels by Time ---")
    for i, k in enumerate(results.get('top5_kernels', []), 1):
        print(f"  {i}. {k['name'][:40]:<40} {k['total_ms']:>10.2f} ms ({k['pct']:.1f}%)")


def compare_traces(baseline: dict, eplb: dict):
    print("\n" + "="*70)
    print("  COMPARISON: EPLB OFF vs EPLB ON")
    print("="*70)
    
    metrics = [
        ('fused_moe_total_ms', 'fused_moe Total Time', 'ms', 'lower'),
        ('fused_moe_avg_us', 'fused_moe Avg Duration', 'μs', 'lower'),
        ('fused_moe_max_us', 'fused_moe Max Duration', 'μs', 'lower'),
        ('fused_moe_std_us', 'fused_moe StdDev', 'μs', 'lower'),
        ('fused_moe_cv', 'fused_moe CV (σ/μ)', '', 'lower'),
        ('fused_moe_max_min_ratio', 'fused_moe Max/Min Ratio', 'x', 'lower'),
        ('nccl_total_time_ms', 'NCCL Total Time', 'ms', 'lower'),
        ('alltoall_total_ms', 'AllToAll Total Time', 'ms', 'lower'),
        ('alltoall_instances', 'AllToAll Instances', '', 'neutral'),
        ('alltoall_max_min_ratio', 'AllToAll Max/Min Ratio', 'x', 'lower'),
        ('allreduce_total_ms', 'AllReduce Total Time', 'ms', 'lower'),
    ]
    
    print(f"\n{'Metric':<30} {'EPLB OFF':>14} {'EPLB ON':>14} {'Change':>12} {'Verdict':>10}")
    print("-" * 82)
    
    improved_count = 0
    total_count = 0
    
    for key, name, unit, better in metrics:
        if key not in baseline or key not in eplb:
            continue
        
        b_val = baseline[key]
        e_val = eplb[key]
        
        if b_val == 0:
            change_pct = 0 if e_val == 0 else float('inf')
        else:
            change_pct = (e_val - b_val) / b_val * 100
        
        if better == 'lower':
            is_improved = e_val < b_val
        elif better == 'higher':
            is_improved = e_val > b_val
        else:
            is_improved = None  # neutral
        
        if unit == 'x':
            b_str = f"{b_val:.2f}x"
            e_str = f"{e_val:.2f}x"
        elif unit == 'ms':
            b_str = f"{b_val:.2f} ms"
            e_str = f"{e_val:.2f} ms"
        elif unit == 'μs':
            b_str = f"{b_val:.2f} μs"
            e_str = f"{e_val:.2f} μs"
        elif isinstance(b_val, int) or (isinstance(b_val, float) and b_val == int(b_val)):
            b_str = f"{int(b_val):,}"
            e_str = f"{int(e_val):,}"
        else:
            b_str = f"{b_val:.4f}"
            e_str = f"{e_val:.4f}"

        if abs(change_pct) < 0.1:
            direction = "="
        elif change_pct < 0:
            direction = "↓"
        else:
            direction = "↑"
        
        if is_improved is None:
            verdict = "-"
        elif is_improved:
            verdict = "✓ Better"
            improved_count += 1
        else:
            verdict = "✗ Worse"
        
        if is_improved is not None:
            total_count += 1
        
        print(f"{name:<30} {b_str:>14} {e_str:>14} {direction}{abs(change_pct):>10.1f}% {verdict:>10}")
    
    print("\n" + "-" * 82)
    print("KEY INSIGHTS:")
    
    if 'fused_moe_total_ms' in baseline and 'alltoall_total_ms' in baseline:
        b_ratio = baseline['alltoall_total_ms'] / baseline['fused_moe_total_ms'] if baseline['fused_moe_total_ms'] > 0 else 0
        e_ratio = eplb['alltoall_total_ms'] / eplb['fused_moe_total_ms'] if eplb['fused_moe_total_ms'] > 0 else 0
        change = (e_ratio - b_ratio) / b_ratio * 100 if b_ratio > 0 else 0
        print(f"  Comm/Compute Ratio (AllToAll/MoE): {b_ratio:.3f} → {e_ratio:.3f} ({change:+.1f}%)")
    
    if 'fused_moe_cv' in baseline and 'fused_moe_cv' in eplb:
        cv_reduction = (1 - eplb['fused_moe_cv'] / baseline['fused_moe_cv']) * 100 if baseline['fused_moe_cv'] > 0 else 0
        print(f"  Load Balance Improvement (CV reduction): {cv_reduction:.1f}%")
    
    if 'fused_moe_max_min_ratio' in baseline and 'fused_moe_max_min_ratio' in eplb:
        ratio_reduction = (1 - eplb['fused_moe_max_min_ratio'] / baseline['fused_moe_max_min_ratio']) * 100 if baseline['fused_moe_max_min_ratio'] > 0 else 0
        print(f"  Straggler Reduction (Max/Min ratio reduction): {ratio_reduction:.1f}%")
    
    print("\n" + "="*82)
    print(f"OVERALL: {improved_count}/{total_count} metrics improved with EPLB")
    
    if improved_count > total_count * 0.6:
        print("VERDICT: EPLB is providing significant benefit ✓✓")
    elif improved_count > total_count * 0.4:
        print("VERDICT: EPLB shows mixed results, may need tuning")
    else:
        print("VERDICT: EPLB overhead may outweigh benefits ✗")
    print("="*82)
    
    return {
        'improved_count': improved_count,
        'total_count': total_count,
        'baseline': baseline,
        'eplb': eplb
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_kernel_summary.py <eplb_off.csv> <eplb_on.csv>")
        print("\nExport CSVs from Nsight Systems:")
        print("  1. Open trace in nsys-ui")
        print("  2. Go to 'CUDA GPU Kernel Summary' in Analysis Summary")
        print("  3. Right-click → Export to CSV")
        sys.exit(1)
    
    baseline_csv = sys.argv[1]
    eplb_csv = sys.argv[2]
    
    print("Loading kernel summaries...")
    baseline_df = parse_csv(baseline_csv)
    eplb_df = parse_csv(eplb_csv)
    
    print("\nExtracting metrics...")
    baseline_metrics = extract_metrics(baseline_df, "EPLB OFF (Baseline)")
    eplb_metrics = extract_metrics(eplb_df, "EPLB ON")
    
    print_metrics(baseline_metrics)
    print_metrics(eplb_metrics)
    
    comparison = compare_traces(baseline_metrics, eplb_metrics)
    
    import json
    
    def convert_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    output = {
        'baseline': convert_types(baseline_metrics),
        'eplb': convert_types(eplb_metrics),
        'comparison': {
            'improved_count': comparison['improved_count'],
            'total_count': comparison['total_count']
        }
    }
    
    output_file = 'kernel_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()