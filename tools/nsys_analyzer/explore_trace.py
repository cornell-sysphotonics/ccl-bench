#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore nsys trace structure to understand available data
探索 nsys trace 的结构，了解可用的数据
"""

import sqlite3
import os
import sys


def explore_trace(sqlite_file):
    """探索 trace 中的可用信息"""
    
    if not os.path.exists(sqlite_file):
        print(f"Error: File not found: {sqlite_file}")
        return
    
    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()
    
    print("="*80)
    print("TRACE EXPLORATION REPORT")
    print("="*80)
    
    # 1. 列出所有表
    print("\n[1] Available Tables:")
    print("-"*40)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cursor.fetchone()[0]
        print(f"  {table[0]:<50} {count:>10} rows")
    
    # 2. 检查 NVTX 事件
    print("\n[2] NVTX Events Analysis:")
    print("-"*40)
    try:
        # 查看 NVTX 表结构
        cursor.execute("PRAGMA table_info(NVTX_EVENTS)")
        columns = cursor.fetchall()
        print("  NVTX_EVENTS columns:")
        for col in columns:
            print(f"    {col[1]:<20} {col[2]}")
        
        # 统计 NVTX 范围类型
        cursor.execute("""
            SELECT s.value, COUNT(*) as cnt
            FROM NVTX_EVENTS e
            LEFT JOIN StringIds s ON e.text = s.id
            WHERE e.end IS NOT NULL
            GROUP BY s.value
            ORDER BY cnt DESC
            LIMIT 30
        """)
        nvtx_ranges = cursor.fetchall()
        print(f"\n  Top NVTX ranges (with text):")
        for name, count in nvtx_ranges:
            if name:
                print(f"    {count:>8}x  {name[:70]}")
        
        # 检查是否有 NCCL 相关的 NVTX
        cursor.execute("""
            SELECT s.value, COUNT(*) as cnt
            FROM NVTX_EVENTS e
            LEFT JOIN StringIds s ON e.text = s.id
            WHERE s.value LIKE '%NCCL%' OR s.value LIKE '%nccl%'
               OR s.value LIKE '%COMM%' OR s.value LIKE '%comm%'
               OR s.value LIKE '%DP%' OR s.value LIKE '%TP%'
               OR s.value LIKE '%parallel%'
            GROUP BY s.value
            ORDER BY cnt DESC
        """)
        comm_nvtx = cursor.fetchall()
        print(f"\n  Communication-related NVTX:")
        for name, count in comm_nvtx:
            print(f"    {count:>8}x  {name}")
            
    except Exception as e:
        print(f"  Error querying NVTX: {e}")
    
    # 3. 检查 CUDA Kernels
    print("\n[3] CUDA Kernels Analysis:")
    print("-"*40)
    try:
        # 查看 kernel 表结构
        cursor.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")
        columns = cursor.fetchall()
        print("  CUPTI_ACTIVITY_KIND_KERNEL columns:")
        for col in columns:
            print(f"    {col[1]:<20} {col[2]}")
        
        # 统计 NCCL kernels
        cursor.execute("""
            SELECT s.value, COUNT(*) as cnt, SUM(k.end - k.start)/1e6 as total_ms
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName = s.id
            WHERE s.value LIKE '%nccl%'
            GROUP BY s.value
            ORDER BY total_ms DESC
            LIMIT 20
        """)
        nccl_kernels = cursor.fetchall()
        print(f"\n  NCCL Kernels:")
        print(f"    {'Count':>8}  {'Time(ms)':>12}  Kernel Name")
        for name, count, time_ms in nccl_kernels:
            print(f"    {count:>8}  {time_ms:>12.2f}  {name[:60]}")
            
        # 检查是否有 stream 信息
        cursor.execute("""
            SELECT streamId, COUNT(*) as cnt
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName = s.id
            WHERE s.value LIKE '%nccl%'
            GROUP BY streamId
            ORDER BY cnt DESC
            LIMIT 10
        """)
        streams = cursor.fetchall()
        print(f"\n  NCCL kernels by stream:")
        for stream_id, count in streams:
            print(f"    Stream {stream_id}: {count} kernels")
            
    except Exception as e:
        print(f"  Error querying kernels: {e}")
    
    # 4. 检查是否有 NCCL 特定信息
    print("\n[4] Looking for NCCL-specific tables:")
    print("-"*40)
    for table in tables:
        if 'nccl' in table[0].lower() or 'comm' in table[0].lower():
            print(f"  Found: {table[0]}")
            cursor.execute(f"PRAGMA table_info({table[0]})")
            cols = cursor.fetchall()
            for col in cols:
                print(f"    {col[1]:<20} {col[2]}")
    
    # 5. 检查 GPU 和设备信息
    print("\n[5] Device Information:")
    print("-"*40)
    try:
        cursor.execute("""
            SELECT DISTINCT deviceId, contextId
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            ORDER BY deviceId
        """)
        devices = cursor.fetchall()
        print(f"  Devices used:")
        for device_id, context_id in devices:
            print(f"    Device {device_id}, Context {context_id}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 6. 检查是否可以关联 NVTX 和 Kernel
    print("\n[6] NVTX-Kernel Correlation Test:")
    print("-"*40)
    try:
        # 取一个 NCCL kernel，检查是否有包含它的 NVTX 范围
        cursor.execute("""
            SELECT k.start, k.end, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName = s.id
            WHERE s.value LIKE '%nccl%AllGather%'
            LIMIT 1
        """)
        sample_kernel = cursor.fetchone()
        
        if sample_kernel:
            k_start, k_end, k_name = sample_kernel
            print(f"  Sample kernel: {k_name[:50]}")
            print(f"  Kernel time: {k_start} - {k_end}")
            
            # 查找包含这个 kernel 的 NVTX 范围
            cursor.execute("""
                SELECT s.value, e.start, e.end
                FROM NVTX_EVENTS e
                LEFT JOIN StringIds s ON e.text = s.id
                WHERE e.start <= ? AND e.end >= ?
                ORDER BY (e.end - e.start) ASC
                LIMIT 5
            """, (k_start, k_end))
            containing_nvtx = cursor.fetchall()
            
            print(f"\n  NVTX ranges containing this kernel:")
            for name, start, end in containing_nvtx:
                print(f"    {name[:60]}")
        else:
            print("  No NCCL AllGather kernel found")
            
    except Exception as e:
        print(f"  Error: {e}")
    
    # 7. 检查是否有 ProcessGroup 或 Communicator 信息
    print("\n[7] Looking for ProcessGroup/Communicator info:")
    print("-"*40)
    try:
        # 在 NVTX 中搜索
        cursor.execute("""
            SELECT DISTINCT s.value
            FROM NVTX_EVENTS e
            LEFT JOIN StringIds s ON e.text = s.id
            WHERE s.value LIKE '%group%' 
               OR s.value LIKE '%comm%'
               OR s.value LIKE '%rank%'
               OR s.value LIKE '%world%'
            LIMIT 20
        """)
        group_info = cursor.fetchall()
        if group_info:
            print("  Found in NVTX:")
            for name in group_info:
                print(f"    {name[0]}")
        else:
            print("  No ProcessGroup/Communicator info found in NVTX")
            
    except Exception as e:
        print(f"  Error: {e}")
    
    # 8. 总结和建议
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    print("""
Current trace contains:
  ✓ CUDA kernel timing (precise GPU execution time)
  ✓ NVTX ranges (DeepSpeed function markers)
  ✓ Stream information
  
To accurately classify DP/TP/PP/EP communications, you need:

Option A: Add custom NVTX markers in training code
  - Add nvtx.range_push("COMM:DP:AllReduce") before NCCL calls
  - This provides direct classification without inference
  
Option B: Use NCCL_DEBUG logs
  - Run with NCCL_DEBUG=INFO to get communicator info
  - Cross-reference with nsys trace by call order
  
Option C: Use heuristics (current approach, less accurate)
  - Classify by kernel name (AllGather→DP, etc.)
  - Works for simple cases but fails with mixed parallelism
""")
    
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore_trace.py <trace.sqlite>")
        sys.exit(1)
    
    sqlite_file = sys.argv[1]
    if sqlite_file.endswith('.nsys-rep'):
        sqlite_file = sqlite_file.replace('.nsys-rep', '.sqlite')
    
    explore_trace(sqlite_file)

