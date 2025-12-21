import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.trace_analyzer import TraceAnalyzer

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/calc_all_metrics.py <trace_file_csv>")
        sys.exit(1)
        
    trace_path = sys.argv[1]
    if not os.path.exists(trace_path):
        print(f"Error: File {trace_path} not found")
        sys.exit(1)
        
    print(f"Analyzing {trace_path}...")
    try:
        analyzer = TraceAnalyzer(trace_path)
        
        bubble = analyzer.calculate_bubble_ratio()
        comm = analyzer.calculate_comm_overhead()
        sm = analyzer.calculate_sm_efficiency()
        
        print("-" * 40)
        print(f"Pipeline Bubble Ratio:   {bubble:.2f}%")
        print(f"Communication Overhead:  {comm:.2f}%")
        print(f"SM Efficiency (Active):  {sm:.2f}%")
        print("-" * 40)
        
    except Exception as e:
        print(f"Error analyzing trace: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
