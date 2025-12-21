# Extracts request_throughput metric from benchmark JSON files
import json
import sys
import yaml
from pathlib import Path

def validate_trace(trace_dir):
    trace_dir = Path(trace_dir)
    yaml_files = list(trace_dir.glob("*.yaml"))
    if not yaml_files:
        return False, "No workload YAML file found in trace directory"
    
    try:
        with open(yaml_files[0]) as f:
            workload = yaml.safe_load(f)
        
        if not workload or "workload" not in workload:
            return False, "Invalid workload YAML format"
        
        model_info = workload.get("workload", {}).get("model", {})
        phase = model_info.get("phase", "")
        
        if phase != "inference":
            return False, f"Request throughput metric requires inference phase, found: {phase}"
        
        return True, None
    except Exception as e:
        return False, f"Error reading workload YAML: {e}"

exp_dir = Path(sys.argv[1])
is_valid, error_msg = validate_trace(exp_dir)
if not is_valid:
    print(f"Error: {error_msg}", file=sys.stderr)
    sys.exit(1)

results_dir = exp_dir / "results_json"
if not results_dir.exists():
    print("Error: results_json directory not found", file=sys.stderr)
    sys.exit(1)

values = []
for json_file in results_dir.glob("*.json"):
    with open(json_file) as f:
        data = json.load(f)
        if "request_throughput" in data:
            values.append(data["request_throughput"])
print(sum(values) / len(values) if values else 0)

