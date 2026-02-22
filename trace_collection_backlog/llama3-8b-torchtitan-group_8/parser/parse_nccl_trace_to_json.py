import sys, json, re

input_file = sys.argv[1]
output_file = sys.argv[2]

pattern = re.compile(r"(AllReduce|AllGather|ReduceScatter|Broadcast).*?(\d+\.\d+)\s*ms")

events = []

with open(input_file, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            events.append(float(m.group(2)))

if events:
    result = {
        "rank": int(re.findall(r"rank(\d+)", input_file)[0]),
        "num_events": len(events),
        "avg_latency_ms": sum(events) / len(events),
        "max_latency_ms": max(events),
        "min_latency_ms": min(events),
    }
else:
    result = {
        "rank": int(re.findall(r"rank(\d+)", input_file)[0]),
        "num_events": 0,
        "avg_latency_ms": None,
        "max_latency_ms": None,
        "min_latency_ms": None,
    }

with open(output_file, "w") as f:
    json.dump(result, f, indent=2)

print(f"Saved JSON to {output_file}")