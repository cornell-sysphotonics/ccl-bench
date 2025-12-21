import matplotlib.pyplot as plt

ranks = list(range(4))
end_times = [
    2450.3513408203125,
    2450.1744833984376,
    2450.176791015625,
    2450.537306640625,
]

plt.figure()
plt.bar(ranks, end_times)
plt.xlabel("Rank ID")
plt.ylabel("Iteration End Time (ms)")
plt.title("Average Per-rank Iteration End Time (4 Ranks, TP)")

# Set y-axis range to focus on the upper part
plt.ylim(2447, 2452)  # Adjust this range based on the desired focus

plt.tight_layout()
plt.show()
