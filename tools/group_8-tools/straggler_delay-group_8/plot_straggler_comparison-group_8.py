import matplotlib.pyplot as plt

# Data for Straggler Delay in Experiment 1 and Experiment 2
experiment_names = ['Experiment 2 (DP + TP)', 'Experiment 1 (TP)']
straggler_delays = [3.028456054686103, 0.36282324218746]

# Colors for the bars
colors = ['#003366', '#D2691E']  # Deep blue and golden brown (土黄)

# Plot
plt.figure(figsize=(8, 5))

# Create horizontal bar chart with different colors
plt.barh(experiment_names, straggler_delays, color=colors)

# Set labels and title
plt.xlabel("Straggler Delay (ms)")
plt.title("Comparison of Straggler Delay: Experiment 1 vs Experiment 2")

plt.xlim(0, 3.5)

# Display the value on each bar
for i, v in enumerate(straggler_delays):
    plt.text(v + 0.1, i, f"{v:.2f} ms", va='center', fontweight='bold')

plt.tight_layout()
plt.show()
