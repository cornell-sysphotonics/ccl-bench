import matplotlib.pyplot as plt

# Experiment names
experiments = ["Experiment 1", "Experiment 2"]

# Straggler ratios (computed from your CSVs)
straggler_ratios = [
    0.36282324218746 / 2450.03998046875,     # Exp1
    3.028456054686103 / 12796.272885742188   # Exp2
]

# Colors (same style as previous plot)
colors = ['#003366', '#D2691E']  # deep blue, golden brown

plt.figure(figsize=(6, 5))
plt.bar(experiments, straggler_ratios, color=colors)

plt.ylabel("Straggler Ratio (Straggler Delay / Iteration Length)")
plt.title("Straggler Ratio Comparison")

# Annotate values on bars
for i, v in enumerate(straggler_ratios):
    plt.text(i, v, f"{v:.2e}", ha="center", va="bottom")

plt.tight_layout()
plt.show()
