import matplotlib.pyplot as plt

ranks_16 = list(range(16))
end_times_16 = [
    12796.272885742188, 12798.2704765625, 12798.111913085937, 12796.1488359375,
    12795.4242109375, 12798.419485351562, 12798.166973632813, 12796.351691406251,
    12795.692663085938, 12798.270850585937, 12798.117356445313, 12796.368751953125,
    12795.509884765625, 12798.452666992187, 12798.314068359376, 12796.607759765626
]

plt.figure()
plt.bar(ranks_16, end_times_16)

plt.xlabel("Rank ID")
plt.ylabel("Iteration End Time (ms)")
plt.title("Average Per-rank Iteration End Time (16 Ranks, DP + TP)")

plt.ylim(12795, 12800)

ax = plt.gca()
ax.ticklabel_format(axis='y', style='plain', useOffset=False)

plt.tight_layout()
plt.show()
