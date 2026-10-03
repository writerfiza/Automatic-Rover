import csv
import numpy as np
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results/experiment_results.csv")))
methods = []
for r in rows:
    if r["method"] not in methods:
        methods.append(r["method"])

success_rates = []
avg_collisions = []
for method in methods:
    method_rows = [r for r in rows if r["method"] == method]
    reached = [r["reached_goal"] == "True" for r in method_rows]
    success_rates.append(100 * sum(reached) / len(reached))
    avg_collisions.append(np.mean([float(r["collisions"]) for r in method_rows]))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

colors = ["#8b8d99", "#e8582b", "#2fd08c"]
axes[0].bar(methods, success_rates, color=colors)
axes[0].set_ylabel("Success rate (%)")
axes[0].set_title("Mission success rate")
axes[0].set_ylim(0, 100)
for i, v in enumerate(success_rates):
    axes[0].text(i, v + 2, f"{v:.0f}%", ha="center")

axes[1].bar(methods, avg_collisions, color=colors)
axes[1].set_ylabel("Average collisions per mission")
axes[1].set_title("Collision rate")
for i, v in enumerate(avg_collisions):
    axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center")

for axis in axes:
    axis.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("results/comparison_chart.png", dpi=150)
plt.show()
print("Saved to results/comparison_chart.png")