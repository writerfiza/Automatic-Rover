import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from config import TERRAIN_COLORS, START, GOAL
from planet import generate_planet
from perception_baseline import observe_with_baseline
from perception_ai import observe_with_ai
from mission import run_mission

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
haze = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
terrain, elevation = generate_planet(seed)

print(f"Seed {seed}, haze {haze}\n")

baseline_result = run_mission(terrain, elevation, observe_with_baseline, haze,
                               rng=np.random.default_rng(seed))
ai_result = run_mission(terrain, elevation, observe_with_ai, haze,
                         rng=np.random.default_rng(seed))

for name, result in [("Baseline", baseline_result), ("AI", ai_result)]:
    print(f"{name:10s}  outcome: {result['outcome']:16s}  "
          f"collisions: {result['collisions']:3d}  "
          f"replans: {result['replans']:3d}  "
          f"distance: {result['distance_m']:.1f} m")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
cmap = ListedColormap(TERRAIN_COLORS)
cmap.set_bad(color="#333344")

axes[0].imshow(terrain, cmap=cmap, vmin=0, vmax=4)
axes[0].set_title("Ground truth", fontsize=13)

for axis, name, result in [(axes[1], "Baseline (rule-based)", baseline_result),
                             (axes[2], "AI (Random Forest)", ai_result)]:
    belief_display = np.where(result["belief"] == -1, np.nan, result["belief"])
    axis.imshow(belief_display, cmap=cmap, vmin=0, vmax=4)
    axis.set_title(f"{name}\n{result['outcome']}, {result['collisions']} collisions",
                    fontsize=13)
    trail = np.array(result["rover"].trail)
    axis.plot(trail[:, 1], trail[:, 0], "w-", linewidth=1.5)

for axis in axes:
    axis.plot(START[1], START[0], "wo", markersize=8)
    axis.plot(GOAL[1], GOAL[0], "r*", markersize=12)
    axis.set_xticks([])
    axis.set_yticks([])

plt.suptitle(f"Autonomous navigation: Baseline vs AI  (seed {seed}, haze {haze})", fontsize=14)
plt.tight_layout()
plt.savefig("results/demo_comparison.png", dpi=150)
plt.show()