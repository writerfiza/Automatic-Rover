import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from config import TERRAIN_COLORS, START, GOAL
from planet import generate_planet
from perception_ai import observe_with_ai
from mission import run_mission

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
haze = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
terrain, elevation = generate_planet(seed)

result = run_mission(terrain, elevation, observe_with_ai, haze,
                      rng=np.random.default_rng(seed))

print("Outcome:", result["outcome"])
print("Steps:", result["steps"])
print("Collisions:", result["collisions"])
print("Replans:", result["replans"])
print("Distance:", round(result["distance_m"], 1), "m")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].imshow(terrain, cmap=ListedColormap(TERRAIN_COLORS), vmin=0, vmax=4)
axes[0].set_title("Ground truth")

belief_display = np.where(result["belief"] == -1, np.nan, result["belief"])
cmap = ListedColormap(TERRAIN_COLORS)
cmap.set_bad(color="#333344")
axes[1].imshow(belief_display, cmap=cmap, vmin=0, vmax=4)
axes[1].set_title(f"Rover's belief (AI perception), outcome: {result['outcome']}")

for axis in axes:
    trail = np.array(result["rover"].trail)
    axis.plot(trail[:, 1], trail[:, 0], "w-", linewidth=1.5)
    axis.plot(START[1], START[0], "wo", markersize=8)
    axis.plot(GOAL[1], GOAL[0], "r*", markersize=12)

plt.tight_layout()
plt.show()
