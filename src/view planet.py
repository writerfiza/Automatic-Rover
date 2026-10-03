import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from config import TERRAIN_NAMES, TERRAIN_COLORS, START, GOAL
from planet import generate_planet

# Use a seed typed after the file name, or 2026 if none is given
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
terrain, elevation = generate_planet(seed)

print("Seed:", seed)
print("Map shape:", terrain.shape)
for class_id, name in TERRAIN_NAMES.items():
    count = int(np.sum(terrain == class_id))
    print(f"  {name:12s} {count:5d} cells  ({100 * count / terrain.size:.1f}%)")

plt.imshow(terrain, cmap=ListedColormap(TERRAIN_COLORS), vmin=0, vmax=4)
plt.plot(START[1], START[0], "wo", markersize=10, label="start")
plt.plot(GOAL[1], GOAL[0], "r*", markersize=14, label="goal")

legend_items = [Patch(color=TERRAIN_COLORS[i], label=TERRAIN_NAMES[i])
                for i in TERRAIN_NAMES]
plt.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.01, 1))
plt.title(f"Simulated planet (seed {seed})")
plt.tight_layout()
plt.show()