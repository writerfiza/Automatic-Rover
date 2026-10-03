import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from config import TERRAIN_COLORS, START, GOAL
from planet import generate_planet
from planner import true_cost_function, find_path

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
terrain, _ = generate_planet(seed)
height, width = terrain.shape

cost_fn_grid = true_cost_function(terrain)
cost_fn = lambda r, c: cost_fn_grid(r, c)
path = find_path(cost_fn, START, GOAL, height, width)

if path is None:
    print("No path found for this seed. Try a different seed.")
else:
    print("Path length (steps):", len(path) - 1)
    plt.imshow(terrain, cmap=ListedColormap(TERRAIN_COLORS), vmin=0, vmax=4)
    path_arr = np.array(path)
    plt.plot(path_arr[:, 1], path_arr[:, 0], "y--", linewidth=2, label="Oracle path")
    plt.plot(START[1], START[0], "wo", markersize=10)
    plt.plot(GOAL[1], GOAL[0], "r*", markersize=14)
    plt.legend()
    plt.title(f"Oracle path planning (seed {seed})")
    plt.show()