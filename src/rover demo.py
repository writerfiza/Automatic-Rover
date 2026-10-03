import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from config import TERRAIN_COLORS, START, GOAL, MOVES
from planet import generate_planet
from rover import Rover

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
terrain, _ = generate_planet(seed)

rover = Rover()
rng = np.random.default_rng(seed)
move_names = list(MOVES)
outcomes = {"moved": 0, "collision": 0, "out_of_bounds": 0}

# A blind random walk. This is NOT smart. It only tests the rover mechanics.
for _ in range(400):
    result = rover.try_move(rng.choice(move_names), terrain)
    outcomes[result] += 1

print("Outcomes:", outcomes)
print("Distance travelled:", round(rover.distance_m, 1), "m")
print("Dust crossings:", rover.dust_crossings)
print("Final position (row, col):", (rover.row, rover.col))

plt.imshow(terrain, cmap=ListedColormap(TERRAIN_COLORS), vmin=0, vmax=4)
trail = np.array(rover.trail)
plt.plot(trail[:, 1], trail[:, 0], "w-", linewidth=1)
plt.plot(START[1], START[0], "wo", markersize=10)
plt.plot(GOAL[1], GOAL[0], "r*", markersize=14)
plt.title(f"Blind random walk (seed {seed})")
plt.show()