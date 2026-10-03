import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from config import TERRAIN_COLORS, CAMERA_RANGE
from planet import generate_planet
from camera import take_photo

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
row = int(sys.argv[2]) if len(sys.argv) > 2 else 12
col = int(sys.argv[3]) if len(sys.argv) > 3 else 20
terrain, elevation = generate_planet(seed)
r = CAMERA_RANGE

# What is really there (ground truth), cut to the same window as the photo
padded_truth = np.pad(terrain, r, constant_values=0)
truth_window = padded_truth[row:row + 2 * r + 1, col:col + 2 * r + 1]

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(truth_window, cmap=ListedColormap(TERRAIN_COLORS), vmin=0, vmax=4)
axes[0].set_title("Ground truth (AI never sees this)")

for axis, haze in zip(axes[1:], (0.0, 0.4, 0.9)):
    photo = take_photo(terrain, elevation, (row, col), haze,
                       np.random.default_rng(1))
    axis.imshow(photo, cmap="gray", vmin=0, vmax=1)
    axis.set_title(f"Camera photo, haze {haze}")

for axis in axes:
    axis.axis("off")
plt.tight_layout()
plt.show()