import sys
import numpy as np
import matplotlib.pyplot as plt

print("Python version:", sys.version.split()[0])
print("NumPy version:", np.__version__)

# Make a 20-row by 30-column grid filled with zeros (zero = free ground)
grid = np.zeros((20, 30), dtype=int)

# Put a "rock" block (value 1) in part of the grid
grid[5:8, 10:14] = 1

print("Grid shape:", grid.shape)
print("Number of rock cells:", int(np.sum(grid == 1)))

plt.imshow(grid, cmap="viridis")
plt.title("Hello map")
plt.show()
