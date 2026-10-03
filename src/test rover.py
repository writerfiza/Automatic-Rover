import numpy as np
from config import FREE, ROCK
from rover import Rover

# A tiny 5x5 world: all free ground, with one rock at row 2, column 3
grid = np.full((5, 5), FREE)
grid[2, 3] = ROCK

# Test 1: walking into the rock is a collision and the rover stays put
rover = Rover(start=(2, 2))
assert rover.try_move("E", grid) == "collision"
assert (rover.row, rover.col) == (2, 2)
assert rover.collisions == 1

# Test 2: a move into free ground works
assert rover.try_move("N", grid) == "moved"
assert (rover.row, rover.col) == (1, 2)

# Test 3: walking off the map is refused
edge_rover = Rover(start=(0, 0))
assert edge_rover.try_move("N", grid) == "out_of_bounds"

print("All rover tests passed")