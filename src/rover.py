import numpy as np
from config import CELL_SIZE_M, START, HAZARDS, DUST, MOVES


class Rover:
    def __init__(self, start=START):
        self.row, self.col = start
        self.trail = [start]          # every position visited, in order
        self.distance_m = 0.0
        self.steps_taken = 0
        self.collisions = 0
        self.dust_crossings = 0

    def try_move(self, move_name, terrain):
        """Try one step. Returns 'moved', 'collision' or 'out_of_bounds'."""
        d_row, d_col = MOVES[move_name]
        new_row = self.row + d_row
        new_col = self.col + d_col

        height, width = terrain.shape
        if not (0 <= new_row < height and 0 <= new_col < width):
            return "out_of_bounds"

        if terrain[new_row, new_col] in HAZARDS:
            self.collisions += 1
            return "collision"

        if terrain[new_row, new_col] == DUST:
            self.dust_crossings += 1

        self.distance_m += np.hypot(d_row, d_col) * CELL_SIZE_M
        self.row, self.col = new_row, new_col
        self.steps_taken += 1
        self.trail.append((new_row, new_col))
        return "moved"