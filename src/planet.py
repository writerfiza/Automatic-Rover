import numpy as np
import config
from config import (
    GRID_HEIGHT, GRID_WIDTH, CELL_SIZE_M,
    FREE, DUST, ROCK, CRATER, STEEP,
    NUM_CRATERS, NUM_HILLS, NUM_DUST_PATCHES,
    STEEP_SLOPE_DEG, START, GOAL,
)


def make_elevation(rng):
    """Create smooth hills. Returns a grid of heights in metres."""
    rows, cols = np.mgrid[0:GRID_HEIGHT, 0:GRID_WIDTH]
    elevation = np.zeros((GRID_HEIGHT, GRID_WIDTH))
    for _ in range(NUM_HILLS):
        centre_row = rng.uniform(0, GRID_HEIGHT)
        centre_col = rng.uniform(0, GRID_WIDTH)
        height = rng.uniform(3, 7)
        spread = rng.uniform(2.5, 5)
        dist_sq = (rows - centre_row) ** 2 + (cols - centre_col) ** 2
        elevation += height * np.exp(-dist_sq / (2 * spread ** 2))
    return elevation


def slope_in_degrees(elevation):
    """Work out how steep every cell is, in degrees."""
    d_row, d_col = np.gradient(elevation, CELL_SIZE_M)
    return np.degrees(np.arctan(np.hypot(d_row, d_col)))


def circle_mask(centre_row, centre_col, radius):
    """A True/False grid: True for cells inside a circle."""
    rows, cols = np.mgrid[0:GRID_HEIGHT, 0:GRID_WIDTH]
    return np.hypot(rows - centre_row, cols - centre_col) < radius


def generate_planet(seed):
    """Build a planet. Returns (terrain, elevation)."""
    rng = np.random.default_rng(seed)
    terrain = np.full((GRID_HEIGHT, GRID_WIDTH), FREE, dtype=int)

    # 1. Dust patches
    for _ in range(NUM_DUST_PATCHES):
        mask = circle_mask(rng.uniform(0, GRID_HEIGHT),
                           rng.uniform(0, GRID_WIDTH),
                           rng.uniform(3, 7))
        terrain[mask] = DUST

    # 2. Hills and steep slopes
    elevation = make_elevation(rng)
    terrain[slope_in_degrees(elevation) > STEEP_SLOPE_DEG] = STEEP

    # 3. Scattered rocks
    terrain[rng.random(terrain.shape) < config.ROCK_FRACTION] = ROCK

    # 4. Craters (kept away from the start and goal)
    placed = 0
    attempts = 0
    while placed < NUM_CRATERS and attempts < 200:
        attempts += 1
        radius = rng.uniform(2.5, 5)
        c_row = rng.uniform(3, GRID_HEIGHT - 3)
        c_col = rng.uniform(3, GRID_WIDTH - 3)
        too_close = (np.hypot(c_row - START[0], c_col - START[1]) < radius + 4 or
                     np.hypot(c_row - GOAL[0], c_col - GOAL[1]) < radius + 4)
        if too_close:
            continue
        terrain[circle_mask(c_row, c_col, radius * 0.85)] = CRATER
        placed += 1

    # 5. Keep the start and goal areas free of hazards
    for row, col in (START, GOAL):
        terrain[max(0, row - 2):row + 3, max(0, col - 2):col + 3] = FREE

    return terrain, elevation