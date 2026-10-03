import heapq
import numpy as np
from config import HAZARDS, DUST, COST_FREE, COST_DUST, COST_NEAR_HAZARD_1, COST_NEAR_HAZARD_2


def hazard_distance_map(terrain):
    """For every cell, how many steps to the nearest hazard (0 = is a hazard)."""
    height, width = terrain.shape
    dist = np.full((height, width), 99, dtype=int)
    queue = []
    for row in range(height):
        for col in range(width):
            if terrain[row, col] in HAZARDS:
                dist[row, col] = 0
                queue.append((row, col))
    head = 0
    while head < len(queue):
        row, col = queue[head]
        head += 1
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                nr, nc = row + d_row, col + d_col
                if 0 <= nr < height and 0 <= nc < width and dist[nr, nc] > dist[row, col] + 1:
                    dist[nr, nc] = dist[row, col] + 1
                    queue.append((nr, nc))
    return dist


def true_cost_function(terrain):
    """Build a cost(row, col) function based on the real map."""
    dist = hazard_distance_map(terrain)

    def cost(row, col):
        if terrain[row, col] in HAZARDS:
            return float("inf")
        c = COST_DUST if terrain[row, col] == DUST else COST_FREE
        if dist[row, col] == 1:
            c += COST_NEAR_HAZARD_1
        elif dist[row, col] == 2:
            c += COST_NEAR_HAZARD_2
        return c
    return cost


def find_path(cost_fn, start, goal, height, width):
    """A* search. cost_fn(row, col) -> cost to enter that cell (inf = blocked).
    Returns a list of (row, col) from start to goal, or None if no path exists."""
    def heuristic(row, col):
        return np.hypot(row - goal[0], col - goal[1])

    open_set = [(heuristic(*start), 0.0, start)]
    came_from = {}
    best_g = {start: 0.0}
    visited = set()

    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            break

        row, col = current
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                if d_row == 0 and d_col == 0:
                    continue
                nr, nc = row + d_row, col + d_col
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                step_cost = cost_fn(nr, nc)
                if step_cost == float("inf"):
                    continue
                # diagonal moves cost more, and can't cut between two blocked corners
                if d_row != 0 and d_col != 0:
                    if cost_fn(row + d_row, col) == float("inf") and cost_fn(row, col + d_col) == float("inf"):
                        continue
                    step_cost *= 1.414
                new_g = g + step_cost
                neighbour = (nr, nc)
                if new_g < best_g.get(neighbour, float("inf")):
                    best_g[neighbour] = new_g
                    came_from[neighbour] = current
                    heapq.heappush(open_set, (new_g + heuristic(nr, nc), new_g, neighbour))

    if goal not in came_from and goal != start:
        return None

    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path