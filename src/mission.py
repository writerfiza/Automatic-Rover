import numpy as np
from config import (
    GRID_HEIGHT, GRID_WIDTH, START, GOAL, CAMERA_RANGE,
    HAZARDS, DUST, UNKNOWN, COST_UNKNOWN, COST_FREE, COST_DUST,
    COST_NEAR_HAZARD_1, COST_NEAR_HAZARD_2, MOVES,
)
from rover import Rover
from camera import take_photo
from planner import find_path, hazard_distance_map


def belief_cost_function(belief):
    """Like true_cost_function in planner.py, but built from the rover's
    BELIEF map instead of the ground truth. Unknown cells get a moderate
    cost, so the rover is willing to explore but prefers confirmed-safe ground."""
    height, width = belief.shape
    known_hazard = np.isin(belief, HAZARDS)
    dist = np.full((height, width), 99, dtype=int)
    queue = []
    for r in range(height):
        for c in range(width):
            if known_hazard[r, c]:
                dist[r, c] = 0
                queue.append((r, c))
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))

    def cost(r, c):
        val = belief[r, c]
        if val in HAZARDS:
            return float("inf")
        if val == UNKNOWN:
            c_base = COST_UNKNOWN
        elif val == DUST:
            c_base = COST_DUST
        else:
            c_base = COST_FREE
        if dist[r, c] == 1:
            c_base += COST_NEAR_HAZARD_1
        elif dist[r, c] == 2:
            c_base += COST_NEAR_HAZARD_2
        return c_base
    return cost


def run_mission(terrain, elevation, perceive_fn, haze, start=START, goal=GOAL,
                 max_steps=400, rng=None, view_range=CAMERA_RANGE):
    """Run one full autonomous mission.

    perceive_fn(photo, rover_pos, view_range) -> {(row,col): predicted_class}
    This is the pluggable part: pass in the rule-based baseline, or later,
    the trained AI classifier. Everything else in this loop stays identical,
    which is what makes the later Baseline vs AI comparison fair.

    Returns a result dictionary with the rover, belief map, and outcome.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    height, width = terrain.shape
    belief = np.full((height, width), UNKNOWN, dtype=int)
    rover = Rover(start=start)
    path = []
    outcome = "running"
    replans = 0

    for step in range(max_steps):
        # OBSERVE
        photo = take_photo(terrain, elevation, (rover.row, rover.col), haze, rng, view_range)

        # DETECT (perceive_fn is the baseline rule, or later, the AI)
        predictions = perceive_fn(photo, (rover.row, rover.col), view_range)
        changed_hazard = False
        for (r, c), pred in predictions.items():
            if 0 <= r < height and 0 <= c < width:
                old = belief[r, c]
                if pred in HAZARDS and old not in HAZARDS:
                    changed_hazard = True
                belief[r, c] = pred

        # DECIDE
        old_path = path
        cost_fn = belief_cost_function(belief)
        new_path = find_path(cost_fn, (rover.row, rover.col), goal, height, width)
        if new_path is None:
            outcome = "stuck"
            break
        blocked_ahead = any(
            belief[r, c] in HAZARDS for r, c in old_path[1:6]
        ) if len(old_path) > 1 else False
        if blocked_ahead or not old_path:
            replans += 1
        path = new_path

        if (rover.row, rover.col) == goal:
            outcome = "reached_goal"
            break

        # NAVIGATE
        next_row, next_col = path[1]
        move_name = None
        for name, (dr, dc) in MOVES.items():
            if (rover.row + dr, rover.col + dc) == (next_row, next_col):
                move_name = name
                break
        result = rover.try_move(move_name, terrain)
        if result == "collision":
            belief[next_row, next_col] = terrain[next_row, next_col]  # now it knows
            path = []

    else:
        outcome = "max_steps_reached"

    return {
        "rover": rover,
        "belief": belief,
        "outcome": outcome,
        "replans": replans,
        "steps": rover.steps_taken,
        "collisions": rover.collisions,
        "distance_m": rover.distance_m,
        "dust_crossings": rover.dust_crossings,
    }