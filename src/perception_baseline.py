import numpy as np
from config import PIXELS_PER_CELL, FREE, DUST, ROCK, CRATER, STEEP


def classify_patch_rule_based(patch_pixels):
    """A hand-written rule, NOT machine learning. Looks at one cell's worth
    of pixels (a PIXELS_PER_CELL x PIXELS_PER_CELL block) and guesses terrain
    using simple brightness thresholds. This is the baseline we compare
    the trained AI classifier against."""
    mean_b = patch_pixels.mean()
    std_b = patch_pixels.std()

    if mean_b < 0.30:
        return CRATER
    if std_b > 0.13:
        return ROCK
    if mean_b > 0.63:
        return DUST
    return FREE
    # Note: this rule never predicts STEEP, because brightness alone
    # can't separate a steep slope from ordinary ground. That's a real
    # limitation of a rule-based approach, and a good thing to mention
    # when you explain why the AI classifier (Step 7) does better.


def observe_with_baseline(photo, rover_pos, view_range):
    """Slice the camera photo into per-cell patches and classify each one.
    Returns a dict {(row, col): predicted_class} for the visible area."""
    p = PIXELS_PER_CELL
    predictions = {}
    size_cells = 2 * view_range + 1
    for cell_row in range(size_cells):
        for cell_col in range(size_cells):
            patch = photo[cell_row * p:(cell_row + 1) * p, cell_col * p:(cell_col + 1) * p]
            world_row = rover_pos[0] - view_range + cell_row
            world_col = rover_pos[1] - view_range + cell_col
            predictions[(world_row, world_col)] = classify_patch_rule_based(patch)
    return predictions