import pickle
import numpy as np
from config import PIXELS_PER_CELL
from features import extract_features

with open("results/model.pkl", "rb") as f:
    _model = pickle.load(f)


def observe_with_ai(photo, rover_pos, view_range):
    """Same job as observe_with_baseline (Step 5), but classifies each
    visible cell using the trained Random Forest instead of hand-written
    rules. Returns {(row, col): predicted_class}."""
    p = PIXELS_PER_CELL
    size_cells = 2 * view_range + 1
    patches = []
    positions = []

    for cell_row in range(size_cells):
        for cell_col in range(size_cells):
            patch = photo[cell_row * p:(cell_row + 1) * p, cell_col * p:(cell_col + 1) * p]
            patches.append(extract_features(patch))
            world_row = rover_pos[0] - view_range + cell_row
            world_col = rover_pos[1] - view_range + cell_col
            positions.append((world_row, world_col))

    predictions = _model.predict(np.array(patches))
    return dict(zip(positions, predictions))