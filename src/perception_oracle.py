def observe_with_oracle(photo, rover_pos, view_range, terrain=None):
    """A stand-in for 'perfect perception' - reads the ground truth directly
    instead of looking at the photo at all. Used only as an upper-bound
    comparison, since a real system could never have this with certainty."""
    predictions = {}
    size_cells = 2 * view_range + 1
    height, width = terrain.shape
    for dr in range(-view_range, view_range + 1):
        for dc in range(-view_range, view_range + 1):
            r, c = rover_pos[0] + dr, rover_pos[1] + dc
            if 0 <= r < height and 0 <= c < width:
                predictions[(r, c)] = terrain[r, c]
    return predictions