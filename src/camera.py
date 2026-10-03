import numpy as np
from config import (
    CELL_SIZE_M, PIXELS_PER_CELL, CAMERA_RANGE,
    BASE_BRIGHTNESS, TEXTURE_NOISE, SENSOR_NOISE, HAZE_BRIGHTNESS,
)

# Turn the dictionaries into arrays so terrain numbers can index them
BASE_LOOKUP = np.array([BASE_BRIGHTNESS[k] for k in range(5)])
TEXTURE_LOOKUP = np.array([TEXTURE_NOISE[k] for k in range(5)])


def lighting_factor(elevation):
    """Sun from the west: slopes facing west are brighter, others darker."""
    d_row, d_col = np.gradient(elevation, CELL_SIZE_M)
    return np.clip(1.0 + 1.5 * d_col, 0.6, 1.4)


def enlarge(cell_grid):
    """Turn each map cell into a block of pixels."""
    p = PIXELS_PER_CELL
    return np.repeat(np.repeat(cell_grid, p, axis=0), p, axis=1)


def render_full_image(terrain, elevation, rng):
    """Draw the whole planet as brightness values (before haze)."""
    brightness = BASE_LOOKUP[terrain] * lighting_factor(elevation)
    texture_size = TEXTURE_LOOKUP[terrain]
    brightness_px = enlarge(brightness)
    texture_px = enlarge(texture_size)
    return brightness_px + rng.normal(0, 1, brightness_px.shape) * texture_px


def take_photo(terrain, elevation, position, haze, rng, view_range=CAMERA_RANGE):
    """The rover's camera. Returns a square grayscale image, values 0 to 1.

    position: (row, col) of the rover
    haze: 0.0 (clear air) up to 1.0 (heavy dust)
    """
    p = PIXELS_PER_CELL
    full = render_full_image(terrain, elevation, rng)

    # Pad with black so the camera can look past the edge of the map
    padded = np.pad(full, view_range * p, constant_values=0.0)
    size = (2 * view_range + 1) * p
    top = position[0] * p
    left = position[1] * p
    window = padded[top:top + size, left:left + size]

    # Haze: distant pixels are washed toward grey more than close ones
    ys, xs = np.mgrid[0:size, 0:size]
    centre = size / 2
    dist_cells = np.hypot(ys + 0.5 - centre, xs + 0.5 - centre) / p
    weight = 1 - np.exp(-2.0 * haze * dist_cells / view_range)
    photo = window * (1 - weight) + HAZE_BRIGHTNESS * weight

    # Camera sensor noise, then keep values inside 0 to 1
    photo = photo + rng.normal(0, SENSOR_NOISE, photo.shape)
    return np.clip(photo, 0.0, 1.0)