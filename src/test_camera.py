import numpy as np
from planet import generate_planet
from camera import take_photo
from config import CAMERA_RANGE, PIXELS_PER_CELL

terrain, elevation = generate_planet(2026)
expected_size = (2 * CAMERA_RANGE + 1) * PIXELS_PER_CELL

# Test 1: the photo has the right shape and valid brightness values
photo = take_photo(terrain, elevation, (15, 20), 0.0, np.random.default_rng(1))
assert photo.shape == (expected_size, expected_size)
assert photo.min() >= 0.0 and photo.max() <= 1.0

# Test 2: the same random seed gives exactly the same photo
photo_again = take_photo(terrain, elevation, (15, 20), 0.0, np.random.default_rng(1))
assert np.array_equal(photo, photo_again)

# Test 3: heavy haze lowers the contrast
hazy = take_photo(terrain, elevation, (15, 20), 1.0, np.random.default_rng(1))
assert hazy.std() < photo.std()

# Test 4: the camera works at the map corner (looks off the edge)
corner = take_photo(terrain, elevation, (0, 0), 0.0, np.random.default_rng(1))
assert corner.shape == (expected_size, expected_size)

print("All camera tests passed")