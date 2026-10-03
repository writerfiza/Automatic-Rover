import numpy as np
from config import PIXELS_PER_CELL, CAMERA_RANGE
from planet import generate_planet
from camera import take_photo
from features import extract_features

# Seeds used ONLY for generating training data.
# Different seeds are reserved for testing (see below) - this is the
# "split by map" rule that avoids the model memorizing a specific map.
TRAIN_SEEDS = list(range(1000, 1030))   # 30 different training planets
TEST_SEEDS = list(range(2000, 2010))    # 10 different, unseen test planets


def collect_examples(seeds, haze_values, samples_per_map=300, rng_base=0):
    """Walk a 'virtual camera' over many random points on each map and
    record (features, true_label) pairs. The rover doesn't physically
    move here - we're just sampling lots of photo patches to build a dataset."""
    p = PIXELS_PER_CELL
    X, y = [], []
    for seed in seeds:
        terrain, elevation = generate_planet(seed)
        height, width = terrain.shape
        rng = np.random.default_rng(rng_base + seed)
        for _ in range(samples_per_map):
            row = rng.integers(0, height)
            col = rng.integers(0, width)
            haze = rng.choice(haze_values)
            photo = take_photo(terrain, elevation, (row, col), haze, rng, CAMERA_RANGE)
            # The patch for the rover's OWN cell sits at the centre of the photo
            centre = CAMERA_RANGE * p
            patch = photo[centre:centre + p, centre:centre + p]
            X.append(extract_features(patch))
            y.append(terrain[row, col])
    return np.array(X), np.array(y)


if __name__ == "__main__":
    haze_values = [0.0, 0.2, 0.4, 0.6]

    print("Collecting training data...")
    X_train, y_train = collect_examples(TRAIN_SEEDS, haze_values, samples_per_map=300)
    print("Collecting test data (unseen maps)...")
    X_test, y_test = collect_examples(TEST_SEEDS, haze_values, samples_per_map=300, rng_base=5000)

    np.savez("results/dataset.npz", X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test)

    print("Training examples:", len(y_train))
    print("Test examples:", len(y_test))
    print("\nClass counts in training set:")
    for class_id in range(5):
        print(f"  class {class_id}: {int(np.sum(y_train == class_id))}")