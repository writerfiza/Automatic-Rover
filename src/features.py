import numpy as np


def extract_features(patch_pixels):
    """Turn one cell's camera patch (a small 2D array of brightness values)
    into a short list of numbers the model can learn from.

    These five numbers are simple, human-understandable image statistics -
    not deep learning features. That keeps the model explainable, which is
    good for a student project: you can explain exactly what it's looking at."""
    mean_b = patch_pixels.mean()
    std_b = patch_pixels.std()
    min_b = patch_pixels.min()
    max_b = patch_pixels.max()
    contrast = max_b - min_b
    return np.array([mean_b, std_b, min_b, max_b, contrast])


FEATURE_NAMES = ["mean_brightness", "std_brightness", "min_brightness",
                  "max_brightness", "contrast"]