import numpy as np


def max_min_norm(array: np.ndarray):
    return (array - array.min()) / (array.max() - array.min())


def image_norm(img):
    return (max_min_norm(img) * 255).round(0).astype(np.uint8)
