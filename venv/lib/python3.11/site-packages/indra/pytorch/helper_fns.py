import numpy as np
from deeplake.util.iterable_ordered_dict import IterableOrderedDict

# helper functions used in tests


def make_array_fixed_size(arr: np.ndarray):
    H, W, C = arr.shape
    if H > 500:
        arr = arr[:500, :, :]
    elif H < 500:
        arr = np.pad(arr, ((0, 500 - H), (0, 0), (0, 0)), "constant")
    if W > 500:
        arr = arr[:, :500, :]
    elif W < 500:
        arr = np.pad(arr, ((0, 0), (0, 500 - W), (0, 0)), "constant")
    if C > 3:
        arr = arr[:, :, :3]
    if C < 3:
        arr = np.pad(arr, ((0, 0), (0, 0), (0, 3 - C)), "constant")
    return arr


def transform_fn(sample):
    return IterableOrderedDict(
        {k: make_array_fixed_size(v) if k == "images" else v for k, v in sample.items()}
    )
