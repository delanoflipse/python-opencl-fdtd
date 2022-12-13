import numpy as np
from lib.parameters import GRID_SHAPE


def create_grid(dtype) -> np.ndarray:
  return np.ndarray(shape=GRID_SHAPE, dtype=dtype)
