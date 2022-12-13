import numpy as np

from numba import njit, prange

@njit(parallel=True)
def populate_neighbours(geometry, neighbours) -> None:
  for w in prange(geometry.shape[0]):
    for h in prange(geometry.shape[1]):
      for d in prange(geometry.shape[2]):
        n = 0
        if w > 0 and geometry[w - 1, h, d] == 0:
          n |= 1 << 0
        if w < geometry.shape[0] - 1 and geometry[w + 1, h, d] == 0:
          n |= 1 << 1

        if h > 0 and geometry[w, h - 1, d] == 0:
          n |= 1 << 2
        if h < geometry.shape[1] - 1 and geometry[w, h + 1, d] == 0:
          n |= 1 << 3

        if d > 0 and geometry[w, h, d - 1] == 0:
          n |= 1 << 4
        if d < geometry.shape[2] - 1 and geometry[w, h, d + 1] == 0:
          n |= 1 << 5

        neighbours[w, h, d] = n
