import numpy as np

from numba import njit, prange

BIT_0 = 1 << 0
BIT_1 = 1 << 1
BIT_2 = 1 << 2
BIT_3 = 1 << 3
BIT_4 = 1 << 4
BIT_5 = 1 << 5


@njit(parallel=True)
# set neighbour flags for given geometry
def populate_neighbours(geometry, neighbours) -> None:
  for w in prange(geometry.shape[0]):
    for h in prange(geometry.shape[1]):
      for d in prange(geometry.shape[2]):
        neighour_flags = 0
        if w > 0 and geometry[w - 1, h, d] != 1:
          neighour_flags |= BIT_0
        if w < geometry.shape[0] - 1 and geometry[w + 1, h, d] != 1:
          neighour_flags |= BIT_1

        if h > 0 and geometry[w, h - 1, d] != 1:
          neighour_flags |= BIT_2
        if h < geometry.shape[1] - 1 and geometry[w, h + 1, d] != 1:
          neighour_flags |= BIT_3

        if d > 0 and geometry[w, h, d - 1] != 1:
          neighour_flags |= BIT_4
        if d < geometry.shape[2] - 1 and geometry[w, h, d + 1] != 1:
          neighour_flags |= BIT_5

        neighbours[w, h, d] = neighour_flags
