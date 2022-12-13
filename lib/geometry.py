
import numpy as np

from numba import njit, prange

@njit(parallel=True)
def populate_neighbours(geometry, neighbours):
  for w in prange(geometry.shape[0]):
    for h in prange(geometry.shape[1]):
      for d in prange(geometry.shape[2]):
        n = 6
        if w == 0 or geometry[w-1,h,d] == 1:
          n -= 1
        if w == geometry.shape[0] - 1 or geometry[w+1,h,d] == 1:
          n -= 1
        if h == 0 or geometry[w,h-1,d] == 1:
          n -= 1
        if h == geometry.shape[1] - 1 or geometry[w,h+1,d] == 1:
          n -= 1
        if d == 0 or geometry[w,h,d-2] == 1:
          n -= 1
        if d == geometry.shape[2] - 1 or geometry[w,h,d+1] == 1:
          n -= 1
        neighbours[w,h,d] = n
