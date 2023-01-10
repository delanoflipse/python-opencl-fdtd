import math
import sys
import numpy as np

from numba import njit, prange, float64
from numba.experimental import jitclass

from lib.parameters import SimulationParameters

BIT_0 = 1 << 0
BIT_1 = 1 << 1
BIT_2 = 1 << 2
BIT_3 = 1 << 3
BIT_4 = 1 << 4
BIT_5 = 1 << 5

WALL_FLAG = BIT_0
SOURCE_FLAG = BIT_1
SOURCE_REGION_FLAG = BIT_2
LISTENER_FLAG = BIT_3

BASE_BETA = 0.01


def clamp(value, min_value, max_value):
  return max(min_value, min(max_value, value))


grid_edge_spec = [
    ('height_min', float64),
    ('height_max', float64),
    ('depth_min', float64),
    ('depth_max', float64),
    ('width_min', float64),
    ('width_max', float64),
]


@jitclass(grid_edge_spec)
class GridEdgeBeta:
  """Absorption rate for all walls"""

  def __init__(self) -> None:
    self.height_min = BASE_BETA
    self.height_max = BASE_BETA
    self.depth_min = BASE_BETA
    self.depth_max = BASE_BETA
    self.width_min = BASE_BETA
    self.width_max = BASE_BETA

  def set_all(self, value: float) -> None:
    self.height_min = value
    self.height_max = value
    self.depth_min = value
    self.depth_max = value
    self.width_min = value
    self.width_max = value


class SimulationGrid:
  """All data related to the grid of the simulation"""

  def __init__(self, shape: tuple[float, float, float], parameters: SimulationParameters):
    (width, height, depth) = shape
    self.parameters = parameters
    self.real_shape = shape
    self.width_parts = int(round(width / parameters.dx))
    self.height_parts = int(round(height / parameters.dx))
    self.depth_parts = int(round(depth / parameters.dx))
    self.grid_size = self.width_parts * self.height_parts * self.depth_parts
    self.grid_shape = (self.width_parts, self.height_parts, self.depth_parts)

    self.edge_betas = GridEdgeBeta()

    self.geometry = self.create_grid("int8")
    self.neighbours = self.create_grid("int8")

    self.pressure = self.create_grid("float64")
    self.pressure_previous = self.create_grid("float64")
    self.pressure_next = self.create_grid("float64")

    self.analysis_keys = {
        'PRESSURE': 0,
        'RMS': 1,
        'LEQ': 2,
        'EWMA': 3,
        'EWMA_L': 4,
        '_TEST': 5,
    }
    self.analysis_values = len(self.analysis_keys)
    # Leq, RMS, ERMS, ...
    self.analysis_shape = (self.width_parts, self.height_parts,
                           self.depth_parts, self.analysis_values)
    self.analysis = np.zeros(shape=self.analysis_shape, dtype="float64")
    self.beta = self.create_grid("float64")

    float64_buffers = 5 + self.analysis_values
    int8_buffers = 2
    self.storage_estimate = self.grid_size * (float64_buffers*8 + int8_buffers)
    self.source_count = -1
    self.source_index = -1
    self.is_build = False

  def get_storage_str(self) -> str:
    suffix = "B"
    num = self.storage_estimate
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
      if abs(num) < 1024.0:
        return f"{num:3.1f}{unit}{suffix}"
      num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

  def reset_values(self) -> None:
    self.pressure.fill(0.0)
    self.pressure_previous.fill(0.0)
    self.pressure_next.fill(0.0)
    self.analysis.fill(0.0)

  def fill_region(self, w_min=0.0, w_max=float("inf"), h_min=0.0, h_max=float("inf"), d_min=0.0, d_max=float("inf"), geometry_flag=WALL_FLAG, beta=0.5) -> None:
    d_min_int = clamp(self.scale(d_min), 0, self.depth_parts - 1)
    d_max_int = clamp(self.scale(d_max), d_min_int + 1, self.depth_parts)
    h_min_int = clamp(self.scale(h_min), 0, self.height_parts - 1)
    h_max_int = clamp(self.scale(h_max), h_min_int + 1, self.height_parts)
    w_min_int = clamp(self.scale(w_min), 0, self.width_parts - 1)
    w_max_int = clamp(self.scale(w_max), w_min_int + 1, self.width_parts)

    set_beta = geometry_flag & WALL_FLAG > 0
    for d in range(d_min_int, d_max_int):
      for h in range(h_min_int, h_max_int):
        for w in range(w_min_int, w_max_int):
          self.geometry[w, h, d] |= geometry_flag
          if set_beta:
            self.beta[w, h, d] = beta

  def create_grid(self, dtype) -> np.ndarray:
    """Generalised method to create new nd arrays"""
    return np.zeros(shape=self.grid_shape, dtype=dtype)

  def scale(self, size: float) -> int:
    """Get grid position for real measurement"""
    if math.isinf(size):
      return sys.maxsize
    return int(round(size / self.parameters.dx))

  def build(self) -> None:
    """Once flags are set, build the geometry"""
    populate_neighbours(self.geometry, self.neighbours)
    populate_inner_beta(self.geometry, self.beta, self.edge_betas)
    self.source_count = count_source_locations(self.geometry)
    self.source_index = 0
    set_nth_source_on(self.geometry, self.source_index)
    self.is_build = True

  def next_source(self) -> None:
    # TODO: check overflow
    self.source_index += 1
    set_nth_source_on(self.geometry, self.source_index)

  def rebuild(self) -> None:
    """rebuild the geometry"""
    # TODO
    # populate_inner_beta(self.geometry, self.beta, self.edge_betas)


@njit
def set_nth_source_on(geometry: np.ndarray, index: int, unset=True) -> None:
  """Set the nth source region cell to be the current source"""
  current_index = -1
  # TODO: return location tuple?
  for w in prange(geometry.shape[0]):
    for h in prange(geometry.shape[1]):
      for d in prange(geometry.shape[2]):
        if unset:
          geometry[w, h, d] &= ~SOURCE_FLAG
        if geometry[w, h, d] & SOURCE_REGION_FLAG > 0:
          current_index += 1
          if current_index == index:
            geometry[w, h, d] |= SOURCE_FLAG
  return


@njit(parallel=True)
def count_source_locations(geometry: np.ndarray) -> int:
  """Count the number of cells that have the SOURCE_REGION_FLAG set"""
  count = 0
  for w in prange(geometry.shape[0]):
    for h in prange(geometry.shape[1]):
      for d in prange(geometry.shape[2]):
        if geometry[w, h, d] & SOURCE_REGION_FLAG > 0:
          count += 1
  return count


@njit(parallel=True)
def populate_neighbours(geometry: np.ndarray, neighbours: np.ndarray) -> None:
  """Set neighbour flags for geometry"""
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


@njit(parallel=True)
def populate_inner_beta(geometry: np.ndarray, beta: np.ndarray, edge_betas: GridEdgeBeta) -> None:
  """Set neighbour flags for geometry"""
  for w in prange(geometry.shape[0]):
    for h in prange(geometry.shape[1]):
      for d in prange(geometry.shape[2]):
        reflection_count = 0
        beta_average = 0
        # w min
        if w == 0:
          beta_average += edge_betas.width_min
          reflection_count += 1
        elif geometry[w - 1, h, d] & WALL_FLAG > 0:
          beta_average += beta[w - 1, h, d]
          reflection_count += 1

        # w max
        if w == geometry.shape[0] - 1:
          beta_average += edge_betas.width_max
          reflection_count += 1
        elif geometry[w + 1, h, d] & WALL_FLAG > 0:
          beta_average += beta[w + 1, h, d]
          reflection_count += 1

        # h min
        if h == 0:
          beta_average += edge_betas.height_min
          reflection_count += 1
        elif geometry[w, h - 1, d] & WALL_FLAG > 0:
          beta_average += beta[w, h - 1, d]
          reflection_count += 1

        # h max
        if h == geometry.shape[1] - 1:
          beta_average += edge_betas.height_max
          reflection_count += 1
        elif geometry[w, h + 1, d] & WALL_FLAG > 0:
          beta_average += beta[w, h + 1, d]
          reflection_count += 1

        # d min
        if d == 0:
          beta_average += edge_betas.depth_max
          reflection_count += 1
        elif geometry[w, h, d - 1] & WALL_FLAG > 0:
          beta_average += beta[w, h, d - 1]
          reflection_count += 1

        # d max
        if d == geometry.shape[2] - 1:
          beta_average += edge_betas.depth_min
          reflection_count += 1
        elif geometry[w, h, d + 1] & WALL_FLAG > 0:
          beta_average += beta[w, h, d + 1]
          reflection_count += 1

        if reflection_count > 0:
          beta[w, h, d] = beta_average / reflection_count
