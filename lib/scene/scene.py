"""
Predefined grids besed on a number of scenarios
"""
import math
from lib.grid import SimulationGrid
from lib.parameters import SimulationParameters
from lib.physical_constants import C_AIR


class Scene:
  def __init__(self, parameters: SimulationParameters) -> None:
    self.parameters = parameters
    self.width = 0.0
    self.height = 0.0
    self.depth = 0.0
    self.shape = (self.width, self.height, self.depth)
    self.grid: SimulationGrid = None

  def build(self) -> SimulationGrid:
    self.grid = SimulationGrid(self.shape, self.parameters)
    self.mark_regions()
    self.grid.build()
    return self.grid

  def rebuild(self) -> None:
    if self.grid is None:
      raise Exception("Grid is set. Please call build before calling rebuild!")
    self.mark_regions()
    self.grid.rebuild()
    return None

  def get_room_modes(self) -> list[tuple[float, int]]:
    frequencies = []
    for i in range(4):
      i_active = 1 if i > 0 else 0
      for j in range(4):
        j_active = 1 if j > 0 else 0
        for k in range(4):
          k_active = 1 if k > 0 else 0
          nw = i / self.width
          nh = j / self.height
          nd = k / self.depth
          axis_type = i_active + j_active + k_active
          if axis_type == 0:
            continue
          s_part = nw * nw + nh * nh + nd * nd
          f = (C_AIR / 2) * math.sqrt(s_part)
          if f == 0.0:
            continue
          frequencies.append((f, axis_type))
    return frequencies

  def mark_regions(self) -> None:
    return None
