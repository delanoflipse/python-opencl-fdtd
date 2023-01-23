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

  def get_room_modes(self) -> list[float]:
    frequencies = []
    for i in range(4):
      for j in range(4):
        for k in range(4):
          nw = i / self.width
          nh = j / self.height
          nd = k / self.depth
          s_part = nw * nw + nh * nh + nd * nd
          f = (C_AIR / 2) * math.sqrt(s_part)
          if f == 0.0:
            continue
          if f > 210:
            continue
          frequencies.append(f)
    return frequencies

  def mark_regions(self) -> None:
    return None
