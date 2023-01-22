"""
Predefined grids besed on a number of scenarios
"""
from lib.grid import SimulationGrid
from lib.parameters import SimulationParameters



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

  def mark_regions(self) -> None:
    return None
