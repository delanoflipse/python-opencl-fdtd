from lib.scene.scene import Scene
from lib.parameters import SimulationParameters
from lib.grid import SOURCE_REGION_FLAG

class ConcertHallScene(Scene):
  def __init__(self, parameters: SimulationParameters) -> None:
    super().__init__(parameters)
    self.width = 40.0
    self.height = 8.0
    self.depth = 65.0
    self.shape = (self.width, self.height, self.depth)

  def mark_regions(self) -> None:
    if self.grid is None:
      return
    # TODO: extend this scene
    # run_frequency = self.grid.parameters.signal_frequency
    w_source = self.grid.scale(self.width-1.45)
    h_source = self.grid.scale(1.35)
    d_source = self.grid.scale(self.depth-0.59)
    self.grid.geometry[w_source, h_source, d_source] |= SOURCE_REGION_FLAG

