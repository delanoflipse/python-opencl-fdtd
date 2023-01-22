from lib.scene.scene import Scene
from lib.parameters import SimulationParameters
from lib.grid import LISTENER_FLAG, SOURCE_REGION_FLAG, WALL_FLAG


class ShoeboxReferenceScene(Scene):
  def __init__(self, parameters: SimulationParameters) -> None:
    super().__init__(parameters)
    self.width = 5.5
    self.height = 2.8
    self.depth = 4.0
    self.shape = (self.width, self.height, self.depth)

  def mark_regions(self) -> None:
    if self.grid is None:
      return

    self.grid.edge_betas.set_all(0.1)
    self.grid.edge_betas.height_min = 0.05

    sub_pos = self.grid.pos(0.3, 0.3, 0.3)
    self.grid.geometry[sub_pos] |= SOURCE_REGION_FLAG

    list_pos = self.grid.pos(1.33, 1.0, 1.38)
    self.grid.geometry[list_pos] |= LISTENER_FLAG
