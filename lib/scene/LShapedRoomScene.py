from lib.scene.scene import Scene
from lib.parameters import SimulationParameters
from lib.grid import LISTENER_FLAG, SOURCE_REGION_FLAG, WALL_FLAG


class LShapedRoomScene(Scene):
  def __init__(self, parameters: SimulationParameters) -> None:
    super().__init__(parameters)
    self.height = 6.0
    self.width = 6.0
    self.depth = 6.0
    self.shape = (self.width, self.height, self.depth)

  def mark_regions(self) -> None:
    if self.grid is None:
      return
    # run_frequency = self.grid.parameters.signal_frequency

    # self.grid.fill_region(
    #     w_min=self.width / 2, d_max=self.depth/2, geometry_flag=WALL_FLAG, beta=0.0)

    # w_source = self.grid.scale(self.width / 3)
    # h_source = self.grid.scale(self.height / 2)
    # d_source = self.grid.scale(self.depth / 3)
    # self.grid.geometry[w_source, h_source, d_source] |= SOURCE_REGION_FLAG

    self.grid.fill_region(
        w_min=2.0,
        w_max=4.0,
        d_min=2.0,
        d_max=4.0, geometry_flag=WALL_FLAG, beta=0.1)

    w_source = self.grid.scale(3.0)
    h_source = self.grid.scale(self.height / 2)
    d_source = self.grid.scale(1.0)
    self.grid.geometry[w_source, h_source, d_source] |= SOURCE_REGION_FLAG

    self.grid.edge_betas.set_all(0.0)
    self.grid.fill_region(geometry_flag=LISTENER_FLAG)
