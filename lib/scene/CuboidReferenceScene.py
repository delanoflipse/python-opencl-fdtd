from lib.scene.scene import Scene
from lib.parameters import SimulationParameters
from lib.grid import LISTENER_FLAG, SOURCE_REGION_FLAG


class CuboidReferenceScene(Scene):
  def __init__(self, parameters: SimulationParameters) -> None:
    super().__init__(parameters)
    self.width = self.height = self.depth = 7.0
    self.shape = (self.width, self.height, self.depth)

  def mark_regions(self) -> None:
    if self.grid is None:
      return

    # run_frequency = self.grid.parameters.signal_frequency
    # w1_2_source = self.grid.scale(self.width / 2)
    # h1_2_source = self.grid.scale(self.height / 2)
    # d1_2_source = self.grid.scale(self.depth / 2)
    # w1_4_source = self.grid.scale(self.width / 4)
    # h1_4_source = self.grid.scale(self.height / 4)
    # d1_4_source = self.grid.scale(self.depth / 4)
    # w3_4_source = self.grid.scale(self.width * 3 / 4)
    # h3_4_source = self.grid.scale(self.height * 3 / 4)
    # d3_4_source = self.grid.scale(self.depth * 3 / 4)
    # for w_opt in [w1_2_source, w1_4_source, w3_4_source]:
    #   for h_opt in [h1_2_source, h1_4_source, h3_4_source]:
    #     for d_opt in [d1_2_source, d1_4_source, d3_4_source]:
    #       self.grid.geometry[w_opt, h_opt, d_opt] |= SOURCE_REGION_FLAG

    self.grid.edge_betas.set_all(0.1)
    sub_pos = self.grid.pos(0.3, 0.15, 0.15)
    list_pos = self.grid.pos(1.33, 1.0, 1.38)
    self.grid.geometry[sub_pos] |= SOURCE_REGION_FLAG
    self.grid.geometry[list_pos] |= LISTENER_FLAG
