from lib.scene.scene import Scene
from lib.materials import wood_material
from lib.parameters import SimulationParameters
from lib.grid import LISTENER_FLAG, SOURCE_REGION_FLAG, WALL_FLAG


class BellBoxScene(Scene):
  def __init__(self, parameters: SimulationParameters) -> None:
    super().__init__(parameters)
    self.width = 2.19
    self.height = 2.42
    self.depth = 3.03
    self.shape = (self.width, self.height, self.depth)

  def mark_regions(self) -> None:
    if self.grid is None:
      return

    run_frequency = self.grid.parameters.signal_frequency
    wood = wood_material.get_beta(run_frequency)

    # wall
    self.grid.fill_region(
        d_min=1.91,
        d_max=1.91+0.05,
        w_max=1.26,
        geometry_flag=WALL_FLAG,
        beta=wood,
    )

    source_pos = self.grid.pos(self.width-1.45, 1.35, self.depth-0.59)
    self.grid.geometry[source_pos] |= SOURCE_REGION_FLAG

    self.grid.fill_region(
        d_min=0.2,
        w_min=0.2,
        w_max=self.width - 0.2,
        h_min=0.2,
        h_max=self.height - 0.2,
        geometry_flag=LISTENER_FLAG,
    )
