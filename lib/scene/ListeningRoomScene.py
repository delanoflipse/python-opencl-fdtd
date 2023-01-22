from lib.scenes import Scene, wood_material, carpet_material, glass_material, laminate_material
from lib.parameters import SimulationParameters
from lib.grid import LISTENER_FLAG, SOURCE_REGION_FLAG, WALL_FLAG


class ListeningRoomScene(Scene):
  def __init__(self, parameters: SimulationParameters) -> None:
    super().__init__(parameters)
    self.width = 2.9
    self.height = 2.1
    self.depth = 4.3
    self.shape = (self.width, self.height, self.depth)

  def mark_regions(self) -> None:
    if self.grid is None:
      return
    run_frequency = self.grid.parameters.signal_frequency

    wood = 0.12
    carpet = 0.12
    laminate = 0.12

    # wood = 0.01
    # carpet = 0.01
    # laminate = 0.01

    wood = wood_material.get_beta(run_frequency)
    carpet = carpet_material.get_beta(run_frequency)
    glass = glass_material.get_beta(run_frequency)
    laminate = laminate_material.get_beta(run_frequency)
    nopper = 0.5
    # nopper = 0.01

    # set edge beta values
    self.grid.edge_betas.depth_max = carpet
    self.grid.edge_betas.depth_min = carpet
    self.grid.edge_betas.height_max = nopper
    self.grid.edge_betas.height_min = laminate
    self.grid.edge_betas.width_min = carpet
    self.grid.edge_betas.width_max = carpet

    # closet lower part
    self.grid.fill_region(
        w_min=self.width - 0.5,
        d_min=0.45,
        d_max=0.45+2.1,
        h_max=1.3,
        geometry_flag=WALL_FLAG,
        beta=wood,
    )

    # closet higher part
    self.grid.fill_region(
        w_min=self.width - 0.5,
        d_min=0.45,
        d_max=0.45+0.6,
        geometry_flag=WALL_FLAG,
        beta=wood,
    )

    # SUB location
    spacing = 0.05
    self.grid.fill_region(
        w_min=self.width - 1.9 + 0.4 - spacing,
        w_max=self.width - 1.9 + 0.4 + spacing,
        d_min=self.depth - 0.22 + spacing,
        d_max=self.depth - 0.22 - spacing,
        h_max=0.2,
        h_min=0.2,
        geometry_flag=SOURCE_REGION_FLAG,
    )

    # Listener
    spacing = 0.4
    # (w, h, d) = self.grid.pos(self.width / 2, self.height / 2, self.depth / 2)
    # self.grid.geometry[w, h, d] |= LISTENER_FLAG
    self.grid.fill_region(
        w_min=spacing,
        w_max=self.width - spacing,
        d_min=spacing,
        d_max=self.width - spacing,
        h_max=1.5,
        h_min=1.9,
        geometry_flag=LISTENER_FLAG,
    )
