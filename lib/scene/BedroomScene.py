from lib.scene.scene import Scene
from lib.materials import wood_material, painted_concrete_material, glass_material, laminate_material, plaster_material
from lib.parameters import SimulationParameters
from lib.grid import LISTENER_FLAG, SOURCE_REGION_FLAG, WALL_FLAG


class BedroomScene(Scene):
  def __init__(self, parameters: SimulationParameters) -> None:
    super().__init__(parameters)
    self.width = 3.6
    self.height = 2.6
    self.depth = 4.2
    self.shape = (self.width, self.height, self.depth)

  def mark_regions(self) -> None:
    if self.grid is None:
      return
    run_frequency = self.grid.parameters.signal_frequency

    painted_concrete = painted_concrete_material.get_beta(run_frequency)
    glass = glass_material.get_beta(run_frequency)
    laminate = laminate_material.get_beta(run_frequency)
    plaster = plaster_material.get_beta(run_frequency)
    wood = wood_material.get_beta(run_frequency)

    avg_glass_concrete = (painted_concrete + glass) / 2

    # set edge beta values
    self.grid.edge_betas.depth_max = painted_concrete
    self.grid.edge_betas.depth_min = painted_concrete
    self.grid.edge_betas.height_max = painted_concrete
    self.grid.edge_betas.height_min = laminate
    self.grid.edge_betas.width_min = avg_glass_concrete
    self.grid.edge_betas.width_max = plaster

    # --- Blocking objects ---
    # closet 1
    # 100x60x200
    self.grid.fill_region(
        d_min=0.9,  # 90cm from door
        d_max=1.9,  # 1m wide
        h_max=2.0,  # 2M high
        w_min=self.width - 0.6,  # 60cm deep, against wall
        geometry_flag=WALL_FLAG,
        beta=wood,
    )

    # closet 2
    # 39.5x147x77
    self.grid.fill_region(
        d_min=0.04,
        d_max=0.43,
        h_max=0.77,
        w_min=1.1,
        w_max=1.1 + 1.47,
        geometry_flag=WALL_FLAG,
        beta=wood,
    )
    # bed
    # 140x200x50
    self.grid.fill_region(
        d_min=self.depth - 2.0,
        h_max=0.5,
        w_min=1.75,
        w_max=self.width - 0.45,
        geometry_flag=WALL_FLAG,
        beta=wood,
    )

    # --- SOURCE LOCATIONS ---
    # speaker_locations
    # on closet 2
    self.grid.fill_region(
        d_min=0.12,
        d_max=0.38,
        h_min=.97,
        h_max=.97 + 0.1,
        w_min=1.1,
        w_max=1.1 + 1.47,
        geometry_flag=SOURCE_REGION_FLAG,
    )

    # on the floor, left
    self.grid.fill_region(
        d_min=0.12,
        d_max=1.35,
        h_min=.2,
        h_max=.2 + 0.1,
        w_min=0.04,
        w_max=0.3,
        geometry_flag=SOURCE_REGION_FLAG,
    )

    # on the floor, right
    self.grid.fill_region(
        d_min=2.1,
        d_max=2.6,
        h_min=.2,
        h_max=.2 + 0.1,
        w_min=self.width - 0.5,
        geometry_flag=SOURCE_REGION_FLAG,
    )

    # OR single source
    # self.grid.geometry[self.grid.scale(1.83), speaker_height,
    #               self.grid.scale(.2)] |= SOURCE_REGION_FLAG

    # --- LISTENER LOCATIONS ---
    # sit/stand
    self.grid.fill_region(
        d_min=0.6,
        d_max=self.depth - 0.8,
        h_min=1.25,
        h_max=1.9,
        w_min=1.14,
        w_max=self.width - 0.6,
        geometry_flag=LISTENER_FLAG,
    )
    # bed
    self.grid.fill_region(
        d_min=self.depth - 2.0,
        h_min=0.8,
        h_max=1.3,
        w_min=1.75,
        w_max=self.width - 0.45,
        geometry_flag=LISTENER_FLAG,
    )
