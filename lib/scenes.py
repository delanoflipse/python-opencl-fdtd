"""
Predefined grids besed on a number of scenarios
"""
from lib.grid import LISTENER_FLAG, SOURCE_FLAG, SOURCE_REGION_FLAG, WALL_FLAG, SimulationGrid
from lib.materials import SimulatedMaterial
from lib.parameters import SimulationParameters

wood_material = SimulatedMaterial("wood")
concrete_material = SimulatedMaterial("concrete")
laminate_material = SimulatedMaterial("laminate")
plaster_material = SimulatedMaterial("plaster")
glass_material = SimulatedMaterial("glass")


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


class ShoeboxRoomScene(Scene):
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
    avg_glass_concrete = (concrete_material.get_beta(
        run_frequency) + glass_material.get_beta(run_frequency)) / 2

    # set edge beta values
    self.grid.edge_betas.depth_max = concrete_material.get_beta(run_frequency)
    self.grid.edge_betas.depth_min = concrete_material.get_beta(run_frequency)
    self.grid.edge_betas.height_max = concrete_material.get_beta(run_frequency)
    self.grid.edge_betas.height_min = laminate_material.get_beta(run_frequency)
    self.grid.edge_betas.width_min = avg_glass_concrete
    self.grid.edge_betas.width_max = plaster_material.get_beta(run_frequency)

    # --- Blocking objects ---
    # closet 1
    # 100x60x200
    self.grid.fill_region(
        d_min=0.9,  # 90cm from door
        d_max=1.9,  # 1m wide
        h_max=2.0,  # 2M high
        w_min=self.width - 0.6,  # 60cm deep, against wall
        geometry_flag=WALL_FLAG,
        beta=wood_material.get_beta(run_frequency),
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
        beta=wood_material.get_beta(run_frequency),
    )

    # bed
    # 39.5x147x77
    self.grid.fill_region(
        d_min=0.04,
        d_max=0.43,
        h_max=0.77,
        w_min=1.1,
        w_max=1.1 + 1.47,
        geometry_flag=WALL_FLAG,
        beta=wood_material.get_beta(run_frequency),
    )

    # --- SOURCE LOCATIONS ---
    speaker_height = self.grid.scale(.97)
    speaker_offset = self.grid.scale(.1)

    # speaker_locations
    # on closet 2
    self.grid.fill_region(
        d_min=0.04,
        d_max=0.43,
        h_min=speaker_height,
        h_max=speaker_height + speaker_offset,
        w_min=1.1,
        w_max=1.1 + 1.47,
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


class BellBoxScene(Scene):
  def __init__(self, parameters: SimulationParameters, has_wall: bool = True) -> None:
    super().__init__(parameters)
    self.width = 2.19
    self.height = 2.42
    self.depth = 3.03
    self.has_wall = has_wall
    self.shape = (self.width, self.height, self.depth)

  def mark_regions(self) -> None:
    if self.grid is None:
      return
    run_frequency = self.grid.parameters.signal_frequency

    # wall
    if self.has_wall:
      self.grid.fill_region(
          d_min=1.91,
          d_max=1.91+0.05,
          w_max=1.26,
          geometry_flag=WALL_FLAG,
          beta=wood_material.get_beta(run_frequency),
      )

    w_source = self.grid.scale(self.width-1.45)
    h_source = self.grid.scale(1.35)
    d_source = self.grid.scale(self.depth-0.59)
    self.grid.geometry[w_source, h_source, d_source] |= SOURCE_REGION_FLAG
    self.grid.fill_region(
        d_min=0.2,
        w_min=0.2,
        w_max=self.width - 0.2,
        h_min=0.2,
        h_max=self.height - 0.2,
        geometry_flag=LISTENER_FLAG,
    )


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
    # run_frequency = self.grid.parameters.signal_frequency
    w_source = self.grid.scale(self.width-1.45)
    h_source = self.grid.scale(1.35)
    d_source = self.grid.scale(self.depth-0.59)
    self.grid.geometry[w_source, h_source, d_source] |= SOURCE_REGION_FLAG
