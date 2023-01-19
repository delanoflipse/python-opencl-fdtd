"""
Predefined grids besed on a number of scenarios
"""
from lib.grid import LISTENER_FLAG, SOURCE_FLAG, SOURCE_REGION_FLAG, WALL_FLAG, SimulationGrid
from lib.materials import SimulatedMaterial
from lib.parameters import SimulationParameters

wood_material = SimulatedMaterial("wood")
painted_concrete_material = SimulatedMaterial("painted_concrete")
laminate_material = SimulatedMaterial("laminate")
plaster_material = SimulatedMaterial("plaster")
glass_material = SimulatedMaterial("glass")
carpet_material = SimulatedMaterial("carpet")
cellulose_material = SimulatedMaterial("cellulose")
whiteboard_material = SimulatedMaterial("whiteboard")
metal_material = SimulatedMaterial("metal")
hard_wood_material = SimulatedMaterial("hard_wood")


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
    avg_glass_concrete = (painted_concrete_material.get_beta(
        run_frequency) + glass_material.get_beta(run_frequency)) / 2

    # set edge beta values
    self.grid.edge_betas.depth_max = painted_concrete_material.get_beta(
        run_frequency)
    self.grid.edge_betas.depth_min = painted_concrete_material.get_beta(
        run_frequency)
    self.grid.edge_betas.height_max = painted_concrete_material.get_beta(
        run_frequency)
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
    # 140x200x50
    self.grid.fill_region(
        d_min=self.depth - 2.0,
        h_max=0.5,
        w_min=1.75,
        w_max=self.width - 0.45,
        geometry_flag=WALL_FLAG,
        beta=wood_material.get_beta(run_frequency),
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


class CuboidReferenceScene(Scene):
  def __init__(self, parameters: SimulationParameters) -> None:
    super().__init__(parameters)
    self.width = self.height = self.depth = 7.0
    self.shape = (self.width, self.height, self.depth)

  def mark_regions(self) -> None:
    if self.grid is None:
      return
    # run_frequency = self.grid.parameters.signal_frequency
    w1_2_source = self.grid.scale(self.width / 2)
    h1_2_source = self.grid.scale(self.height / 2)
    d1_2_source = self.grid.scale(self.depth / 2)
    w1_4_source = self.grid.scale(self.width / 4)
    h1_4_source = self.grid.scale(self.height / 4)
    d1_4_source = self.grid.scale(self.depth / 4)
    w3_4_source = self.grid.scale(self.width * 3 / 4)
    h3_4_source = self.grid.scale(self.height * 3 / 4)
    d3_4_source = self.grid.scale(self.depth * 3 / 4)
    for w_opt in [w1_2_source, w1_4_source, w3_4_source]:
      for h_opt in [h1_2_source, h1_4_source, h3_4_source]:
        for d_opt in [d1_2_source, d1_4_source, d3_4_source]:
          self.grid.geometry[w_opt, h_opt, d_opt] |= SOURCE_REGION_FLAG

    self.grid.edge_betas.set_all(0.0)
    self.grid.fill_region(geometry_flag=LISTENER_FLAG)


class LShapedRoom(Scene):
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
    self.grid.fill_region(
        w_min=self.width / 2, d_max=self.depth/2, geometry_flag=WALL_FLAG, beta=0.0)

    w_source = self.grid.scale(self.width / 3)
    h_source = self.grid.scale(self.height / 2)
    d_source = self.grid.scale(self.depth / 3)
    self.grid.geometry[w_source, h_source, d_source] |= SOURCE_REGION_FLAG

    self.grid.edge_betas.set_all(0.0)
    self.grid.fill_region(geometry_flag=LISTENER_FLAG)


class OfficeScene(Scene):
  def __init__(self, parameters: SimulationParameters) -> None:
    super().__init__(parameters)
    self.width = 5.25
    self.height = 3.0
    self.depth = 5.06
    self.shape = (self.width, self.height, self.depth)

  def mark_regions(self) -> None:
    if self.grid is None:
      return

    run_frequency = self.grid.parameters.signal_frequency

    # set outer edge beta values
    self.grid.edge_betas.depth_max = painted_concrete_material.get_beta(
        run_frequency)
    self.grid.edge_betas.depth_min = whiteboard_material.get_beta(
        run_frequency)
    self.grid.edge_betas.height_max = cellulose_material.get_beta(
        run_frequency)
    self.grid.edge_betas.height_min = carpet_material.get_beta(run_frequency)
    self.grid.edge_betas.width_min = whiteboard_material.get_beta(
        run_frequency)
    self.grid.edge_betas.width_max = painted_concrete_material.get_beta(
        run_frequency)

    # metal closet 100x45x2000
    self.grid.fill_region(
        d_min=1.2,
        d_max=1.2 + 1,
        w_max=0.45,
        h_max=2.0,
        geometry_flag=WALL_FLAG,
        beta=metal_material.get_beta(run_frequency),
    )

    # --- inset walls of concrete and wood in width direction ---
    # desk height concrete
    self.grid.fill_region(
        d_min=self.depth - 0.29,
        h_max=0.92,
        geometry_flag=WALL_FLAG,
        beta=painted_concrete_material.get_beta(run_frequency),
    )
    # small range of wood, overwrite concrete
    self.grid.fill_region(
        d_min=self.depth - 0.54,
        h_max=0.92,
        h_min=0.92 - 0.19,
        geometry_flag=WALL_FLAG,
        beta=hard_wood_material.get_beta(run_frequency),
    )
    # upper part of concrete
    self.grid.fill_region(
        d_min=self.depth - 0.29,
        h_min=self.height - 0.2,
        geometry_flag=WALL_FLAG,
        beta=painted_concrete_material.get_beta(run_frequency),
    )
    # non-glass concrete part
    self.grid.fill_region(
        w_min=1.0,
        d_min=self.depth - 0.29,
        geometry_flag=WALL_FLAG,
        beta=painted_concrete_material.get_beta(run_frequency),
    )

    # --- inset walls of concrete and wood in depth direction ---
    # outset wood range
    self.grid.fill_region(
        w_min=self.width-0.54,
        h_min=0.92 - 0.19,
        h_max=0.92,
        d_max=self.depth-0.29,
        geometry_flag=WALL_FLAG,
        beta=hard_wood_material.get_beta(run_frequency),
    )
    # upper concrete
    self.grid.fill_region(
        w_min=self.width-0.54,
        h_min=self.height - 0.2,
        geometry_flag=WALL_FLAG,
        beta=painted_concrete_material.get_beta(run_frequency),
    )

    # below wood concrete
    self.grid.fill_region(
        w_min=self.width-0.29,
        h_max=0.92,
        geometry_flag=WALL_FLAG,
        beta=painted_concrete_material.get_beta(run_frequency),
    )

    # wall concrete 1
    self.grid.fill_region(
        w_min=self.width-0.29,
        d_max=0.085,
        geometry_flag=WALL_FLAG,
        beta=painted_concrete_material.get_beta(run_frequency),
    )

    # wall concrete 2
    self.grid.fill_region(
        w_min=self.width-0.29,
        d_min=1.085,
        d_max=2.465,
        geometry_flag=WALL_FLAG,
        beta=painted_concrete_material.get_beta(run_frequency),
    )

    # wall concrete 3
    self.grid.fill_region(
        w_min=self.width-0.29,
        d_min=3.465,
        geometry_flag=WALL_FLAG,
        beta=painted_concrete_material.get_beta(run_frequency),
    )

    # ---- SUB LOCATIONS ----
    sub_size = 0.2
    sub_offset = 0.2
    # next to door
    self.grid.fill_region(
        d_min=sub_size,
        d_max=1.2-sub_size,
        w_min=sub_size,
        w_max=0.45-sub_size,
        h_min=sub_size,
        h_max=sub_size+sub_offset,
        geometry_flag=SOURCE_REGION_FLAG,
    )
    # after closet
    self.grid.fill_region(
        d_min=2.2 - sub_size,
        d_max=self.depth - 0.5 - sub_size,
        w_min=sub_size,
        w_max=0.45-sub_size,
        h_min=sub_size,
        h_max=sub_size+sub_offset,
        geometry_flag=SOURCE_REGION_FLAG,
    )

    # depth wall, width start
    self.grid.fill_region(
        d_min=self.depth - 1.0 + sub_size,
        d_max=self.depth - 0.5 - sub_size,
        w_min=sub_size,
        w_max=1.2-sub_size,
        h_min=sub_size,
        h_max=sub_size+sub_offset,
        geometry_flag=SOURCE_REGION_FLAG,
    )
    # depth wall, width end
    self.grid.fill_region(
        d_min=self.depth - 1.0 + sub_size,
        d_max=self.depth - 0.5 - sub_size,
        w_min=self.width - 1.0 + sub_size,
        w_max=self.width - 0.5 + sub_size,
        h_min=sub_size,
        h_max=sub_size+sub_offset,
        geometry_flag=SOURCE_REGION_FLAG,
    )
    # width wall
    self.grid.fill_region(
        d_min=sub_size,
        d_max=self.depth - 0.5 - sub_size,
        w_min=self.width - 1.0 + sub_size,
        w_max=self.width - 0.5 + sub_size,
        h_min=sub_size,
        h_max=sub_size+sub_offset,
        geometry_flag=SOURCE_REGION_FLAG,
    )

    # ---- LISTENER REGIONS -----
    wall_offset = 0.7
    self.grid.fill_region(
        d_min=wall_offset,
        d_max=self.depth - wall_offset,
        w_min=wall_offset,
        w_max=self.width - wall_offset,
        h_min=1.2,
        h_max=2.0,
        geometry_flag=LISTENER_FLAG,
    )
