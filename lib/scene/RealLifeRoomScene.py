from lib.scene.scene import Scene
from lib.materials import wood_material, carpet_material, double_glass_material, metal_material, suspended_ceiling_material, hard_wall_material, cushion_material
from lib.parameters import SimulationParameters
from lib.grid import LISTENER_FLAG, SOURCE_REGION_FLAG, WALL_FLAG


class RealLifeRoomScene(Scene):
  def __init__(self, parameters: SimulationParameters, reference_values=False) -> None:
    super().__init__(parameters)
    self.width = 4.4
    self.height = 2.7
    self.depth = 3.32
    self.reference_values = reference_values
    self.shape = (self.width, self.height, self.depth)

  def mark_regions(self) -> None:
    if self.grid is None:
      return

    run_frequency = self.grid.parameters.signal_frequency

    carpet = carpet_material.get_beta(run_frequency)
    glass = double_glass_material.get_beta(run_frequency)
    suspended_ceiling = suspended_ceiling_material.get_beta(run_frequency)
    wood = wood_material.get_beta(run_frequency)
    hard_wall = hard_wall_material.get_beta(run_frequency)
    cushion = cushion_material.get_beta(run_frequency)
    metal = metal_material.get_beta(run_frequency)
    tv_screen = 0.1
    whiteboard = 0.1

    # set edge beta values
    if self.reference_values:
      self.grid.edge_betas.depth_min = 0.5
      self.grid.edge_betas.depth_max = 0.5
      self.grid.edge_betas.width_min = 0.5
      self.grid.edge_betas.width_max = 0.5
      self.grid.edge_betas.height_max = 0.5
      self.grid.edge_betas.height_min = 0.5
    else:
      self.grid.edge_betas.height_max = suspended_ceiling
      self.grid.edge_betas.height_min = carpet
      self.grid.edge_betas.depth_min = glass
      self.grid.edge_betas.depth_max = hard_wall
      self.grid.edge_betas.width_min = hard_wall
      self.grid.edge_betas.width_max = hard_wall

    if not self.reference_values:
      #  ---- OBJECTS -----
      # TV
      self.grid.fill_region(
          d_min=1.05,
          d_max=1.05 + 1.24,
          w_min=0.07,
          w_max=0.11,
          h_min=1.09,
          h_max=1.09 + 0.71,
          geometry_flag=WALL_FLAG,
          beta=tv_screen,
      )
      # Radiator
      self.grid.fill_region(
          d_min=self.depth - 0.77,
          d_max=self.depth - 0.2,
          w_min=0.03,
          w_max=0.10,
          h_min=0.15,
          h_max=0.15 + 0.7,
          geometry_flag=WALL_FLAG,
          beta=metal,
      )
      # whiteboards
      self.grid.fill_region(
          d_min=self.depth - 1.7,
          w_min=self.width - 0.02,
          h_min=0.145,
          geometry_flag=WALL_FLAG,
          beta=whiteboard,
      )

      # wood below whiteboard
      self.grid.fill_region(
          d_min=1.05,
          w_min=self.width - 0.03,
          h_max=0.145,
          geometry_flag=WALL_FLAG,
          beta=wood,
      )

      # Poof 1
      self.grid.fill_region(
          d_min=self.depth / 2 - 0.23,
          d_max=self.depth / 2 + 0.23,
          h_max=0.46,
          w_max=0.46,
          geometry_flag=WALL_FLAG,
          beta=cushion,
      )

      # Poof 2
      self.grid.fill_region(
          d_min=self.depth - 0.46,
          h_max=0.46,
          w_min=1.4,
          w_max=1.4 + 0.46,
          geometry_flag=WALL_FLAG,
          beta=cushion,
      )

      # couch, sitting area
      self.grid.fill_region(
          d_min=0.1,
          d_max=0.1 + 0.85,
          h_min=0.18,
          h_max=0.18+0.22,
          w_min=1.06,
          w_max=1.06 + 2.35,
          geometry_flag=WALL_FLAG,
          beta=cushion,
      )

      # couch, back rest
      self.grid.fill_region(
          d_min=0.1,
          d_max=0.1 + 0.26,
          h_min=0.18+0.22,
          h_max=0.18+0.22+0.3,
          w_min=1.06 + 0.25,
          w_max=1.06 + 2.35 - 0.48,
          geometry_flag=WALL_FLAG,
          beta=cushion,
      )

      # couch, extra cushion
      self.grid.fill_region(
          d_min=0.1,
          d_max=0.1 + 0.26,
          h_min=0.18+0.22,
          h_max=0.18+0.22+0.15,
          w_min=1.06 + 2.35 - 0.48,
          w_max=1.06 + 2.35,
          geometry_flag=WALL_FLAG,
          beta=cushion,
      )

    if self.reference_values:
      sub_pos = self.grid.pos(1.06, 0.22, self.depth / 2)
      self.grid.geometry[sub_pos] |= SOURCE_REGION_FLAG
      from_wall = max(self.parameters.dx * 2, 0.03)
      mic_pos = self.grid.pos(self.width - from_wall, 1.77, self.depth - 0.18)
      self.grid.geometry[mic_pos] |= LISTENER_FLAG
    else:
      #  ---- SUB LOCATIONS -----
      sub_size = 0.4
      sub_offset = sub_size / 2
      sub_max_from_wall = 0.5

      # Between couch wall
      self.grid.fill_region(
          d_min=sub_offset,
          d_max=sub_max_from_wall,
          h_min=sub_offset,
          h_max=sub_size,
          w_min=sub_offset,
          w_max=1.06-sub_size,
          geometry_flag=SOURCE_REGION_FLAG,
      )

      # Under TV, left
      self.grid.fill_region(
          d_min=sub_offset,
          d_max=1.44-sub_offset,
          h_min=sub_offset,
          h_max=sub_size,
          w_min=sub_offset,
          w_max=sub_max_from_wall,
          geometry_flag=SOURCE_REGION_FLAG,
      )

      # outer wall, left
      self.grid.fill_region(
          d_min=self.depth - sub_max_from_wall,
          d_max=self.depth - sub_offset,
          h_min=sub_offset,
          h_max=sub_size,
          w_min=0.15 + sub_offset,
          w_max=1.4-sub_size,
          geometry_flag=SOURCE_REGION_FLAG,
      )

      # outer wall, right
      self.grid.fill_region(
          d_min=self.depth - sub_max_from_wall,
          d_max=self.depth - sub_offset,
          h_min=sub_offset,
          h_max=sub_size,
          w_min=2.0,
          w_max=self.width - sub_offset,
          geometry_flag=SOURCE_REGION_FLAG,
      )

      # below whiteboard
      self.grid.fill_region(
          d_min=self.depth - 1.7 + sub_offset,
          d_max=self.depth - sub_offset,
          h_min=sub_offset,
          h_max=sub_size,
          w_min=self.width - sub_max_from_wall,
          w_max=self.width - sub_offset,
          geometry_flag=SOURCE_REGION_FLAG,
      )

      #  ---- LISTENER LOCATIONS -----
      # Poof 2
      self.grid.fill_region(
          d_min=self.depth - 0.46,
          h_min=1.0,
          h_max=1.4,
          w_min=1.4,
          w_max=1.4 + 0.46,
          geometry_flag=LISTENER_FLAG,
      )

      # couch, sitting area
      self.grid.fill_region(
          h_min=1.0,
          h_max=1.4,
          d_min=0.1,
          d_max=0.1 + 0.85,
          w_min=1.06,
          w_max=1.06 + 2.35,
          geometry_flag=LISTENER_FLAG,
      )

      # Standing
      self.grid.fill_region(
          h_min=1.6,
          h_max=2.0,
          d_min=1.0,
          d_max=self.depth - 0.8,
          w_min=0.8,
          w_max=self.width - 0.8,
          geometry_flag=LISTENER_FLAG,
      )
