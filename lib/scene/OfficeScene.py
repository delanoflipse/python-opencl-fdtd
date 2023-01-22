from lib.scene.scene import Scene
from lib.materials import metal_material, painted_concrete_material, whiteboard_material, cellulose_material, carpet_material, hard_wood_material
from lib.parameters import SimulationParameters
from lib.grid import LISTENER_FLAG, SOURCE_REGION_FLAG, WALL_FLAG


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

    # materials
    painted_concrete = painted_concrete_material.get_beta(run_frequency)
    whiteboard = whiteboard_material.get_beta(run_frequency)
    cellulose = cellulose_material.get_beta(run_frequency)
    carpet = carpet_material.get_beta(run_frequency)
    metal = metal_material.get_beta(run_frequency)
    hard_wood = hard_wood_material.get_beta(run_frequency)

    # set outer edge beta values
    self.grid.edge_betas.depth_max = painted_concrete
    self.grid.edge_betas.depth_min = whiteboard
    self.grid.edge_betas.height_max = cellulose
    self.grid.edge_betas.height_min = carpet
    self.grid.edge_betas.width_min = whiteboard
    self.grid.edge_betas.width_max = painted_concrete

    # metal closet 100x45x2000
    self.grid.fill_region(
        d_min=1.2,
        d_max=1.2 + 1,
        w_max=0.45,
        h_max=2.0,
        geometry_flag=WALL_FLAG,
        beta=metal,
    )

    # --- inset walls of concrete and wood in width direction ---
    # desk height concrete
    self.grid.fill_region(
        d_min=self.depth - 0.29,
        h_max=0.92,
        geometry_flag=WALL_FLAG,
        beta=painted_concrete,
    )
    # small range of wood, overwrite concrete
    self.grid.fill_region(
        d_min=self.depth - 0.54,
        h_max=0.92,
        h_min=0.92 - 0.19,
        geometry_flag=WALL_FLAG,
        beta=hard_wood,
    )
    # upper part of concrete
    self.grid.fill_region(
        d_min=self.depth - 0.29,
        h_min=self.height - 0.2,
        geometry_flag=WALL_FLAG,
        beta=painted_concrete,
    )
    # non-glass concrete part
    self.grid.fill_region(
        w_min=1.0,
        d_min=self.depth - 0.29,
        geometry_flag=WALL_FLAG,
        beta=painted_concrete,
    )

    # --- inset walls of concrete and wood in depth direction ---
    # outset wood range
    self.grid.fill_region(
        w_min=self.width-0.54,
        h_min=0.92 - 0.19,
        h_max=0.92,
        d_max=self.depth-0.29,
        geometry_flag=WALL_FLAG,
        beta=hard_wood,
    )
    # upper concrete
    self.grid.fill_region(
        w_min=self.width-0.54,
        h_min=self.height - 0.2,
        geometry_flag=WALL_FLAG,
        beta=painted_concrete,
    )

    # below wood concrete
    self.grid.fill_region(
        w_min=self.width-0.29,
        h_max=0.92,
        geometry_flag=WALL_FLAG,
        beta=painted_concrete,
    )

    # wall concrete 1
    self.grid.fill_region(
        w_min=self.width-0.29,
        d_max=0.085,
        geometry_flag=WALL_FLAG,
        beta=painted_concrete,
    )

    # wall concrete 2
    self.grid.fill_region(
        w_min=self.width-0.29,
        d_min=1.085,
        d_max=2.465,
        geometry_flag=WALL_FLAG,
        beta=painted_concrete,
    )

    # wall concrete 3
    self.grid.fill_region(
        w_min=self.width-0.29,
        d_min=3.465,
        geometry_flag=WALL_FLAG,
        beta=painted_concrete,
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
    # wall_offset = 0.7
    # self.grid.fill_region(
    #     d_min=wall_offset,
    #     d_max=self.depth - wall_offset,
    #     w_min=wall_offset,
    #     w_max=self.width - wall_offset,
    #     h_min=1.2,
    #     h_max=2.0,
    #     geometry_flag=LISTENER_FLAG,
    # )
    self.grid.fill_region(
        d_min=2.2,
        d_max=2.5,
        w_min=1,
        w_max=2,
        h_min=0.5,
        h_max=1.5,
        geometry_flag=LISTENER_FLAG,
    )
