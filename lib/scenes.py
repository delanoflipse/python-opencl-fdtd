"""
Predefined grids besed on a number of scenarios
"""
from lib.grid import LISTENER_FLAG, SOURCE_FLAG, SOURCE_REGION_FLAG, WALL_FLAG, SimulationGrid
from lib.materials import SimulatedMaterial
from lib.parameters import SimulationParameters


def shoebox_room(parameters: SimulationParameters) -> SimulationGrid:
  _width = 3.6
  _height = 2.6
  _depth = 4.2
  shape = (_width, _height, _depth)
  grid = SimulationGrid(shape, parameters)
  material_frequency = 20

  concrete_material = SimulatedMaterial("concrete")
  laminate_material = SimulatedMaterial("laminate")
  plaster_material = SimulatedMaterial("plaster")
  glass_material = SimulatedMaterial("glass")
  avg_glass_concrete = (concrete_material.get_beta(
      material_frequency) + glass_material.get_beta(material_frequency)) / 2

  # set edge beta values
  grid.edge_betas.depth_max = concrete_material.get_beta(material_frequency)
  grid.edge_betas.depth_min = concrete_material.get_beta(material_frequency)
  grid.edge_betas.height_max = concrete_material.get_beta(material_frequency)
  grid.edge_betas.height_min = laminate_material.get_beta(material_frequency)
  grid.edge_betas.width_min = avg_glass_concrete
  grid.edge_betas.width_max = plaster_material.get_beta(material_frequency)

  # --- Blocking objects ---
  # closet 1
  # 100x60x200
  closet1_material = SimulatedMaterial("wood")
  grid.fill_region(
      d_min=0.9,  # 90cm from door
      d_max=1.9,  # 1m wide
      h_max=2.0,  # 2M high
      w_min=_width - 0.6,  # 60cm deep, against wall
      geometry_flag=WALL_FLAG,
      beta=closet1_material.get_beta(material_frequency),
  )

  # closet 2
  # 39.5x147x77
  closet2_material = SimulatedMaterial("wood")
  grid.fill_region(
      d_min=0.04,
      d_max=0.43,
      h_max=0.77,
      w_min=1.1,
      w_max=1.1 + 1.47,
      geometry_flag=WALL_FLAG,
      beta=closet2_material.get_beta(material_frequency),
  )

  # bed
  # 39.5x147x77
  closet2_material = SimulatedMaterial("wood")
  grid.fill_region(
      d_min=0.04,
      d_max=0.43,
      h_max=0.77,
      w_min=1.1,
      w_max=1.1 + 1.47,
      geometry_flag=WALL_FLAG,
      beta=closet2_material.get_beta(material_frequency),
  )

  # --- SOURCE LOCATIONS ---
  speaker_height = grid.scale(.97)
  speaker_offset = grid.scale(.1)

  # speaker_locations
  # on closet 2
  grid.fill_region(
      d_min=0.04,
      d_max=0.43,
      h_min=speaker_height,
      h_max=speaker_height + speaker_offset,
      w_min=1.1,
      w_max=1.1 + 1.47,
      geometry_flag=SOURCE_REGION_FLAG,
  )

  # OR single source
  # grid.geometry[grid.scale(1.83), speaker_height,
  #               grid.scale(.2)] |= SOURCE_FLAG

  # --- LISTENER LOCATIONS ---
  # sit/stand
  grid.fill_region(
      d_min=0.6,
      d_max=_depth - 0.8,
      h_min=1.25,
      h_max=1.9,
      w_min=1.14,
      w_max=_width - 0.6,
      geometry_flag=LISTENER_FLAG,
  )
  # bed
  grid.fill_region(
      d_min=_depth - 2.0,
      h_min=0.8,
      h_max=1.3,
      w_min=1.75,
      w_max=_width - 0.45,
      geometry_flag=LISTENER_FLAG,
  )

  grid.build()
  return grid


def bell_box(parameters: SimulationParameters, has_wall: bool) -> SimulationGrid:
  _width = 2.19
  _height = 2.42
  _depth = 3.03
  shape = (_width, _height, _depth)
  grid = SimulationGrid(shape, parameters)

  # wall
  if has_wall:
    grid.fill_region(
        d_min=1.91,
        d_max=1.91+0.05,
        w_max=1.26,
        geometry_flag=WALL_FLAG,
        beta=0.0,
    )

  grid.geometry[grid.scale(_width-1.45), grid.scale(1.35),
                grid.scale(_depth-0.59)] |= SOURCE_FLAG

  grid.build()
  return grid


def concert_hall(parameters: SimulationParameters) -> SimulationGrid:
  _width = 40
  _height = 8
  _depth = 65
  shape = (_width, _height, _depth)
  grid = SimulationGrid(shape, parameters)

  grid.geometry[grid.scale(_width-1.45), grid.scale(1.35),
                grid.scale(_depth-0.59)] |= SOURCE_FLAG

  grid.build()
  return grid
