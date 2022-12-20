from lib.grid import LISTENER_FLAG, SOURCE_FLAG, WALL_FLAG, SimulationGrid
from lib.parameters import SimulationParameters


def shoebox_room(parameters: SimulationParameters) -> SimulationGrid:
  _width = 3.6
  _height = 2.6
  _depth = 4.3
  shape = (_width, _height, _depth)
  grid = SimulationGrid(shape, parameters)

  # closet 1
  grid.fill_region(
      d_min=0.9,
      d_max=1.9,
      h_max=2,
      w_min=3.0,
      geometry_flag=WALL_FLAG,
      beta=0.15,
  )

  # closet 2
  grid.fill_region(
      d_max=0.4,
      h_max=0.8,
      w_min=1.1,
      w_max=2.55,
      geometry_flag=WALL_FLAG,
      beta=0.3,
  )

  speaker_height = grid.scale(.97)

  # speaker_locations
  grid.fill_region(
      d_max=0.4,
      h_min=.8,
      h_max=1.1,
      w_min=1.1,
      w_max=2.55,
      geometry_flag=LISTENER_FLAG,
  )

  grid.fill_region(
      d_min=0.8,
      d_max=_depth-0.4,
      h_min=1.6,
      h_max=1.9,
      w_min=1.1,
      w_max=2.55,
      geometry_flag=SOURCE_FLAG,
  )

  # grid.geometry[grid.width_parts // 2, grid.height_parts //
  #               2, grid.depth_parts // 2] |= SOURCE_FLAG

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
