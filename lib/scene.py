from lib.simulation import SOURCE_FLAG, WALL_FLAG, SimulationState


def shoebox_room() -> SimulationState:
  shape = (3.6, 2.6, 4.3)
  sim = SimulationState(shape)
  sim.geometry[sim.width_parts // 2, sim.height_parts //
               2, sim.depth_parts // 2] |= SOURCE_FLAG

  # closet 1
  for d in range(sim.scale(.9), sim.scale(1.9)):
    for h in range(sim.scale(2)):
      for w in range(sim.scale(3.0), sim.scale(3.6)):
        sim.geometry[w, h, d] |= WALL_FLAG

  # closet 2
  for d in range(sim.scale(.4)):
    for h in range(sim.scale(.8)):
      for w in range(sim.scale(1.1), sim.scale(2.55)):
        sim.geometry[w, h, d] |= WALL_FLAG

  # speaker_heigth = sim.scale(.97)
  # for d in range(sim.scale(.4)):
  #   for w in range(sim.scale(1.1), sim.scale(2.55)):
  #     sim.geometry[w, speaker_heigth, d] |= SOURCE_FLAG

  return sim


def bell_box(has_wall: bool) -> SimulationState:
  _width = 2.19
  _height = 2.42
  _depth = 3.03
  shape = (_width, _height, _depth)
  sim = SimulationState(shape)

  # wall
  if (has_wall):
    for w in range(sim.scale(1.26)):
      for h in range(sim.height_parts - 1):
        for d in range(sim.scale(0.05)):
          sim.geometry[w, h, sim.scale(1.91) + d] |= WALL_FLAG

  sim.geometry[sim.scale(_width-1.45), sim.scale(1.35),
               sim.scale(_depth-0.59)] |= SOURCE_FLAG

  return sim
