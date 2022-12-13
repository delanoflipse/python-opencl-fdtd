from lib.parameters import DEPTH_PARTS, HEIGHT_PARTS, WIDTH_PARTS
from lib.simulation import GridPosition, SimulationState, Source


def shoebox_room() -> SimulationState:
  sim = SimulationState()
  s1 = Source()
  s1.position = GridPosition(
      WIDTH_PARTS // 2, HEIGHT_PARTS // 2, DEPTH_PARTS // 2)
  s1.frequency = 1000
  sim.sources.append(s1)
  return sim
