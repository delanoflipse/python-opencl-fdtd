
import numpy as np
from lib.impulse_generators import GaussianModulatedImpulseGenerator
from lib.simulation import Simulation


def frequency_sweep(sim: Simulation) -> np.ndarray:
  sweep_analysis = sim.grid.create_grid("float64")
  runtime_steps = int(0.5 / sim.parameters.dt)
  for f in range(sim.parameters.min_frequency, sim.parameters.max_frequency, sim.parameters.frequency_interval):
    sim.generator = GaussianModulatedImpulseGenerator(f)
    sim.grid.reset_values()
    sim.step(runtime_steps)
    # TODO process values
  return sweep_analysis
