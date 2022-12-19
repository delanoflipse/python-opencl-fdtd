import numpy as np
import time

from lib.impulse_generators import GaussianModulatedImpulseGenerator
from lib.parameters import SimulationParameters
from lib.scenes import shoebox_room
from lib.simulation import Simulation

iterations_per_step = 4000
count = 40000 // iterations_per_step
params = SimulationParameters()
params.set_max_frequency(500)

grid = shoebox_room(params)
sim = Simulation(params, grid)
sim.generator = GaussianModulatedImpulseGenerator(params.max_frequency)
print(f'w: {grid.width_parts} h:{grid.height_parts} d:{grid.depth_parts}. Total: {grid.grid_size}. {params.sampling_frequency}hz')
print('Setting up simulation...')
sim.step()

print('Starting up simulation...')
start = time.time()
for i in range(count):
  sim.step(iterations_per_step)
end = time.time()

diff = end - start

print(f'Elapsed: {diff} IRL, {sim.time} simulated')
print(f'Average: {(diff / count)}')
print(f'Factor: {(diff / sim.time)}x')
print('Ran simulation!')
