import numpy as np
import time

from lib.impulse_generators import GaussianModulatedImpulseGenerator
from lib.parameters import SimulationParameters
from lib.scenes import concert_hall, shoebox_room
from lib.simulation import Simulation

iterations_per_step = 4000
# count = 40000 // iterations_per_step
count = 2
params = SimulationParameters()
params.set_max_frequency(200)
# params.set_oversampling(9)

grid = concert_hall(params)
# grid = shoebox_room(params)
sim = Simulation(params, grid)
sim.generator = GaussianModulatedImpulseGenerator(params.max_frequency)
print(f'w: {grid.width_parts} h:{grid.height_parts} d:{grid.depth_parts}. Total: {grid.grid_size}.')
print(f'{params.sampling_frequency}hz target. {params.dt_hz:0.0f}hz speed. {params.dx}m size. {grid.get_storage_str()} needed')
print('Setting up simulation...')
sim.step()

print('Starting up simulation...')
start = time.time()
for i in range(count):
  sim.step(iterations_per_step)
end = time.time()

diff = end - start

print(f'Elapsed: {diff}s IRL, {sim.time}s simulated.')
print(f'Factor: {(diff / sim.time)}x.')
print(f'Average: {(1000 * diff / count / iterations_per_step)}ms per step.')
print('Ran simulation!')
