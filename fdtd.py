import numpy as np
import time
from lib.parameters import SAMPLING_FREQUENCY
from lib.scene import shoebox_room

iterations_per_step = 4000
count = 40000 // iterations_per_step

sim = shoebox_room()
print(f'w: {sim.width_parts} h:{sim.height_parts} d:{sim.depth_parts}. Total: {sim.grid_size}. {SAMPLING_FREQUENCY}hz')
print('Setting up simulation...')
sim.setup()
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
