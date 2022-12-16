import numpy as np
import time
from lib.parameters import MAX_FREQUENCY
from lib.scene import shoebox_room

count = 1000
sim = shoebox_room()
print(f'w: {sim.width_parts} h:{sim.height_parts} d:{sim.depth_parts}. Total: {sim.grid_size}. {MAX_FREQUENCY}hz')
print('Setting up simulation...')
sim.setup()
sim.step()

print('Starting up simulation...')
start = time.time()
for i in range(count):
  sim.step()
end = time.time()

diff = end - start

print(f'Elapsed: {diff} IRL, {sim.time} simulated')
print(f'Average: {(diff / count)}')
print(f'Factor: {(diff / sim.time)}x')
print('Ran simulation!')
