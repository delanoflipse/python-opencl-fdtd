import numpy as np
import time
from lib.parameters import DEPTH_PARTS, GRID_SIZE, HEIGHT_PARTS, WIDTH_PARTS
from lib.scene import shoebox_room
from lib.simulation import simulation_setup, simulation_step

count = 1000
sim = shoebox_room()
print(f'w: {WIDTH_PARTS} h:{HEIGHT_PARTS} d:{DEPTH_PARTS}. Total: {GRID_SIZE}')
print('Setting up simulation...')
simulation_setup(sim)
simulation_step(sim)

print('Starting up simulation...')
start = time.time()
for i in range(count):
    simulation_step(sim)
end = time.time()

diff = end - start

print(f'Elapsed: {diff} IRL, {sim.time} simulated')
print(f'Average: {(diff / count)}')
print(f'Factor: {(diff / sim.time)}x')
print('Ran simulation!')