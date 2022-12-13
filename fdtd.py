import numpy as np
import time
from lib.simulation import simulation_setup, simulation_step, sim

count = 1000
print('Setting up simulation...')
simulation_setup()
simulation_step()

print('Starting up simulation...')
start = time.time()
for i in range(count):
    simulation_step()
end = time.time()

diff = end - start

print(f'Elapsed: {diff} IRL, {sim.time} simulated')
print(f'Average: {(diff / count)}')
print(f'Factor: {(diff / sim.time)}x')
print("Ran simulation!")