import numpy as np
import time

from lib.impulse_generators import GaussianModulatedImpulseGenerator
from lib.parameters import SimulationParameters
from lib.scenes import ShoeboxRoomScene, BellBoxScene, ConcertHallScene
from lib.simulation import Simulation

iterations_per_step = 2 ** 11
# count = 40000 // iterations_per_step
step_count = 2 ** 3
params = SimulationParameters()
# params.set_oversampling(9)
params.set_max_frequency(400)
params.set_signal_frequency(200)

scene = ShoeboxRoomScene(params)
# scene = BellBoxScene(parameters, has_wall=True)
# scene = ConcertHallScene(parameters)
grid = scene.build()
sim = Simulation(params, grid)
sim.generator = GaussianModulatedImpulseGenerator(params.max_frequency)
sim.print_statistics()

iteration_count = iterations_per_step * step_count
test_time = iteration_count * params.dt
print('Setting up simulation...')
print(
    f'[Benchmark] iterations/step: {iterations_per_step}\tcount: {step_count}x')
print(f'[Benchmark] iterations: {iteration_count}\ttime: {test_time}s')
sim.step()

print('Starting up simulation...')
start = time.time()
for i in range(step_count):
  sim.step(iterations_per_step)
  p = (i + 1) / step_count
  print(f'{p:.1%} done')
end = time.time()

diff = end - start

print(f'Elapsed: {diff}s IRL, {sim.time}s simulated.')
print(f'Factor: {(diff / sim.time)}x.')
print(
    f'Average: {(1000 * diff / step_count / iterations_per_step)}ms per step.')
print('Ran simulation!')
