from lib.scene.ShoeboxReferenceScene import ShoeboxReferenceScene
from lib.scene.RealLifeRoomScene import RealLifeRoomScene
from lib.simulation import Simulation
from lib.parameters import SimulationParameters
from lib.impulse_generators import GaussianModulatedImpulseGenerator
import numpy as np
import time

iterations_per_step = 2 ** 14
# count = 40000 // iterations_per_step
step_count = 2 ** 6
params = SimulationParameters()
params.set_max_frequency(200)
params.set_oversampling(16)
# params.set_scheme(1.0, 1 / 4, 1 / 16)

# ---- SCENE ----
scene = RealLifeRoomScene(params)
# ----

grid = scene.build()
grid.select_source_locations([grid.source_set[0]])
sim = Simulation(params, grid)
sim.generator = GaussianModulatedImpulseGenerator(params.max_frequency)

sim.print_statistics()

iteration_count = iterations_per_step * step_count
test_time = iteration_count * params.dt
volume = scene.width * scene.height * scene.depth
print('Setting up simulation...')
print(
    f'[Benchmark] iterations/step: {iterations_per_step}\tcount: {step_count}x')
print(
    f'[Benchmark] iterations: {iteration_count}\ttime: {test_time}s\t vol: {volume:0.1f} m3')
sim.step()

print('Starting up simulation...')
start = time.time()
for i in range(step_count):
  sim.step(iterations_per_step)
  p = (i + 1) / step_count
  print(f'{p:.1%} done')
end = time.time()

diff = end - start
time_factor = diff / sim.time
normalised_volume = volume / time_factor

print(f'Elapsed: {diff}s IRL, {sim.time}s simulated.')
print(f'Factor: {(diff / sim.time)}x.')
print(f'm3/s: {(normalised_volume)}')
print(
    f'Average: {(1000 * diff / step_count / iterations_per_step)}ms per step.')
print('Ran simulation!')
