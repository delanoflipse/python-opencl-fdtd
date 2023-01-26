from lib.scene.VolumetricScene import VolumetricScene
from lib.simulation import Simulation
from lib.parameters import SimulationParameters
from lib.impulse_generators import GaussianModulatedImpulseGenerator
import time
import argparse

cli_argument_parser = argparse.ArgumentParser()
cli_argument_parser.add_argument(
    "-i", "--iterations", default=2 ** 13, type=int)
cli_argument_parser.add_argument("-s", "--steps", default=2 ** 3, type=int)
cli_argument_parser.add_argument(
    "-o", "--oversampling", default=16, type=float)
cli_argument_parser.add_argument("-v", "--volume", default=50, type=float)

arguments = cli_argument_parser.parse_args()
print(arguments)

iterations_per_step = arguments.iterations
step_count = arguments.steps
params = SimulationParameters()
params.set_max_frequency(200)
params.set_oversampling(arguments.oversampling)
scene = VolumetricScene(params, arguments.volume)

grid = scene.build()
grid.select_source_locations([grid.source_set[0]])
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
time_factor = diff / sim.time
cell_count = grid.grid_size
normalised_cells = cell_count / time_factor
normalised_volume = arguments.volume / time_factor

print(f'Elapsed: {diff}s IRL, {sim.time}s simulated.')
print(f'Factor: {(time_factor)}x.')
print(
    f'Average: {(1000 * diff / step_count / iterations_per_step)}ms per step.')
print(f'cells/s:\t{normalised_cells:0.1f}')
print(f'm^3/s:\t{normalised_volume:0.4f}')
print('Ran simulation!')
