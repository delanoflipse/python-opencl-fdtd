"""Time the throughput of the simulation kernel."""
import time
from typing import Optional

from fdtd.impulse_generators import GaussianModulatedImpulseGenerator
from fdtd.parameters import SimulationParameters
from fdtd.scenes import VolumetricScene, create_scene
from fdtd.simulation import Simulation


def run_benchmark(iterations_per_step: int = 2 ** 13, step_count: int = 2 ** 3,
                  oversampling: float = 16, scene: Optional[str] = None,
                  volume: float = 50, precision: Optional[str] = None) -> float:
  """Run the benchmark and return the simulated cells per second.

  Without a scene name, an empty cube of ``volume`` m^3 is simulated.
  """
  params = SimulationParameters()
  params.set_max_frequency(200)
  params.set_oversampling(oversampling)
  if precision is not None:
    params.set_precision(precision)

  if scene is None:
    bench_scene = VolumetricScene(params, volume)
  else:
    bench_scene = create_scene(scene, params)
    volume = bench_scene.width * bench_scene.height * bench_scene.depth

  grid = bench_scene.build()
  grid.select_source_locations([grid.source_set[0]])
  sim = Simulation(params, grid)
  sim.generator = GaussianModulatedImpulseGenerator(params.max_frequency)
  sim.print_statistics()

  iteration_count = iterations_per_step * step_count
  print('Setting up simulation...')
  print(
      f'[Benchmark] iterations/step: {iterations_per_step}\tcount: {step_count}x')
  print(f'[Benchmark] iterations: {iteration_count}\t'
        f'time: {iteration_count * params.dt}s')
  sim.step()

  print('Starting up simulation...')
  start = time.time()
  for i in range(step_count):
    sim.step(iterations_per_step)
    print(f'{(i + 1) / step_count:.1%} done')
  diff = time.time() - start

  time_factor = diff / sim.time
  cells_per_second = grid.grid_size / time_factor
  print(f'Elapsed: {diff}s IRL, {sim.time}s simulated.')
  print(f'Factor: {time_factor}x.')
  print(
      f'Average: {(1000 * diff / step_count / iterations_per_step)}ms per step.')
  print(f'cells/s:\t{cells_per_second:0.1f}')
  print(f'm^3/s:\t{volume / time_factor:0.4f}')
  return cells_per_second
