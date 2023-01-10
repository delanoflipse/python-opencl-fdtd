
import math
import os
from xmlrpc.client import boolean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lib.analysis.frequency_sweep import get_avg_dev, get_avg_spl, run_sweep_analysis
from lib.impulse_generators import SimpleSinoidGenerator
from lib.math.octaves import get_octaval_center_frequencies
from lib.parameters import SimulationParameters
from lib.scenes import ShoeboxRoomScene, BellBoxScene, ConcertHallScene
from lib.simulation import Simulation

# --- SELECT PARAMETERS ---
SIMULATED_TIME = 0.5
MAX_FREQUENCY = 200
OVERSAMPLING = 16
OCTAVE_BANDS = 3
# -----


# ---- Simulation ----
parameters = SimulationParameters()
parameters.set_oversampling(OVERSAMPLING)
parameters.set_max_frequency(MAX_FREQUENCY)
runtime_steps = int(SIMULATED_TIME / parameters.dt)
testing_frequencies = get_octaval_center_frequencies(
    20, 200, fraction=OCTAVE_BANDS)

# -- SELECT SCENE --
scene = ShoeboxRoomScene(parameters)
# scene = BellBoxScene(parameters, has_wall=True)
# scene = ConcertHallScene(parameters)
grid = scene.build()
# -----

SLICE_HEIGHT = grid.scale(1.82)
# SLICE_HEIGHT = grid.scale(.97)
# SLICE_HEIGHT = grid.scale(.97) + 1
sim = Simulation(grid=grid, parameters=parameters)
sim.print_statistics()
analysis_key_index = sim.grid.analysis_keys["LEQ"]
print(f'{runtime_steps} steps per sim, {testing_frequencies.size} frequencies. Scene: {scene.__class__.__name__}')
# ----- Simulation end -----


# state
spl_values_per_source = []
sources_covered = []
deviations = []

# ---- Chart & Axis ----
# get and set style
file_dir = os.path.dirname(__file__)
plt.style.use(os.path.join(file_dir, './styles/poster.mplstyle'))

# create subplot axis
fig = plt.gcf()
axes_shape = (1, 2)
axis_deviation = plt.subplot2grid(axes_shape, (0, 0))
axis_spl = plt.subplot2grid(axes_shape, (0, 1))

axis_deviation.set_title("Standard deviation per sweep")
axis_deviation.set_xlabel("Sweep Index")
axis_deviation.set_ylabel("Standard deviation")

axis_spl.set_title("SPL values per frequency band")
axis_spl.set_xlabel("Frequency (hz)")
axis_spl.set_ylabel("SPL Leq (dB)")
axis_spl.set_xscale('log')

# ---- Analysis ----
# per source
sweep_sum = sim.grid.create_grid("float64")
sweep_sum_sqr = sim.grid.create_grid("float64")
sweep_deviation = sim.grid.create_grid("float64")
sweep_ranking = sim.grid.create_grid("float64")


deviation_plot, = axis_deviation.plot([], [], "-")


def run_source_analysis_iteration() -> boolean:
  source_index = grid.source_index
  try:
    sim.grid.select_source(source_index)
  except:
    return True

  spl_values = []
  frequencies_covered = []
  sources_covered.append(source_index)

  sweep_sum.fill(0.0)
  sweep_sum_sqr.fill(0.0)
  sweep_deviation.fill(0.0)
  sweep_ranking.fill(0.0)

  print(f'Picked source {source_index}/{grid.source_count}')
  for (index,), frequency in np.ndenumerate(testing_frequencies):
    parameters.set_signal_frequency(frequency)
    frequencies_covered.append(frequency)
    sim.generator = SimpleSinoidGenerator(parameters.signal_frequency)
    scene.rebuild()
    sim.write_read_buffer()
    sim.reset()
    # run single simulation
    sim.step(runtime_steps)
    run_sweep_analysis(grid.analysis, sweep_sum, sweep_sum_sqr,
                       sweep_deviation, sweep_ranking, analysis_key_index, index + 1)
    avg_spl = get_avg_spl(grid.analysis, grid.geometry, analysis_key_index)
    spl_values.append(avg_spl)
    print(f'[{source_index}] {frequency:.2f}hz: {avg_spl:.2f} SPL (dB)')
  deviation = get_avg_dev(sweep_deviation, grid.geometry)
  deviations.append(deviation)
  spl_values_per_source.append(spl_values)
  print(f'[{source_index}] deviation: {deviation} ')
  grid.source_index += 1
  axis_spl.plot(testing_frequencies, spl_values)


plt.show(block=False)

for i in range(grid.source_count):
  run_source_analysis_iteration()
  deviation_plot.set_data(sources_covered, deviations)

  axis_deviation.relim()
  axis_deviation.autoscale_view()
  axis_spl.relim()
  axis_spl.autoscale_view()

  fig.canvas.draw()
  fig.canvas.flush_events()
  # TODO: write to csv file, in case of crash

plt.show(block=True)
