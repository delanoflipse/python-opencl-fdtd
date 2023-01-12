
import csv
import math
import logging
import os
import sys
from datetime import datetime
from time import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lib.analysis.frequency_sweep import get_avg_dev, get_avg_spl, run_sweep_analysis
from lib.impulse_generators import SimpleSinoidGenerator
from lib.math.decibel_weightings import get_a_weighting
from lib.math.octaves import get_octaval_center_frequencies
from lib.parameters import SimulationParameters
from lib.scenes import ShoeboxRoomScene, BellBoxScene, ConcertHallScene
from lib.simulation import Simulation

# --- SELECT PARAMETERS ---
SIMULATED_TIME = 0.5
MAX_FREQUENCY = 200
OVERSAMPLING = 16
OCTAVE_BANDS = 6
USE_REALTIME_VISUALS = False
USE_VISUALS = True
USE_FILE_LOGS = True
WRITE_CSV = True
LOG_LEVEL = logging.DEBUG
# -----

# ---- Logging ----
file_dir = os.path.dirname(__file__)
log = logging.getLogger("FDTD")
log.setLevel(LOG_LEVEL)
# logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] - %(message)s")
logFormatter = logging.Formatter(
    "%(asctime)s [%(levelname)-5.5s] - %(message)s")

output_uid = f'{datetime.now().strftime("%Y-%m-%d %H_%M_%S")} [{MAX_FREQUENCY}-{OVERSAMPLING}-{OCTAVE_BANDS}]'

if USE_FILE_LOGS:
  fileHandler = logging.FileHandler(
      "{0}/{1}.log".format(os.path.join(file_dir, "output"), output_uid))
  fileHandler.setFormatter(logFormatter)
  log.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)

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
log.info('---- Statring simulation ----')
log.info('%d steps per sim', runtime_steps)
log.info('%d frequencies', testing_frequencies.size)
log.info('Scene: %s', scene.__class__.__name__)
log.info('-----------------------------')
# ----- Simulation end -----

# ---- CSV ----
if WRITE_CSV:
  csv_path = os.path.join(file_dir, "output", f"{output_uid}.csv")
  csv_file = open(csv_path, 'w', encoding="utf-8", newline='')
  writer = csv.writer(csv_file)

  header_row = [
      "Time",
      "Index",
      "W idx",
      "w (m)",
      "H idx",
      "h (m)",
      "D idx",
      "d (m)",
      "Deviation",
      "SPL (avg dB)",
      "Bands (SPL dB):",
      *map(lambda x: f'{x:.2f}', testing_frequencies.tolist())
  ]
  writer.writerow(header_row)
  csv_file.flush()

# state
spl_values_per_source = []
sources_covered = []
deviations = []

# ---- Chart & Axis ----
# get and set style
plt.style.use(os.path.join(file_dir, './styles/poster.mplstyle'))

# create subplot axis
fig = plt.gcf()
fig.set_dpi(150)
fig.set_size_inches(1920/fig.get_dpi(), 1080/fig.get_dpi(), forward=True)
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
axis_spl.set_xticks([20, 25, 30, 40, 50, 60, 80, 100, 120, 160, 200])
axis_spl.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# ---- Analysis ----
# per source
sweep_sum = sim.grid.create_grid("float64")
sweep_sum_sqr = sim.grid.create_grid("float64")
sweep_deviation = sim.grid.create_grid("float64")
sweep_ranking = sim.grid.create_grid("float64")


deviation_plot, = axis_deviation.plot([], [], "-")

timings = []


def run_source_analysis_iteration() -> bool:
  source_index = grid.source_index
  location = None
  try:
    location = sim.grid.select_source(source_index)
  except:
    return True

  spl_values = []
  frequencies_covered = []
  sources_covered.append(source_index)

  (w, h, d) = location

  sweep_sum.fill(0.0)
  sweep_sum_sqr.fill(0.0)
  sweep_deviation.fill(0.0)
  sweep_ranking.fill(0.0)

  start = time()
  log.info(
      f'Picked source {source_index}/{grid.source_count} [{w}, {h}, {d}]')
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
    # a_weighting = get_a_weighting(frequency)
    a_spl = avg_spl
    spl_values.append(a_spl)
    log.info(f'[{source_index}] {frequency:.2f}hz: {a_spl:.2f} SPL (dB)')
  deviation = get_avg_dev(sweep_deviation, grid.geometry)
  avg_spl = np.average(spl_values)
  deviations.append(deviation)
  spl_values_per_source.append(spl_values)
  log.info(f'Deviation: {deviation} ')
  if USE_VISUALS:
    axis_spl.plot(testing_frequencies, spl_values)

  if WRITE_CSV:
    csv_row = [
        time(),
        source_index,
        w,
        f'{(w + 0.5) * parameters.dx:.2f}',
        h,
        f'{(h + 0.5) * parameters.dx:.2f}',
        d,
        f'{(d + 0.5) * parameters.dx:.2f}',
        f'{deviation:.4f}',
        f'{avg_spl:.4f}',
        "",
        *map(lambda x: f'{x:.2f}', spl_values),
    ]
    writer.writerow(csv_row)
    csv_file.flush()
  # report timing
  end = time()
  diff = end - start
  timings.append(diff)
  avg_timing = np.average(timings)
  indexes_left = grid.source_count - source_index
  time_left = indexes_left * avg_timing
  hours = math.floor(time_left / 60 / 60)
  minutes = math.floor((time_left - hours * 60 * 60) / 60)
  seconds = time_left - hours * 60 * 60 - minutes * 60
  log.info(
      f'Elapsed: {diff:.1f}s, est: {avg_timing:.1f}s/run = {hours:}h {minutes:.0f}m {seconds:.2f}s left')

  # done, signal next iteration
  log.info('----- End iteration %d -----', source_index)
  grid.source_index += 1
  return False


if USE_REALTIME_VISUALS:
  plt.show(block=False)
  # wm = plt.get_current_fig_manager()
  # wm.full_screen_toggle()
  # wm.resize(1920, 1080)
start_time = time()
for i in range(grid.source_count):
  run_source_analysis_iteration()
  deviation_plot.set_data(sources_covered, deviations)

  if USE_REALTIME_VISUALS:
    axis_deviation.relim()
    axis_deviation.autoscale_view()
    axis_spl.relim()
    axis_spl.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.tight_layout()

end_time = time()
diff = end_time - start_time
log.info(f'Time elapsed for full sweep: {diff:.1f}s')

if USE_VISUALS:
  axis_deviation.relim()
  axis_deviation.autoscale_view()
  axis_spl.relim()
  axis_spl.autoscale_view()
  plt.tight_layout()
  plt.savefig(os.path.join(file_dir, "output", f'{output_uid}.png'), dpi=300)
if USE_REALTIME_VISUALS:
  plt.show(block=True)

if WRITE_CSV:
  csv_file.close()
