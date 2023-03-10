
from lib.simulation import Simulation
from lib.scene.scene import Scene
from lib.scene.BedroomScene import BedroomScene
from lib.scene.BellBoxScene import BellBoxScene
from lib.scene.ConcertHallScene import ConcertHallScene
from lib.scene.CuboidReferenceScene import CuboidReferenceScene
from lib.scene.LShapedRoomScene import LShapedRoomScene
from lib.scene.OfficeScene import OfficeScene
from lib.scene.ShoeboxReferenceScene import ShoeboxReferenceScene
from lib.scene.StudioRoomScene import StudioRoomScene
from lib.scene.RealLifeRoomScene import RealLifeRoomScene

from lib.parameters import SimulationParameters
from lib.math.octaves import get_octaval_center_frequencies
from lib.math.decibel_weightings import get_a_weighting
from lib.impulse_generators import SimpleSinoidGenerator
from lib.analysis.source_pairs import get_n_pairs_with_min_distance
from lib.analysis.frequency_sweep import get_avg_dev, get_avg_spl, run_sweep_analysis
import matplotlib.pyplot as plt

import csv
import math
import logging
import os
import sys
import argparse
from datetime import datetime
from time import time
import numpy as np
import matplotlib
matplotlib.use('Agg')

# TODO: combine with full sweep!

cli_argument_parser = argparse.ArgumentParser()
cli_argument_parser.add_argument("-s", "--scene", default="shoebox")
cli_argument_parser.add_argument("-t", "--time", default=0.3, type=float)
cli_argument_parser.add_argument(
    "-o", "--oversampling", default=16, type=float)
cli_argument_parser.add_argument("-f", "--frequency", default=200, type=float)
cli_argument_parser.add_argument("-b", "--bands", default=24, type=float)
cli_argument_parser.add_argument("-x", "--speakers", default=1, type=int)
cli_argument_parser.add_argument("--distance", default=2.0, type=int)
cli_argument_parser.add_argument(
    "--novisuals", default=False, action="store_true")
cli_argument_parser.add_argument(
    "--nologs", default=False, action="store_true")
cli_argument_parser.add_argument("--nocsv", default=False, action="store_true")

arguments = cli_argument_parser.parse_args()

# --- SELECT PARAMETERS ---
SIMULATED_TIME = arguments.time
MAX_FREQUENCY = arguments.frequency
OVERSAMPLING = arguments.oversampling
OCTAVE_BANDS = arguments.bands
SPEAKERS = arguments.speakers
MIN_DISTANCE_BETWEEN_SPEAKERS = arguments.distance
OUTPUT_VISUALS = not arguments.novisuals
OUTPUT_FILE_LOGS = not arguments.nologs
OUTPUT_CSV = not arguments.nocsv
LOG_LEVEL = logging.DEBUG
# -----

print(arguments)

# ---- Simulation ----
parameters = SimulationParameters()
parameters.set_oversampling(OVERSAMPLING)
parameters.set_max_frequency(MAX_FREQUENCY)
# parameters.set_scheme(1.0, 1 / 4, 1 / 16)

runtime_steps = int(SIMULATED_TIME / parameters.dt)
testing_frequencies = get_octaval_center_frequencies(
    20, 200, fraction=OCTAVE_BANDS)

# -- SELECT SCENE --
scene: Scene = None
if arguments.scene == "real-reference":
  scene = RealLifeRoomScene(parameters, True)
if arguments.scene == "real-scene":
  scene = RealLifeRoomScene(parameters)
if arguments.scene == "bedroom":
  scene = BedroomScene(parameters)
elif arguments.scene == "bellbox":
  scene = BellBoxScene(parameters)
elif arguments.scene == "concert":
  scene = ConcertHallScene(parameters)
elif arguments.scene == "shoebox":
  scene = ShoeboxReferenceScene(parameters)
elif arguments.scene == "lshape":
  scene = LShapedRoomScene(parameters)
elif arguments.scene == "cuboid":
  scene = CuboidReferenceScene(parameters)
elif arguments.scene == "office":
  scene = OfficeScene(parameters)
elif arguments.scene == "studio":
  scene = StudioRoomScene(parameters)

grid = scene.build()
# -----

sim = Simulation(grid=grid, parameters=parameters)
sim.print_statistics()
analysis_key_index = sim.grid.analysis_keys["LEQ"]
position_sets = get_n_pairs_with_min_distance(
    grid.source_set, SPEAKERS, parameters.dx, MIN_DISTANCE_BETWEEN_SPEAKERS)

# ---- Logging ----
file_dir = os.path.dirname(__file__)
log = logging.getLogger("FDTD")
log.setLevel(LOG_LEVEL)
logFormatter = logging.Formatter(
    "%(asctime)s [%(levelname)-5.5s] - %(message)s")

output_uid = f'{datetime.now().strftime("%Y-%m-%d %H_%M_%S")} {scene.__class__.__name__} [{SIMULATED_TIME*1000:.0f}ms-{MAX_FREQUENCY}f-{OVERSAMPLING}o-{OCTAVE_BANDS}b-{SPEAKERS}x]'

if OUTPUT_FILE_LOGS:
  fileHandler = logging.FileHandler(
      "{0}/{1}.log".format(os.path.join(file_dir, "output"), output_uid))
  fileHandler.setFormatter(logFormatter)
  log.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)

log.info('---- Statring simulation ----')
log.info('%d pair(s)', len(position_sets))
log.info('%d steps per sim', runtime_steps)
log.info('%d frequencies', testing_frequencies.size)
log.info('Scene: %s', scene.__class__.__name__)
log.info('-----------------------------')
# ----- Simulation end -----

# ---- CSV ----
if OUTPUT_CSV:
  csv_path = os.path.join(file_dir, "output", f"{output_uid}.csv")
  csv_file = open(csv_path, 'w', encoding="utf-8", newline='')
  writer = csv.writer(csv_file)

  header_row = [
      "Time",
      "Index",
  ]

  for i in range(SPEAKERS):
    header_row = header_row + [
        f"W{i} idx",
        f"w{i} (m)",
        f"H{i} idx",
        f"h{i} (m)",
        f"D{i} idx",
        f"d{i} (m)",
    ]

  header_row += [
      "Deviation",
      "SPL (avg dB)",
      "Bands (SPL dB):",
      *map(lambda x: f'{x:.2f}', testing_frequencies.tolist())
  ]

  writer.writerow(header_row)
  csv_file.flush()

# state
spl_values_per_source = []
max_spl_values_per_source = []
min_spl_values_per_source = []
sources_covered = []
deviations = []

# ---- Chart & Axis ----
# get and set style
plt.style.use(os.path.join(file_dir, './styles/poster.mplstyle'))

# create subplot axis
fig = plt.gcf()
fig.set_dpi(150)
fig.set_size_inches(1920/fig.get_dpi(), 1080/fig.get_dpi(), forward=True)
axes_shape = (2, 2)
axis_deviation = plt.subplot2grid(axes_shape, (0, 0))
axis_spl = plt.subplot2grid(axes_shape, (0, 1))
axis_best_spl = plt.subplot2grid(axes_shape, (1, 0), colspan=2)

axis_deviation.set_title("Standard deviation per sweep")
axis_deviation.set_xlabel("Sweep Index")
axis_deviation.set_ylabel("Standard deviation")

axis_spl.set_title("SPL values per frequency band")
axis_best_spl.set_title("-")
for ax in [axis_best_spl, axis_spl]:
  ax.set_xlabel("Frequency (hz)")
  ax.set_ylabel("SPL Leq (dB)")
  ax.set_xscale('log')
  ax.set_xticks([20, 25, 30, 40, 50, 60, 80, 100, 120, 160, 200])
  ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# ---- Analysis ----
deviation_plot, = axis_deviation.plot([], [], "-")
best_spl_plot, worst_spl_plot = axis_best_spl.plot([], [], [], "-")
max_spl_plot, = axis_best_spl.plot([], [], "--")
min_spl_plot, = axis_best_spl.plot([], [], "--")

timings = []

min_dev = float("inf")
max_dev = -float("inf")

room_modes = scene.get_room_modes()
for (modal_frequency, axis_type) in room_modes:
  if modal_frequency > testing_frequencies[-1]:
    continue
  axis_best_spl.axvline(modal_frequency, linestyle='--', color='k', alpha=0.5)


def run_source_analysis_iteration(source_index: int) -> bool:
  global min_dev, max_dev
  source_set = position_sets[source_index]
  sim.grid.select_source_locations(source_set)
  spl_values = []
  max_spl_values = []
  min_spl_values = []
  frequencies_covered = []
  sources_covered.append(source_index)

  start = time()
  log.info(
      f'Picked source set {source_index + 1}/{len(position_sets)} with positions:')

  for pos in source_set:
    log.info('%d, %d, %d', *pos)

  for (index,), frequency in np.ndenumerate(testing_frequencies):
    parameters.set_signal_frequency(frequency)
    frequencies_covered.append(frequency)
    sim.generator = SimpleSinoidGenerator(parameters.signal_frequency)
    scene.rebuild()
    sim.reset()
    sim.sync_read_buffers()
    # run single simulation
    sim.step(runtime_steps)
    avg_spl, min_spl, max_spl = get_avg_spl(
        grid.analysis, grid.geometry, analysis_key_index)
    # a_weighting = get_a_weighting(frequency)
    # a_spl = avg_spl + a_weighting
    a_spl = avg_spl
    max_spl_values.append(max_spl)
    min_spl_values.append(min_spl)
    spl_values.append(a_spl)
    log.info(f'[{source_index}] {frequency:.2f}hz: {a_spl:.2f} SPL (dB)')
  derrivative2 = np.diff(spl_values, n=1)
  deviation = np.sum(np.power(derrivative2, 2))
  avg_spl = np.average(spl_values)
  deviations.append(deviation)
  spl_values_per_source.append(spl_values)
  log.info(f'Deviation: {deviation} ')

  if OUTPUT_VISUALS and deviation != 0.0:
    axis_spl.plot(testing_frequencies, spl_values)
    if deviation < min_dev:
      axis_best_spl.set_title(
          f"best/worst ({source_index}) SPL values per frequency band")
      min_dev = deviation
      best_spl_plot.set_data(testing_frequencies, spl_values)
      min_spl_plot.set_data(testing_frequencies, min_spl_values)
      max_spl_plot.set_data(testing_frequencies, max_spl_values)
    if deviation > max_dev:
      max_dev = deviation
      worst_spl_plot.set_data(testing_frequencies, spl_values)

  if OUTPUT_CSV:
    csv_row = [
        time(),
        source_index,
    ]

    for i in range(SPEAKERS):
      (w, h, d) = source_set[i]
      csv_row += [
          w,
          f'{(w + 0.5) * parameters.dx:.2f}',
          h,
          f'{(h + 0.5) * parameters.dx:.2f}',
          d,
          f'{(d + 0.5) * parameters.dx:.2f}',
      ]

    csv_row += [
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
  indexes_left = len(position_sets) - source_index - 1
  time_left = indexes_left * avg_timing
  hours = math.floor(time_left / 60 / 60)
  minutes = math.floor((time_left - hours * 60 * 60) / 60)
  seconds = time_left - hours * 60 * 60 - minutes * 60
  log.info(
      f'Elapsed: {diff:.1f}s, est: {avg_timing:.1f}s/run = {hours:}h {minutes:.0f}m {seconds:.2f}s left')

  # done, signal next iteration
  log.info('----- End iteration %d -----', source_index)
  return False


start_time = time()
for i in range(len(position_sets)):
  run_source_analysis_iteration(i)
  deviation_plot.set_data(sources_covered, deviations)

end_time = time()
diff = end_time - start_time

d_hours = math.floor(diff / 60 / 60)
d_minutes = math.floor((diff - d_hours * 60 * 60) / 60)
d_seconds = diff - d_hours * 60 * 60 - d_minutes * 60

log.info(
    f'Time elapsed for full sweep: {d_hours:}h {d_minutes:.0f}m {d_seconds:.2f}s')

if OUTPUT_VISUALS:
  for ax in [axis_best_spl, axis_deviation, axis_spl]:
    ax.relim()
    ax.autoscale_view()
  plt.tight_layout()
  plt.savefig(os.path.join(file_dir, "output", f'{output_uid}.png'), dpi=300)

if OUTPUT_CSV:
  csv_file.close()
