"""
Perform a frequency sweep and analyse the inter-simulation results to
find the optimal position based on standard deviation
"""

import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lib.analysis.frequency_sweep import get_avg_spl, run_sweep_analysis
from lib.charts.line_chart import LineChart
from lib.impulse_generators import DiracImpulseGenerator, GaussianModulatedImpulseGenerator, GaussianMonopulseGenerator, WindowModulatedSinoidImpulse, SimpleSinoidGenerator
from lib.math.decibel_weightings import get_a_weighting
from lib.math.octaves import get_octaval_center_frequencies
from lib.parameters import SimulationParameters
from lib.scenes import ShoeboxRoomScene, BellBoxScene, ConcertHallScene, CuboidReferenceScene, OfficeScene
from lib.simulation import Simulation

# ---- Simulation ----
parameters = SimulationParameters()
# parameters.set_oversampling(12)
parameters.set_max_frequency(200)

SIM_TIME = 0.35
runtime_steps = int(SIM_TIME / parameters.dt)
testing_frequencies = get_octaval_center_frequencies(20, 200, fraction=24)

# scene = ShoeboxRoomScene(parameters)
# scene = BellBoxScene(parameters, has_wall=True)
# scene = CuboidReferenceScene(parameters)
# scene = ConcertHallScene(parameters)
scene = OfficeScene(parameters)
grid = scene.build()

# SLICE_HEIGHT = grid.scale(1.82)
SLICE_HEIGHT = grid.scale(scene.height / 2)
# SLICE_HEIGHT = grid.scale(.97)
# SLICE_HEIGHT = grid.scale(.97) + 1

grid.select_source_locations([grid.source_set[0]])

sim = Simulation(grid=grid, parameters=parameters)
sim.print_statistics()
print(f'{runtime_steps} steps per sim, {testing_frequencies.size} frequencies')

# ---- Chart & Axis ----
# get and set style
file_dir = os.path.dirname(__file__)
plt.style.use(os.path.join(file_dir, './styles/poster.mplstyle'))

# create subplot axis
axes_shape = (4, 3)
fig = plt.gcf()
fig.set_dpi(150)
fig.set_size_inches(1920/fig.get_dpi(), 1080/fig.get_dpi(), forward=True)
ax_sim = plt.subplot2grid(axes_shape, (0, 0), rowspan=3)
ax_pres = plt.subplot2grid(axes_shape, (0, 1), rowspan=3)
ax_analysis = plt.subplot2grid(axes_shape, (0, 2), rowspan=3)

axis_maximum_spl = plt.subplot2grid(axes_shape, (3, 0))
axis_maximum_pressure = plt.subplot2grid(axes_shape, (3, 1))
axis_mean_spl = plt.subplot2grid(axes_shape, (3, 2))


# datasets
frequency_list = []
mean_spl, max_spl, min_spl = [], [], []
max_pres = []
max_an, min_an = [], []

# charts
slice_tmp = grid.pressure[:, SLICE_HEIGHT, :]
slice_image = ax_sim.imshow(slice_tmp, cmap="OrRd")
color_bar = plt.colorbar(slice_image, ax=ax_sim)

slice_image_2 = ax_analysis.imshow(slice_tmp, cmap="RdYlGn")
color_bar_2 = plt.colorbar(slice_image_2, ax=ax_analysis)

slice_image_3 = ax_pres.imshow(slice_tmp, cmap="seismic")
color_bar_3 = plt.colorbar(slice_image_3, ax=ax_pres)

maximum_spl_chart = LineChart(axis_maximum_spl, "hz_spl", "Max/Min SPL value")
maximum_spl_chart.add_dataset(frequency_list, max_an)
maximum_spl_chart.add_dataset(frequency_list, min_an)

maximum_pressure_chart = LineChart(
    axis_maximum_pressure, "hz_pa", "Maximum Pressure")
maximum_pressure_chart.add_dataset(frequency_list, max_pres)

spl_values_chart = LineChart(
    axis_mean_spl, "hz_spl", "SPL average and range")
spl_values_chart.add_dataset(frequency_list, mean_spl)
spl_values_chart.add_dataset(frequency_list, max_spl, "--")
spl_values_chart.add_dataset(frequency_list, min_spl, "--")

charts = [
    maximum_spl_chart,
    maximum_pressure_chart,
    spl_values_chart,
]


ax_sim.set_title("Simulation")
ax_sim.set_xlabel("Width Index")
ax_sim.set_ylabel("Depth Index")

ax_analysis.set_title("Analysis")
ax_analysis.set_xlabel("Width Index")
ax_analysis.set_ylabel("Depth Index")

ax_pres.set_title("Pressure check")
ax_pres.set_xlabel("Width Index")
ax_pres.set_ylabel("Depth Index")

color_bar.set_label("SPL (dB)")
color_bar_2.set_label("Quality")
color_bar_3.set_label("Relative Pressure(Pa)")


fig.tight_layout()

# ---- Analysis ----
sweep_sum = sim.grid.create_grid("float64")
sweep_sum_sqr = sim.grid.create_grid("float64")
sweep_deviation = sim.grid.create_grid("float64")
sweep_ranking = sim.grid.create_grid("float64")

# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
test_index = 0


def animate(i) -> None:
  global test_index

  if test_index == -1:
    return
  f = testing_frequencies[test_index]
  parameters.set_signal_frequency(f)
  frequency_list.append(f)
  # a_weighting = get_a_weighting(f)
  # sim.generator = GaussianMonopulseGenerator(f)
  # sim.generator = GaussianModulatedImpulseGenerator(f)
  # sim.generator = WindowModulatedSinoidImpulse(f)
  sim.generator = SimpleSinoidGenerator(parameters.signal_frequency)

  scene.rebuild()
  sim.sync_read_buffers()
  sim.reset()
  sim.step(runtime_steps)
  analysis_key_index = sim.grid.analysis_keys["LEQ"]
  run_sweep_analysis(sim.grid.analysis, sweep_sum, sweep_sum_sqr,
                     sweep_deviation, sweep_ranking, analysis_key_index, i + 1)
  avg_spl, min_spl_value, max_spl_value = get_avg_spl(
      sim.grid.analysis, sim.grid.geometry, analysis_key_index)

  leq_analysis = grid.analysis[:, :, :, analysis_key_index]
  max_l_eq = np.nanmax(leq_analysis)
  min_l_eq = np.nanmin(leq_analysis)
  slice_leq = leq_analysis[:, SLICE_HEIGHT, :]
  slice_image.set_data(slice_leq)
  slice_image.set_clim(min_l_eq, max_l_eq)

  slice_2 = sweep_ranking[:, SLICE_HEIGHT, :]
  slice_image_2.set_data(slice_2)
  slice_image_2.set_clim(slice_2.min(), slice_2.max())

  slice_3 = sim.grid.pressure[:, SLICE_HEIGHT, :]
  slice_3_max = max(abs(sim.grid.pressure.min()),
                    abs(sim.grid.pressure.max()))
  slice_image_3.set_data(slice_3)
  slice_image_3.set_clim(-slice_3_max, slice_3_max)

  max_an.append(max_l_eq)
  min_an.append(min_l_eq)
  max_pres.append(slice_3_max)
  mean_spl.append(avg_spl)
  min_spl.append(min_spl_value)
  max_spl.append(max_spl_value)

  for chart in charts:
    chart.render()

  fig.canvas.flush_events()
  fig.tight_layout()
  print(i, f'{f}hz', sim.time, avg_spl)
  test_index += 1

  # End simulation if no frequencies are left
  if test_index == testing_frequencies.size:
    test_index = -1


ani = FuncAnimation(plt.gcf(), animate, interval=1000/60)
plt.show()
