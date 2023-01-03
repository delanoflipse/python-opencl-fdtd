import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lib.impulse_generators import DiracImpulseGenerator, GaussianModulatedImpulseGenerator, GaussianMonopulseGenerator, WindowModulatedSinoidImpulse
from lib.parameters import SimulationParameters
from lib.scenes import bell_box, shoebox_room
from lib.simulation import Simulation
from numba import njit, prange

# get and set style
file_dir = os.path.dirname(__file__)
style_location = os.path.join(file_dir, './styles/poster.mplstyle')
plt.style.use(style_location)

# create subplots
axes_shape = (4, 3)
fig = plt.gcf()
ax_sim = plt.subplot2grid(axes_shape, (0, 0), rowspan=3)
ax_pres = plt.subplot2grid(axes_shape, (0, 1), rowspan=3)
ax_analysis = plt.subplot2grid(axes_shape, (0, 2), rowspan=3)

ax_max_an = plt.subplot2grid(axes_shape, (3, 0))
ax_max_pres = plt.subplot2grid(axes_shape, (3, 1))

recalc_axis = [ax_max_an, ax_max_pres]

params = SimulationParameters()
params.frequency_interval = 0.5
params.set_max_frequency(400)

grid = bell_box(params, True)
slice_h = grid.scale(1.32)

# grid = shoebox_room(params)
# slice_h = grid.scale(1.82)
# slice_h = grid.scale(.97) + 1

sim = Simulation(grid=grid, parameters=params)


it_data, max_an, max_pres = [], [], []

slice_tmp = grid.pressure[:, slice_h, :]
slice_image = ax_sim.imshow(slice_tmp, cmap="OrRd")
color_bar = plt.colorbar(slice_image, ax=ax_sim)

slice_image_2 = ax_analysis.imshow(slice_tmp, cmap="RdYlGn")
color_bar_2 = plt.colorbar(slice_image_2, ax=ax_analysis)

slice_image_3 = ax_pres.imshow(slice_tmp, cmap="seismic")
color_bar_3 = plt.colorbar(slice_image_3, ax=ax_pres)


max_pres_plot, = ax_max_pres.plot([], [], "-")
max_an_plot, = ax_max_an.plot([], [], "-")

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

ax_max_pres.set_title("Maximum Pressure")
ax_max_pres.set_xlabel("Frequency (hz)")
ax_max_pres.set_ylabel("Maximum value")

ax_max_an.set_title("Maximum Analytical value")
ax_max_an.set_xlabel("Frequency (hz)")
ax_max_an.set_ylabel("Maximum value")

fig.tight_layout()

# runtime_steps = int(0.5 / sim.parameters.dt)
runtime_steps = int(1 / sim.parameters.dt)
# runtime_steps = 100
print(f'{runtime_steps} steps per sim')

sweep_sum = sim.grid.create_grid("float64")
sweep_sum_sqr = sim.grid.create_grid("float64")
sweep_deviation = sim.grid.create_grid("float64")
sweep_ranking = sim.grid.create_grid("float64")

testing_frequencies = np.arange(sim.parameters.min_frequency,
                                sim.parameters.max_frequency, sim.parameters.frequency_interval)

# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
test_index = 0


def animate(i) -> None:
  global test_index

  if test_index == -1:
    return
  f = testing_frequencies[test_index]
  it_data.append(f)
  print(f'{f}hz')
  # sim.generator = GaussianMonopulseGenerator(f)
  # sim.generator = GaussianModulatedImpulseGenerator(f)
  sim.generator = WindowModulatedSinoidImpulse(f)
  sim.reset()
  sim.step(runtime_steps)
  run_sweep_analysis(sim.grid.analysis, sweep_sum,
                     sweep_sum_sqr, sweep_deviation, sweep_ranking, i + 1)

  slice = sim.grid.analysis[:, slice_h, :]
  slice_image.set_data(slice)
  slice_image.set_clim(0, sim.grid.analysis.max())

  slice_2 = sweep_ranking[:, slice_h, :]
  slice_image_2.set_data(slice_2)
  slice_image_2.set_clim(slice_2.min(), slice_2.max())

  slice_3 = sim.grid.pressure[:, slice_h, :]
  slice_3_max = max(abs(sim.grid.pressure.min()),
                    abs(sim.grid.pressure.max()))
  slice_image_3.set_data(slice_3)
  slice_image_3.set_clim(-slice_3_max, slice_3_max)

  max_an.append(sim.grid.analysis.max())
  max_pres.append(slice_3_max)

  max_an_plot.set_data(it_data, max_an)
  max_pres_plot.set_data(it_data, max_pres)

  for ax in recalc_axis:
    ax.relim()
    ax.autoscale_view()

  fig.canvas.flush_events()
  print(i, sim.time, slice.max(), slice_3.max())
  test_index += 1
  if test_index == testing_frequencies.size - 1:
    test_index = -1


@ njit(parallel=True)
def run_sweep_analysis(step_analysis: np.ndarray, summation: np.ndarray, sum_sqr: np.ndarray, dev: np.ndarray, ranking: np.ndarray, n: int) -> None:
  """Set neighbour flags for geometry"""
  _max = -1e99
  _min = 1e99
  for w in prange(step_analysis.shape[0]):
    for h in prange(step_analysis.shape[1]):
      for d in prange(step_analysis.shape[2]):
        v = step_analysis[w, h, d]
        _m = summation[w, h, d]
        _new_m = _m + (v - _m)/n
        summation[w, h, d] = _new_m
        sum_sqr[w, h, d] += (v - _new_m) * (v - _m)
        _dev = sum_sqr[w, h, d] / n
        _max = max(_dev, _max)
        _min = min(_dev, _min)
        dev[w, h, d] = _dev

  _range = _max - _min
  for w in prange(step_analysis.shape[0]):
    for h in prange(step_analysis.shape[1]):
      for d in prange(step_analysis.shape[2]):
        v = dev[w, h, d]
        diff = (v - _min) / _range
        r = 1 - diff
        ranking[w, h, d] = r


ani = FuncAnimation(plt.gcf(), animate, interval=1000/60)
plt.show()
