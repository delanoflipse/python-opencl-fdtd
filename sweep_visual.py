import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lib.impulse_generators import DiracImpulseGenerator, GaussianModulatedImpulseGenerator, GaussianMonopulseGenerator
from lib.parameters import SimulationParameters
from lib.scenes import bell_box, shoebox_room
from lib.simulation import Simulation
from numba import njit, prange

# get and set style
file_dir = os.path.dirname(__file__)
style_location = os.path.join(file_dir, './styles/poster.mplstyle')
plt.style.use(style_location)

# create subplots
axes_shape = (3, 3)
fig = plt.gcf()
ax_sim = plt.subplot2grid(axes_shape, (0, 0), rowspan=3)
ax_pres = plt.subplot2grid(axes_shape, (0, 1), rowspan=3)
ax_analysis = plt.subplot2grid(axes_shape, (0, 2), rowspan=3)

recalc_axis = []

params = SimulationParameters()
params.set_max_frequency(200)

# grid = bell_box(params, True)
# slice_h = grid.scale(1.32)
grid = shoebox_room(params)
# slice_h = grid.scale(1.82)
slice_h = grid.scale(.97) + 1

sim = Simulation(grid=grid, parameters=params)

slice_tmp = grid.pressure[:, slice_h, :]
slice_image = ax_sim.imshow(slice_tmp, cmap="OrRd")
color_bar = plt.colorbar(slice_image, ax=ax_sim)

slice_image_2 = ax_analysis.imshow(slice_tmp, cmap="OrRd")
color_bar_2 = plt.colorbar(slice_image_2, ax=ax_analysis)

slice_image_3 = ax_pres.imshow(slice_tmp, cmap="seismic")
color_bar_3 = plt.colorbar(slice_image_3, ax=ax_pres)

ax_sim.set_title("Simulation")
ax_sim.set_xlabel("Width Index")
ax_sim.set_ylabel("Depth Index")

ax_analysis.set_title("Analysis")
ax_analysis.set_xlabel("Width Index")
ax_analysis.set_ylabel("Depth Index")

ax_pres.set_title("Pressure check")
ax_pres.set_xlabel("Width Index")
ax_pres.set_ylabel("Depth Index")

color_bar.set_label("Pressure RMS")
color_bar_2.set_label("Quality")
color_bar_3.set_label("Relative Pressure(Pa)")
fig.tight_layout()

# runtime_steps = int(0.5 / sim.parameters.dt)
runtime_steps = int(0.4 / sim.parameters.dt)
# runtime_steps = 100
print(f'{runtime_steps} steps per sim')

sweep_sum = sim.grid.create_grid("float64")
sweep_sum_sqr = sim.grid.create_grid("float64")
sweep_deviation = sim.grid.create_grid("float64")

iterator = iter(range(sim.parameters.min_frequency,
                sim.parameters.max_frequency, sim.parameters.frequency_interval))
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance


def animate(i) -> None:
  f = next(iterator, -1)
  if f == -1:
    return
  print(f'{f}hz')
  # sim.generator = GaussianMonopulseGenerator(f)
  sim.generator = GaussianModulatedImpulseGenerator(f)
  sim.reset()
  sim.step(runtime_steps)
  run_sweep_analysis(sim.grid.analysis, sweep_sum,
                     sweep_sum_sqr, sweep_deviation, i + 1)

  slice = sim.grid.analysis[:, slice_h, :]
  slice_image.set_data(slice)
  slice_image.set_clim(slice.min(), slice.max())

  slice_2 = sweep_deviation[:, slice_h, :]
  slice_image_2.set_data(slice_2)
  slice_image_2.set_clim(slice_2.min(), slice_2.max())

  slice_3 = sim.grid.pressure[:, slice_h, :]
  slice_3_max = max(abs(slice_3.min()),
                    abs(slice_3.max()))
  slice_image_3.set_data(slice_3)
  slice_image_3.set_clim(-slice_3_max, slice_3_max)

  for ax in recalc_axis:
    ax.relim()
    ax.autoscale_view()

  fig.canvas.flush_events()
  print(i, sim.time, slice.max(), slice_3.max())


@ njit(parallel=True)
def run_sweep_analysis(step_analysis: np.ndarray, summation: np.ndarray, sum_sqr: np.ndarray, dev: np.ndarray, n: int) -> None:
  """Set neighbour flags for geometry"""
  for w in prange(step_analysis.shape[0]):
    for h in prange(step_analysis.shape[1]):
      for d in prange(step_analysis.shape[2]):
        v = step_analysis[w, h, d]
        summation[w, h, d] += v
        _sum = summation[w, h, d]
        sum_sqr[w, h, d] += v * v
        dev[w, h, d] = (sum_sqr[w, h, d] - (_sum * _sum)/n)/(n-1)


ani = FuncAnimation(plt.gcf(), animate, interval=1000/60)
plt.show()
