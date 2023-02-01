
import math
import os
from datetime import datetime
from lib.impulse_generators import GaussianMonopulseGenerator, SimpleSinoidGenerator

import numpy as np
from lib.grid import SimulationGrid
from lib.parameters import SimulationParameters
from lib.simulation import Simulation
from lib.grid import WALL_FLAG

import matplotlib.pyplot as plt


parameters = SimulationParameters()
parameters.set_oversampling(16)
# parameters.set_max_frequency(400)
parameters.set_max_frequency(800)
parameters.set_signal_frequency(118)
# parameters.set_signal_frequency(50)

# scene = OfficeScene(parameters)
scene_size = 4.0
scene_height = 6.0
scene_shape = (scene_size, scene_height, scene_size)

print(f'creating grid')
grid = SimulationGrid(scene_shape, parameters)

print(f'filling grid')
grid.fill_region(
    w_min=(2/5) * scene_size,
    w_max=(3/5) * scene_size,
    d_min=(2/5) * scene_size,
    d_max=(3/5) * scene_size,
    geometry_flag=WALL_FLAG,
    beta=0.5,
)

print(f'building grid')
grid.build()
print(f'setting source')
# grid.select_source_locations(
#     [grid.pos(scene_size / 5, scene_height / 3, scene_size / 2)])
grid.select_source_locations(
    [grid.pos(scene_size / 5, scene_height / 2.5, scene_size / 2)])


print(f'creating simg')
sim = Simulation(grid=grid, parameters=parameters)
sim.print_statistics()

# sim.generator = GaussianMonopulseGenerator(parameters.signal_frequency)
sim.generator = SimpleSinoidGenerator(parameters.signal_frequency)

# get and set style
plt.style.use('./styles/poster.mplstyle')

axes_shape = (1, 2)
fig = plt.gcf()
fig.set_dpi(300)
fig.set_size_inches(2400/fig.get_dpi(), 1000/fig.get_dpi(), forward=True)

axis_pressure = plt.subplot2grid(axes_shape, (0, 0))
axis_spl = plt.subplot2grid(axes_shape, (0, 1))

SLICE_HEIGHT = grid.scale(scene_height / 2)
SPL_INDEX = sim.grid.analysis_keys["LEQ"]

slice_image_pressure = axis_pressure.imshow(
    grid.pressure[:, SLICE_HEIGHT, :], cmap="seismic", interpolation="spline36")
plt.colorbar(slice_image_pressure, ax=axis_pressure, label="Pressure (Pa)")

slice_image_spl = axis_spl.imshow(
    grid.analysis[:, SLICE_HEIGHT, :, SPL_INDEX], cmap="OrRd", interpolation="spline36")
plt.colorbar(slice_image_spl, ax=axis_spl, label="Sound Pressure Level (dB)")

for ax in [axis_pressure, axis_spl]:
  ax.set_xlabel("Depth index")
  ax.set_ylabel("Width index")


def update_pressure_slice():
  pressure_slice = grid.pressure[:, SLICE_HEIGHT, :]
  _max = np.nanmax(pressure_slice)
  _min = np.nanmin(pressure_slice)
  _range = max(abs(_min), abs(_max))

  slice_image_pressure.set_data(pressure_slice)
  slice_image_pressure.set_clim(-_range, _range)

  spl_slice = grid.analysis[:, SLICE_HEIGHT, :, SPL_INDEX]
  _max = np.nanmax(spl_slice)
  _min = np.nanmin(spl_slice)

  slice_image_spl.set_data(spl_slice)
  slice_image_spl.set_clim(_min, _max)


output_uid = f'{datetime.now().strftime("%Y-%m-%d %H_%M_%S")} {parameters.signal_frequency}f'

INTERVAL1 = max(1, int(0.00025 / parameters.dt))
for i in range(50):
  sim.step(INTERVAL1)
  print(sim.iteration)
  update_pressure_slice()
  fig.canvas.draw()
  fig.canvas.flush_events()
  plt.tight_layout()
  plt.show(block=False)
  plt.savefig(
      f"output/capture/{output_uid}-{sim.iteration}i-{sim.time}t.png", dpi=400)
  
INTERVAL2 = max(1, int(0.5 / parameters.dt))
for i in range(5):
  sim.step(INTERVAL2)
  print(sim.iteration)
  update_pressure_slice()
  fig.canvas.draw()
  fig.canvas.flush_events()
  plt.tight_layout()
  plt.show(block=False)
  plt.savefig(
      f"output/capture/{output_uid}-{sim.iteration}i-{sim.time}t.png", dpi=400)

plt.show()
# sim.step(int(3.0 / parameters.dt) - sim.iteration)
# update_pressure_slice()
# plt.savefig(f"output/capture/3s-{sim.iteration}i-{sim.time}t.png", dpi=45)
