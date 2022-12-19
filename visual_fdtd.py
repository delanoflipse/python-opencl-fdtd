import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lib.impulse_generators import DiracImpulseGenerator, GaussianModulatedImpulseGenerator
from lib.parameters import SimulationParameters
from lib.scenes import bell_box, shoebox_room
from lib.simulation import Simulation

ITERATIONS_PER_STEP = 10

# get and set style
file_dir = os.path.dirname(__file__)
style_location = os.path.join(file_dir, './styles/poster.mplstyle')
plt.style.use(style_location)

# create subplots
axes_shape = (3, 3)
fig = plt.gcf()
ax_sim = plt.subplot2grid(axes_shape, (0, 0), rowspan=3)
ax_val = plt.subplot2grid(axes_shape, (0, 1))
ax_rec = plt.subplot2grid(axes_shape, (1, 1))
ax_max = plt.subplot2grid(axes_shape, (2, 1))
ax_fft_sig = plt.subplot2grid(axes_shape, (0, 2))
ax_fft_rec = plt.subplot2grid(axes_shape, (1, 2))

recalc_axis = [ax_val, ax_rec, ax_max, ax_fft_sig, ax_fft_rec]

params = SimulationParameters()
params.set_max_frequency(200)

# grid = bell_box(params, False)
# slice_h = grid.scale(1.32)
grid = shoebox_room(params)
slice_h = grid.scale(1.82)

sim = Simulation(grid=grid, parameters=params)

sim.generator = GaussianModulatedImpulseGenerator(20, 0.03)
# sim.generator = DiracImpulseGenerator()

x_data, source_data, max_data = [], [], []

slice = grid.pressure[:, slice_h, :]
y = np.arange(len(slice))
x = np.arange(len(slice[0]))
(x, y) = np.meshgrid(x, y)
slice_image = ax_sim.imshow(slice)
color_bar = plt.colorbar(slice_image, ax=ax_sim)

value_plot, = ax_val.plot([], [], "-")
source_plot, = ax_rec.plot([], [], "-")
max_plot, = ax_max.plot([], [], "-")

fft_src_plot, = ax_fft_sig.plot([], [], "-")
fft_rec_plot, = ax_fft_rec.plot([], [], "-")

ax_sim.set_title("Simulation")
ax_val.set_title("Input signal")
ax_rec.set_title("Signal at Receiver")
ax_fft_sig.set_title("FFT input signal")

value_plot.axes.set_xlabel("Time (s)")
value_plot.axes.set_ylabel("Relative Pressure (Pa)")
source_plot.axes.set_xlabel("Time (s)")
source_plot.axes.set_ylabel("Relative Pressure (Pa)")
ax_sim.set_xlabel("Width Index")
ax_sim.set_ylabel("Depth Index")
color_bar.set_label("Relative Pressure(Pa)")
fig.tight_layout()

maximum = 1e-32
last_maximum = 1e-32


def animate(i) -> None:
  global maximum, last_maximum

  sim.step(ITERATIONS_PER_STEP)

  x_data.append(sim.time)

  sample_size = 1024
  dt_per_iteration = params.dt*ITERATIONS_PER_STEP
  subset1 = int(2*params.sampling_frequency*params.dt * sample_size)
  subset2 = int(2*params.sampling_frequency*params.dt *
                sample_size*ITERATIONS_PER_STEP)

  calc_sig = np.fft.rfft(sim.signal_set, n=sample_size)
  calc_sig_abs = np.abs(calc_sig) ** 2
  calc_sig_axis = np.fft.rfftfreq(n=sample_size, d=params.dt)
  fft_src_plot.set_data(calc_sig_axis[:subset1], calc_sig_abs[:subset1])

  source_data.append(
      grid.pressure[grid.width_parts // 2, slice_h, grid.depth_parts // 2])

  if ITERATIONS_PER_STEP == 1:
    calc_rec = np.fft.rfft(source_data, n=sample_size)
    calc_rec_abs = np.abs(calc_rec) ** 2
    calc_rec_axis = np.fft.rfftfreq(n=sample_size, d=dt_per_iteration)
    fft_rec_plot.set_data(calc_rec_axis[:subset2], calc_rec_abs[:subset2])

  slice = grid.analysis[:, slice_h, :]
  # slice = grid.pressure[:, slice_h, :]

  maximum = max(abs(slice.min()), abs(slice.max()))
  max_data.append(maximum)
  slice_image.set_data(slice)
  if last_maximum != maximum:
    slice_image.set_clim(-maximum, maximum)
  last_maximum = maximum

  value_plot.set_data(sim.time_set, sim.signal_set)
  source_plot.set_data(x_data, source_data)
  max_plot.set_data(x_data, max_data)

  for ax in recalc_axis:
    ax.relim()
    ax.autoscale_view()

  fig.canvas.flush_events()
  print(i, sim.time, maximum)


ani = FuncAnimation(plt.gcf(), animate, interval=1000/60)
plt.show()
