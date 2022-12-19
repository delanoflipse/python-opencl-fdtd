import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lib.impulses import DiracImpulseGenerator, GaussianModulatedImpulseGenerator
from lib.parameters import DT, DT_FREQUENCY, MAX_FREQUENCY, SAMPLING_FREQUENCY
from lib.scene import bell_box, shoebox_room

ITERATIONS_PER_STEP = 10

file_dir = os.path.dirname(__file__)
style_location = os.path.join(file_dir, './styles/poster.mplstyle')
plt.style.use(style_location)

axes_shape = (3, 3)
fig = plt.gcf()
ax_sim = plt.subplot2grid(axes_shape, (0, 0), rowspan=3)
ax_val = plt.subplot2grid(axes_shape, (0, 1))
ax_rec = plt.subplot2grid(axes_shape, (1, 1))
ax_max = plt.subplot2grid(axes_shape, (2, 1))
ax_fft_sig = plt.subplot2grid(axes_shape, (0, 2))
ax_fft_rec = plt.subplot2grid(axes_shape, (1, 2))

recalc_axis = [ax_val, ax_rec, ax_max, ax_fft_sig, ax_fft_rec]

# sim = bell_box(False)
# slice_h = sim.scale(1.32)
sim = shoebox_room()
slice_h = sim.scale(1.82)

sim.set_frequency(1000)
sim.set_beta(0.01)
# sim.generator = GaussianModulatedImpulseGenerator()
sim.generator = DiracImpulseGenerator()

x_data, source_data, max_data = [], [], []

slice = sim.pressure[:, slice_h, :]
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
  subset1 = int(2*SAMPLING_FREQUENCY*DT * sample_size)
  subset2 = int(2*SAMPLING_FREQUENCY*DT * sample_size*ITERATIONS_PER_STEP)

  calc_sig = np.fft.rfft(sim.signal_set, n=sample_size)
  calc_sig_abs = np.abs(calc_sig) ** 2
  calc_sig_axis = np.fft.rfftfreq(n=sample_size, d=DT)
  fft_src_plot.set_data(calc_sig_axis[:subset1], calc_sig_abs[:subset1])

  source_data.append(
      sim.pressure[sim.width_parts // 2, slice_h, sim.depth_parts // 2])

  calc_rec = np.fft.rfft(source_data, n=sample_size)
  calc_rec_abs = np.abs(calc_rec) ** 2
  calc_rec_axis = np.fft.rfftfreq(n=sample_size, d=DT*ITERATIONS_PER_STEP)
  fft_rec_plot.set_data(calc_rec_axis[:subset2], calc_rec_abs[:subset2])

  # slice = sim.analysis[:, slice_h, :]
  slice = sim.pressure[:, slice_h, :]

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


sim.setup()
ani = FuncAnimation(plt.gcf(), animate, interval=1000/60)
plt.show()
