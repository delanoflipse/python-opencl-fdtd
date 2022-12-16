import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lib.scene import bell_box, shoebox_room

file_dir = os.path.dirname(__file__)
style_location = os.path.join(file_dir, './styles/poster.mplstyle')
plt.style.use(style_location)

axes_shape = (3, 2)
fig = plt.gcf()
ax_sim = plt.subplot2grid(axes_shape, (0, 0), rowspan=3)
ax_val = plt.subplot2grid(axes_shape, (0, 1))
ax_rec = plt.subplot2grid(axes_shape, (1, 1))
ax_max = plt.subplot2grid(axes_shape, (2, 1))

sim = bell_box(False)
slice_h = sim.scale(1.32)
# sim = shoebox_room()
# slice_h = sim.scale(1.82)

x_data, rec_data, source_data, max_data = [], [], [], []

slice = sim.pressure[:, slice_h, :]
y = np.arange(len(slice))
x = np.arange(len(slice[0]))
(x, y) = np.meshgrid(x, y)
slice_image = ax_sim.imshow(slice)
color_bar = plt.colorbar(slice_image, ax=ax_sim)

value_plot, = ax_val.plot(x_data, rec_data, "-")
source_plot, = ax_rec.plot(x_data, source_data, "-")
max_plot, = ax_max.plot(x_data, max_data, "-")

ax_sim.set_title("Simulation")
ax_val.set_title("Input signal")
ax_rec.set_title("Signal at Receiver")
ax_max.set_title("Max value in slice")

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

  for _ in range(2):
    sim.step()

  x_data.append(sim.time)
  rec_data.append(sim.signal)

  source_data.append(
      sim.pressure[sim.width_parts // 2, slice_h, sim.depth_parts // 2])

  # slice = sim.analysis[:, slice_h, :]
  slice = sim.pressure[:, slice_h, :]

  maximum = max(abs(slice.min()), abs(slice.max()))
  max_data.append(maximum)
  slice_image.set_data(slice)
  if last_maximum != maximum:
    slice_image.set_clim(-maximum, maximum)
  last_maximum = maximum

  value_plot.set_data(x_data, rec_data)
  source_plot.set_data(x_data, source_data)
  max_plot.set_data(x_data, max_data)

  ax_val.relim()
  ax_val.autoscale_view()

  ax_max.relim()
  ax_max.autoscale_view()
  ax_rec.relim()
  ax_rec.autoscale_view()

  fig.canvas.flush_events()
  print(i, sim.time, maximum)


sim.setup()
ani = FuncAnimation(plt.gcf(), animate, interval=1000/60)
plt.show()
