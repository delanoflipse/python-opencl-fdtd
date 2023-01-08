from cmath import log10
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lib.impulse_generators import DiracImpulseGenerator, GaussianModulatedImpulseGenerator, GaussianMonopulseGenerator, HannWindow, WindowModulatedSinoidImpulse
from lib.parameters import SimulationParameters
from lib.scenes import bell_box, shoebox_room
from lib.simulation import Simulation

ITERATIONS_PER_STEP = 10

# get and set style
file_dir = os.path.dirname(__file__)
style_location = os.path.join(file_dir, './styles/poster.mplstyle')
plt.style.use(style_location)

# create subplots
axes_shape = (3, 4)
fig = plt.gcf()
ax_sim = plt.subplot2grid(axes_shape, (0, 0), rowspan=3)
ax_analysis = plt.subplot2grid(axes_shape, (0, 1), rowspan=2)
ax_analysis_2 = plt.subplot2grid(axes_shape, (2, 1))
ax_val = plt.subplot2grid(axes_shape, (0, 2))
ax_rec = plt.subplot2grid(axes_shape, (1, 2))
ax_max = plt.subplot2grid(axes_shape, (2, 2))
ax_fft_sig = plt.subplot2grid(axes_shape, (0, 3))
ax_fft_rec = plt.subplot2grid(axes_shape, (1, 3))
ax_an_db = plt.subplot2grid(axes_shape, (2, 3))

recalc_axis = [ax_val, ax_rec, ax_max, ax_fft_sig, ax_fft_rec, ax_an_db]

params = SimulationParameters()
params.set_max_frequency(420)

grid = bell_box(params, True)
slice_h = grid.scale(1.32)
# grid = shoebox_room(params)
# slice_h = grid.scale(1.82)

sim = Simulation(grid=grid, parameters=params)
sim.print_statistics()

# sim.generator = GaussianMonopulseGenerator(50)
# sim.generator = GaussianModulatedImpulseGenerator(50)
hann_window = HannWindow(width=0.5)
sim.generator = WindowModulatedSinoidImpulse(20, hann_window)
# sim.generator = DiracImpulseGenerator()

x_data, source_data, max_data = [], [], []
max_db_data = []

ref_slice_pressure = grid.pressure[:, slice_h, :]
pressure_image = ax_sim.imshow(ref_slice_pressure)
color_bar_pressure = plt.colorbar(pressure_image, ax=ax_sim)

ref_slice_analysis = grid.pressure[:, slice_h, :]
analysis_image = ax_analysis.imshow(ref_slice_analysis)
color_bar_analysis = plt.colorbar(analysis_image, ax=ax_analysis)
analysis_image_2 = ax_analysis_2.imshow(ref_slice_analysis)
color_bar_analysis_2 = plt.colorbar(analysis_image, ax=ax_analysis_2)

value_plot, = ax_val.plot([], [], "-")
source_plot, = ax_rec.plot([], [], "-")
max_plot, = ax_max.plot([], [], "-")

fft_src_plot, = ax_fft_sig.plot([], [], "-")
fft_rec_plot, = ax_fft_rec.plot([], [], "-")
db_over_time_min, db_over_time_max, = ax_an_db.plot([], [], [], "-")

ax_sim.set_title("Simulation")
ax_val.set_title("Input signal")
ax_rec.set_title("Signal at Receiver")
ax_fft_sig.set_title("FFT input signal")
ax_an_db.set_title("Max EWMA DB")

value_plot.axes.set_xlabel("Time (s)")
value_plot.axes.set_ylabel("Relative Pressure (Pa)")
source_plot.axes.set_xlabel("Time (s)")
source_plot.axes.set_ylabel("Relative Pressure (Pa)")
ax_sim.set_xlabel("Width Index")
ax_sim.set_ylabel("Depth Index")
color_bar_pressure.set_label("Relative Pressure(Pa)")
color_bar_analysis.set_label("Pressure(Pa)")

last_an_maximum = 1e-32
last_sim_maximum = 1e-32

fig.tight_layout()


def animate(i) -> None:
  global last_sim_maximum, last_an_maximum
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
  db_fft_sig = 20 * np.log10(50000 * calc_sig_abs[:subset1])
  fft_src_plot.set_data(calc_sig_axis[:subset1], db_fft_sig)

  source_data.append(
      grid.pressure[grid.width_parts // 2, slice_h, grid.depth_parts // 2])

  if ITERATIONS_PER_STEP == 1:
    calc_rec = np.fft.rfft(source_data, n=sample_size)
    calc_rec_abs = np.abs(calc_rec) ** 2
    calc_rec_axis = np.fft.rfftfreq(n=sample_size, d=dt_per_iteration)
    db_fft_rec = 20 * np.log10(50000 * calc_rec_abs[:subset2])
    fft_rec_plot.set_data(calc_rec_axis[:subset2], db_fft_rec)

  leq_slice = grid.analysis[:, :, :, sim.grid.analysis_keys["LEQ"]]
  l_ewma_slice = grid.analysis[:, :, :, sim.grid.analysis_keys["EWMA_L"]]
  ref_slice_analysis_leq = leq_slice[:, slice_h, :]
  ref_slice_analysis_ewma = l_ewma_slice[:, slice_h, :]
  ref_slice_pressure = grid.pressure[:, slice_h, :]

  leq_max = np.nanmax(ref_slice_analysis_leq)
  leq_min = np.nanmin(ref_slice_analysis_leq)
  an_maximum = max(abs(leq_max), abs(leq_min))
  sim_maximum = max(abs(ref_slice_pressure.min()),
                    abs(ref_slice_pressure.max()))

  max_data.append(an_maximum)
  max_db_data.append(np.nanmax(l_ewma_slice))

  pressure_image.set_data(ref_slice_pressure)
  analysis_image.set_data(ref_slice_analysis_leq)
  analysis_image_2.set_data(ref_slice_analysis_ewma)

  if last_an_maximum != an_maximum:
    analysis_image.set_clim(-an_maximum, an_maximum)
    analysis_image_2.set_clim(-an_maximum, an_maximum)
  last_an_maximum = an_maximum

  if last_sim_maximum != sim_maximum:
    pressure_image.set_clim(-sim_maximum, sim_maximum)
  last_sim_maximum = sim_maximum

  value_plot.set_data(sim.time_set, sim.signal_set)
  source_plot.set_data(x_data, source_data)
  max_plot.set_data(x_data, max_data)
  db_over_time_max.set_data(x_data, max_db_data)

  for ax in recalc_axis:
    ax.relim()
    ax.autoscale_view()

  fig.canvas.flush_events()
  print(i, sim.time, sim_maximum, an_maximum)


ani = FuncAnimation(plt.gcf(), animate, interval=1000/60)
plt.show()
