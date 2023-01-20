import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lib.impulse_generators import DiracImpulseGenerator, GaussianModulatedImpulseGenerator, GaussianMonopulseGenerator, HannWindow, WindowModulatedSinoidImpulse, SimpleSinoidGenerator
from lib.parameters import SimulationParameters
from lib.scenes import LShapedRoom, ShoeboxRoomScene, BellBoxScene, ConcertHallScene, OfficeScene
from lib.simulation import Simulation

ITERATIONS_PER_STEP: int = 20

# get and set style
file_dir = os.path.dirname(__file__)
style_location = os.path.join(file_dir, './styles/poster.mplstyle')
plt.style.use(style_location)

# create subplots
axes_shape = (3, 4)
fig = plt.gcf()
fig.set_dpi(150)
fig.set_size_inches(1920/fig.get_dpi(), 720/fig.get_dpi(), forward=True)
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

parameters = SimulationParameters()
parameters.set_oversampling(16)
parameters.set_max_frequency(200)
parameters.set_signal_frequency(400.0)

scene = ShoeboxRoomScene(parameters)
# scene = BellBoxScene(parameters, has_wall=True)
# scene = LShapedRoom(parameters)
# scene = ConcertHallScene(parameters)
# scene = OfficeScene(params)
grid = scene.build()

# SLICE_HEIGHT = grid.scale(1.32)
# SLICE_HEIGHT = grid.scale(1.82)
# SLICE_HEIGHT = grid.scale(.97)
SLICE_HEIGHT = grid.scale(scene.height / 2)


sim = Simulation(grid=grid, parameters=parameters)
sim.print_statistics()

# sim.generator = GaussianMonopulseGenerator(params.signal_frequency)
# sim.generator = GaussianModulatedImpulseGenerator(params.signal_frequency)

hann_window = HannWindow(
    width=2 / parameters.signal_frequency, end_signal=math.nan)
sim.generator = WindowModulatedSinoidImpulse(
    parameters.signal_frequency, hann_window)
# sim.generator = DiracImpulseGenerator()
# sim.generator = SimpleSinoidGenerator(parameters.signal_frequency)

grid.select_source_locations([grid.source_set[0]])
scene.rebuild()
sim.sync_read_buffers()
sim.reset()

analysis_key_index = sim.grid.analysis_keys["LEQ"]

x_data, source_data, max_data = [], [], []
max_db_data = []

temp_slice = np.ndarray(shape=(grid.width_parts, grid.depth_parts))
cmap = plt.cm.seismic.copy()
cmap.set_bad('black', 1.)
pressure_image = ax_sim.imshow(temp_slice, cmap=cmap)
color_bar_pressure = plt.colorbar(pressure_image, ax=ax_sim)

analysis_image = ax_analysis.imshow(temp_slice, cmap=cmap)
color_bar_analysis = plt.colorbar(analysis_image, ax=ax_analysis)
analysis_image_2 = ax_analysis_2.imshow(temp_slice, cmap=cmap)
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
ax_max.set_title("Chart max SPL")
ax_fft_sig.set_title("FFT input signal")
ax_fft_rec.set_title("FFT received signal")
ax_analysis.set_title("SPL (dB)")
ax_analysis_2.set_title("EMWA SPL (dB)")
ax_an_db.set_title("Max EWMA (dB)")

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
  if sim.time > 0.005:
    return
  sim.step(ITERATIONS_PER_STEP)

  x_data.append(sim.time)

  sample_size = 1024
  dt_per_iteration = parameters.dt*ITERATIONS_PER_STEP
  subset1 = int(2*parameters.sampling_frequency*parameters.dt * sample_size)
  subset2 = int(2*parameters.sampling_frequency*parameters.dt *
                sample_size*ITERATIONS_PER_STEP)

  calc_sig = np.fft.rfft(sim.signal_set, n=sample_size)
  calc_sig_abs = np.abs(calc_sig) ** 2
  calc_sig_axis = np.fft.rfftfreq(n=sample_size, d=parameters.dt)
  db_fft_sig = 20 * np.log10(50000 * calc_sig_abs[:subset1])
  fft_src_plot.set_data(calc_sig_axis[:subset1], db_fft_sig)

  source_data.append(
      grid.pressure[grid.width_parts // 2, SLICE_HEIGHT, grid.depth_parts // 2])

  if ITERATIONS_PER_STEP == 1:
    calc_rec = np.fft.rfft(source_data, n=sample_size)
    calc_rec_abs = np.abs(calc_rec) ** 2
    calc_rec_axis = np.fft.rfftfreq(n=sample_size, d=dt_per_iteration)
    db_fft_rec = 20 * np.log10(50000 * calc_rec_abs[:subset2])
    fft_rec_plot.set_data(calc_rec_axis[:subset2], db_fft_rec)

  leq_slice = grid.analysis[:, :, :, analysis_key_index]
  l_ewma_slice = grid.analysis[:, :, :, sim.grid.analysis_keys["EWMA_L"]]
  ref_slice_analysis_leq = leq_slice[:, SLICE_HEIGHT, :]
  ref_slice_analysis_ewma = l_ewma_slice[:, SLICE_HEIGHT, :]
  ref_slice_pressure = grid.pressure[:, SLICE_HEIGHT, :]

  leq_max = np.nanmax(leq_slice)
  leq_min = np.nanmin(leq_slice)
  an_maximum = max(abs(leq_max), abs(leq_min))
  sim_maximum = max(abs(np.nanmax(ref_slice_pressure)),
                    abs(np.nanmin(ref_slice_pressure)))

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
