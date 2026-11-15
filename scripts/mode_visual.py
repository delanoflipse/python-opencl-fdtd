"""
Visualise how the field at a single frequency develops in a room (room modes).

Every few iterations the pressure on a horizontal slice is mixed with
exp(-j*2*pi*f*t) and accumulated per cell: a running single-bin DFT, or
"lock-in" detector. With an exponential window (--window-periods) the result
is the *current* complex amplitude at f, so you can watch the standing-wave
pattern build up (sine excitation) or ring down (impulse excitation).

Panels:
  - instantaneous pressure on the slice
  - amplitude at f (dB, relative to the slice maximum)
  - signed mode shape: Re(X) aligned to the dominant phase, nodal lines are
    where the colour crosses white
  - amplitude at f over time at the receiver and the slice maximum
"""
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fdtd.styles import use_style
from fdtd.impulse_generators import GENERATORS, create_generator
from fdtd.parameters import SimulationParameters
from fdtd.physical_constants import C_AIR
from fdtd.scenes import SCENES, create_scene
from fdtd.simulation import Simulation

parser = argparse.ArgumentParser(
    description="Visualise the field of a single frequency on a horizontal slice")
parser.add_argument("-s", "--scene", default="shoebox", choices=SCENES)
parser.add_argument("-f", "--frequency", type=float, default=60.0,
                    help="frequency to analyse in Hz (default: 60)")
parser.add_argument("-g", "--generator", default="sine", choices=GENERATORS,
                    help="excitation, built at --frequency (default: sine, "
                    "use e.g. monopulse to see the ring-down)")
parser.add_argument("--max-frequency", type=float, default=None,
                    help="grid design frequency, sets the cell size "
                    "(default: max(200, 2 * frequency))")
parser.add_argument("--window-periods", type=float, default=8.0,
                    help="exponential window length in periods of f; "
                    "0 accumulates from t=0 (default: 8)")
parser.add_argument("--samples-per-period", type=int, default=16,
                    help="DFT samples per period of f (default: 16)")
parser.add_argument("--samples-per-frame", type=int, default=4,
                    help="DFT samples per animation frame (default: 4)")
parser.add_argument("--slice-height", type=float, default=None,
                    help="height of the slice in m (default: half the room)")
parser.add_argument("--source", type=float, nargs=3, metavar=("W", "H", "D"),
                    default=None, help="source position in m "
                    "(default: the scene's first source position)")
arguments = parser.parse_args()

FREQUENCY: float = arguments.frequency
OMEGA = 2.0 * math.pi * FREQUENCY

use_style("poster")

# ---- simulation ----
parameters = SimulationParameters()
parameters.set_max_frequency(
    arguments.max_frequency or max(200.0, 2.0 * FREQUENCY))
parameters.set_signal_frequency(FREQUENCY)
parameters.set_scheme(1.0, 1 / 4, 1 / 16)  # IWB

scene = create_scene(arguments.scene, parameters)
grid = scene.build()
if arguments.source is not None:
  grid.select_source_locations([grid.pos(*arguments.source)])
elif grid.source_set:
  grid.select_source_locations([grid.source_set[0]])
else:
  grid.select_source_locations(
      [grid.pos(scene.width / 4, scene.height / 4, scene.depth / 4)])

slice_height_m = arguments.slice_height if arguments.slice_height is not None \
    else scene.height / 2
SLICE_HEIGHT = min(grid.scale(slice_height_m), grid.height_parts - 1)
RECEIVER = (grid.width_parts // 2, grid.depth_parts // 2)

sim = Simulation(grid=grid, parameters=parameters)
sim.print_statistics()
sim.generator = create_generator(arguments.generator, FREQUENCY)
scene.rebuild()
sim.sync_read_buffers()
sim.reset()

# iterations between DFT samples
STEPS_PER_SAMPLE = max(
    1, int(parameters.dt_hz / (arguments.samples_per_period * FREQUENCY)))
sample_dt = STEPS_PER_SAMPLE * parameters.dt
if arguments.window_periods > 0:
  window_time = arguments.window_periods / FREQUENCY
  DECAY = math.exp(-sample_dt / window_time)
else:
  DECAY = 1.0
print(f"[Modes] f={FREQUENCY:.1f}Hz, sampling every {STEPS_PER_SAMPLE} "
      f"iterations ({1 / sample_dt:.0f}Hz), window decay {DECAY:.4f}/sample")

# ---- analytic modes of the bounding box, for reference ----
analytic_modes = []
for i in range(8):
  for j in range(8):
    for k in range(8):
      if i == j == k == 0:
        continue
      f = (C_AIR / 2) * math.sqrt((i / scene.width) ** 2 +
                                  (j / scene.height) ** 2 + (k / scene.depth) ** 2)
      analytic_modes.append((f, (i, j, k)))
analytic_modes.sort()
nearest = sorted(analytic_modes, key=lambda m: abs(m[0] - FREQUENCY))[:3]
nearest_str = ", ".join(f"{f:.1f}Hz {ijk}" for f, ijk in nearest)
print(f"[Modes] nearest box modes (w, h, d): {nearest_str}")

# ---- state ----
slice_shape = (grid.width_parts, grid.depth_parts)
dft = np.zeros(slice_shape, dtype=np.complex128)
weight = 0.0
previous_shape = None
t_data, receiver_data, max_data = [], [], []

# ---- figure ----
fig = plt.gcf()
fig.set_dpi(150)
fig.set_size_inches(1920 / fig.get_dpi(), 960 / fig.get_dpi(), forward=True)
fig.suptitle(f"{arguments.scene} @ {FREQUENCY:.1f}Hz, slice h={slice_height_m:.2f}m"
             f"  |  nearest box modes (w,h,d): {nearest_str}")
ax_p = fig.add_subplot(2, 3, 1)
ax_amp = fig.add_subplot(2, 3, 2)
ax_shape = fig.add_subplot(2, 3, 3)
ax_line = fig.add_subplot(2, 1, 2)

cmap_signed = plt.cm.seismic.copy()
cmap_signed.set_bad("black", 1.0)
cmap_amp = plt.cm.magma.copy()
cmap_amp.set_bad("black", 1.0)

empty = np.zeros(slice_shape)
pressure_image = ax_p.imshow(empty, cmap=cmap_signed)
amp_image = ax_amp.imshow(empty, cmap=cmap_amp, vmin=-40, vmax=0)
shape_image = ax_shape.imshow(empty, cmap=cmap_signed, vmin=-1, vmax=1)
plt.colorbar(pressure_image, ax=ax_p).set_label("Pressure (Pa)")
plt.colorbar(amp_image, ax=ax_amp).set_label("dB re slice max")
plt.colorbar(shape_image, ax=ax_shape).set_label("Normalised")
for ax in (ax_p, ax_amp, ax_shape):
  ax.set_xlabel("Depth index")
  ax.set_ylabel("Width index")
  ax.plot(RECEIVER[1], RECEIVER[0], "x", color="lime")
ax_p.set_title("Pressure")
ax_amp.set_title(f"Amplitude @ {FREQUENCY:.1f}Hz")
ax_shape.set_title("Mode shape (signed)")

receiver_plot, = ax_line.plot([], [], "-", label="receiver (x)")
max_plot, = ax_line.plot([], [], "-", label="slice max")
ax_line.set_title(f"Amplitude @ {FREQUENCY:.1f}Hz over time")
ax_line.set_xlabel("Time (s)")
ax_line.set_ylabel("Amplitude (Pa)")
ax_line.legend(loc="upper left")
fig.tight_layout()


def animate(frame: int) -> None:
  global dft, weight, previous_shape

  for _ in range(arguments.samples_per_frame):
    sim.step(STEPS_PER_SAMPLE)
    # grid.pressure holds the field of the last iteration, at sim.time - dt
    t = sim.time - parameters.dt
    pressure_slice = grid.pressure[:, SLICE_HEIGHT, :]
    dft *= DECAY
    dft += pressure_slice * np.exp(-1j * OMEGA * t)
    weight = weight * DECAY + 1.0

  # amplitude of the sinusoid at f
  complex_amplitude = dft * (2.0 / weight)
  amplitude = np.abs(complex_amplitude)
  amp_max = np.nanmax(amplitude)
  if not amp_max > 0:
    return

  # rotate so the field is as real as possible, then keep the sign stable
  phase = 0.5 * np.angle(np.nansum(complex_amplitude ** 2))
  shape = np.real(complex_amplitude * np.exp(-1j * phase)) / amp_max
  if previous_shape is not None and np.nansum(shape * previous_shape) < 0:
    shape = -shape
  previous_shape = shape

  pressure_slice = grid.pressure[:, SLICE_HEIGHT, :]
  p_max = max(np.nanmax(np.abs(pressure_slice)), 1e-32)
  pressure_image.set_data(pressure_slice)
  pressure_image.set_clim(-p_max, p_max)
  with np.errstate(divide="ignore"):
    amp_image.set_data(20.0 * np.log10(amplitude / amp_max))
  shape_image.set_data(shape)

  t_data.append(sim.time)
  receiver_data.append(amplitude[RECEIVER])
  max_data.append(amp_max)
  receiver_plot.set_data(t_data, receiver_data)
  max_plot.set_data(t_data, max_data)
  ax_line.relim()
  ax_line.autoscale_view()

  print(f"{frame}\tt={sim.time:.4f}s\tmax={amp_max:.4g}\t"
        f"receiver={amplitude[RECEIVER]:.4g}")


ani = FuncAnimation(fig, animate, interval=1000 / 60, cache_frame_data=False)
plt.show()
