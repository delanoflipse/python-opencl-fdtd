"""
Frequency-domain view of a scene: solve the steady-state field at one frequency
directly (fdtd.frequency_domain, no time stepping) and show the room mode shape.

Panels:
  - amplitude (dB re slice max) on a horizontal slice
  - signed mode shape on that slice: Re(P) rotated to the dominant phase,
    nodal lines are where the colour crosses white
  - signed mode shape on a vertical (width x height) slice through the source
  - with --sweep: the frequency response at the receiver, with the
    analytic box modes marked. Click it to solve at that frequency.

Keys: left/right change the frequency by --step Hz, up/down move the slice.
"""
import argparse
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from fdtd.styles import use_style
from fdtd import frequency_domain
from fdtd.grid import LISTENER_FLAG
from fdtd.parameters import SimulationParameters
from fdtd.physical_constants import C_AIR
from fdtd.scenes import SCENES, create_scene

parser = argparse.ArgumentParser(
    description="Solve and visualise the steady-state field at a single frequency")
parser.add_argument("-s", "--scene", default="shoebox", choices=SCENES)
parser.add_argument("-f", "--frequency", type=float, default=60.0,
                    help="frequency to solve in Hz (default: 60)")
parser.add_argument("--max-frequency", type=float, default=200.0,
                    help="grid design frequency, sets the cell size (default: 200)")
parser.add_argument("--oversampling", type=float, default=16,
                    help="cells per wavelength of --max-frequency (default: 16)")
parser.add_argument("--sweep", type=float, nargs=2, metavar=("FMIN", "FMAX"), default=None,
                    help="also compute the frequency response at the receiver")
parser.add_argument("--points", type=int, default=100,
                    help="number of sweep frequencies (default: 100)")
parser.add_argument("--step", type=float, default=1.0,
                    help="frequency step of the left/right keys in Hz (default: 1)")
parser.add_argument("--slice-height", type=float, default=None,
                    help="height of the slice in m (default: half the room)")
parser.add_argument("--source", type=float, nargs=3, metavar=("W", "H", "D"), default=None,
                    help="source position in m (default: the scene's first source position)")
parser.add_argument("--beta", type=float, default=None,
                    help="override the absorption of the six outer walls (the scenes use ~0.1)")
parser.add_argument("--hard-source", action="store_true",
                    help="clamp the source cell like the FDTD kernel instead of a soft source")
arguments = parser.parse_args()

use_style("poster")

# ---- grid and operator ----
parameters = SimulationParameters()
parameters.set_oversampling(arguments.oversampling)
parameters.set_max_frequency(arguments.max_frequency)
parameters.set_scheme(1.0, 1 / 4, 1 / 16)  # IWB

scene = create_scene(arguments.scene, parameters)
grid = scene.build()
scene.rebuild()
if arguments.beta is not None:
  # after scene.rebuild(), which would reset the edge betas in mark_regions
  grid.edge_betas.set_all(arguments.beta)
  grid.populate_inner_beta()

if arguments.source is not None:
  source = grid.pos(*arguments.source)
elif grid.source_set:
  source = grid.source_set[0]
else:
  source = grid.pos(scene.width / 4, scene.height / 4, scene.depth / 4)

listeners = np.argwhere(grid.geometry & LISTENER_FLAG)
receiver = tuple(int(v) for v in listeners[0]) if len(listeners) else \
    (grid.width_parts // 2, grid.height_parts // 2, grid.depth_parts // 2)

start = time.perf_counter()
operator = frequency_domain.build_operator(grid, parameters)
print(f"[FDFD] grid {grid.grid_shape}, {operator.size} unknowns, "
      f"dx={parameters.dx * 1000:.1f}mm, built in {time.perf_counter() - start:.2f}s")
print(f"[FDFD] source {source}, receiver {receiver}")

# ---- analytic modes of the bounding box, for reference ----
analytic_modes = []
for i in range(10):
  for j in range(10):
    for k in range(10):
      if i == j == k == 0:
        continue
      f = (C_AIR / 2) * math.sqrt((i / scene.width) ** 2 +
                                  (j / scene.height) ** 2 + (k / scene.depth) ** 2)
      analytic_modes.append((f, (i, j, k)))
analytic_modes.sort()


def nearest_modes(frequency: float, count: int = 3) -> str:
  nearest = sorted(analytic_modes, key=lambda m: abs(m[0] - frequency))[:count]
  return ", ".join(f"{f:.1f}Hz {ijk}" for f, ijk in nearest)


# ---- optional sweep ----
sweep = None
if arguments.sweep is not None:
  frequencies = np.linspace(arguments.sweep[0], arguments.sweep[1], arguments.points)
  start = time.perf_counter()

  def progress(i: int, frequency: float) -> None:
    print(f"\r[FDFD] sweep {i + 1}/{len(frequencies)} ({frequency:.1f}Hz) "
          f"{time.perf_counter() - start:.0f}s", end="", flush=True)

  response = frequency_domain.frequency_response(
      operator, frequencies, source, [receiver], arguments.hard_source, progress)[:, 0]
  print()
  sweep = (frequencies, response)

# ---- figure ----
state = {
    "frequency": arguments.frequency,
    "slice": min(grid.scale(arguments.slice_height if arguments.slice_height is not None
                            else scene.height / 2), grid.height_parts - 1),
    "field": None,
}

fig = plt.gcf()
fig.set_dpi(150)
fig.set_size_inches(1920 / fig.get_dpi(), 1000 / fig.get_dpi(), forward=True)
rows = 2 if sweep is not None else 1
ax_amp = fig.add_subplot(rows, 3, 1)
ax_shape = fig.add_subplot(rows, 3, 2)
ax_vertical = fig.add_subplot(rows, 3, 3)

cmap_signed = plt.cm.seismic.copy()
cmap_signed.set_bad("black", 1.0)
cmap_amp = plt.cm.magma.copy()
cmap_amp.set_bad("black", 1.0)

amp_image = ax_amp.imshow(np.zeros((grid.width_parts, grid.depth_parts)),
                          cmap=cmap_amp, vmin=-40, vmax=0)
shape_image = ax_shape.imshow(np.zeros((grid.width_parts, grid.depth_parts)),
                              cmap=cmap_signed, vmin=-1, vmax=1)
vertical_image = ax_vertical.imshow(np.zeros((grid.height_parts, grid.width_parts)),
                                    cmap=cmap_signed, vmin=-1, vmax=1, origin="lower")
plt.colorbar(amp_image, ax=ax_amp).set_label("dB re slice max")
plt.colorbar(shape_image, ax=ax_shape).set_label("Normalised")
plt.colorbar(vertical_image, ax=ax_vertical).set_label("Normalised")
for ax in (ax_amp, ax_shape):
  ax.set_xlabel("Depth index")
  ax.set_ylabel("Width index")
  ax.plot(source[2], source[0], "*", color="lime", markersize=10)
  ax.plot(receiver[2], receiver[0], "x", color="cyan")
ax_vertical.set_xlabel("Width index")
ax_vertical.set_ylabel("Height index")
ax_vertical.plot(source[0], source[1], "*", color="lime", markersize=10)
slice_line = ax_vertical.axhline(state["slice"], color="lime", linestyle="--", linewidth=1)

marker = None
if sweep is not None:
  ax_sweep = fig.add_subplot(2, 1, 2)
  frequencies, response = sweep
  ax_sweep.plot(frequencies, 20 * np.log10(np.abs(response)), "-")
  for f, ijk in analytic_modes:
    if frequencies[0] <= f <= frequencies[-1]:
      ax_sweep.axvline(f, color="gray", alpha=0.3, linewidth=1)
  marker = ax_sweep.axvline(state["frequency"], color="red")
  ax_sweep.set_title("Response at receiver (click to select a frequency, gray: box modes)")
  ax_sweep.set_xlabel("Frequency (Hz)")
  ax_sweep.set_ylabel("dB")


def aligned_shape(values: np.ndarray, phase: float) -> np.ndarray:
  """
  Re(values) rotated by phase, normalised to the 99th percentile of the slice
  so the singular source cell does not wash out the rest of the room
  """
  shape = np.real(values * np.exp(-1j * phase))
  scale = np.nanpercentile(np.abs(shape), 99)
  return np.clip(shape / scale, -1, 1) if scale > 0 else shape


def redraw() -> None:
  field = state["field"]
  h = state["slice"]
  horizontal = field[:, h, :]
  vertical = field[:, :, source[2]].T
  amplitude = np.abs(horizontal)
  with np.errstate(divide="ignore"):
    amp_image.set_data(20 * np.log10(amplitude / np.nanmax(amplitude)))
  # the dominant phase of the field, so Re() shows the standing wave
  phase = 0.5 * np.angle(np.nansum(field ** 2))
  shape_image.set_data(aligned_shape(horizontal, phase))
  vertical_image.set_data(aligned_shape(vertical, phase))
  slice_line.set_ydata([h, h])
  f = state["frequency"]
  ax_amp.set_title(f"Amplitude, h={h * parameters.dx:.2f}m")
  ax_shape.set_title("Mode shape (horizontal)")
  ax_vertical.set_title(f"Mode shape (vertical, d={source[2] * parameters.dx:.2f}m)")
  fig.suptitle(f"{arguments.scene} @ {f:.1f}Hz  |  receiver {20 * np.log10(abs(field[receiver])):.1f}dB"
               f"  |  nearest box modes (w,h,d): {nearest_modes(f)}")
  if marker is not None:
    marker.set_xdata([f, f])
  fig.canvas.draw_idle()


def solve_at(frequency: float) -> None:
  state["frequency"] = frequency
  start = time.perf_counter()
  state["field"] = frequency_domain.solve(operator, frequency, source, arguments.hard_source)
  print(f"[FDFD] solved {frequency:.1f}Hz in {time.perf_counter() - start:.2f}s")
  redraw()


def on_click(event) -> None:
  if sweep is not None and event.inaxes is ax_sweep and event.xdata is not None:
    solve_at(float(event.xdata))


def on_key(event) -> None:
  if event.key in ("left", "right"):
    solve_at(max(1.0, state["frequency"] + (arguments.step if event.key == "right" else -arguments.step)))
  elif event.key in ("up", "down"):
    state["slice"] = int(np.clip(state["slice"] + (1 if event.key == "up" else -1), 0, grid.height_parts - 1))
    redraw()


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)
fig.tight_layout(rect=(0, 0, 1, 0.94))
solve_at(state["frequency"])
plt.show()
