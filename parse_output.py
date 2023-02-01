import csv
import math
import os
import sys
import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

from typing import List

import numpy as np
from lib.grid import LISTENER_FLAG, SOURCE_REGION_FLAG, WALL_FLAG
from lib.parameters import SimulationParameters
from lib.scene.RealLifeRoomScene import RealLifeRoomScene

if len(sys.argv) == 1:
  print('No file given. Usage: python parse_output.py <path/to/output.csv>')
  sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument("file_path", type=Path)
parser.add_argument("--export", action=argparse.BooleanOptionalAction)

parsed = parser.parse_args()
should_export = parsed.export

file_dir = os.path.dirname(__file__)
csv_path = str(parsed.file_path)
export_path = csv_path + ".png"
print(csv_path)

csv_file = open(csv_path, 'r', encoding="utf-8", newline='')
reader = csv.reader(csv_file, delimiter=',', quotechar='|')

# scene grid
parameters = SimulationParameters()
parameters.set_oversampling(16)
parameters.set_max_frequency(200)

# scene = OfficeScene(parameters)
scene = RealLifeRoomScene(parameters)

grid = scene.build()

base_map = np.zeros(
    shape=(grid.width_parts, grid.depth_parts, grid.height_parts))

base_map_visible = np.zeros(
    shape=(grid.width_parts, grid.depth_parts, grid.height_parts), dtype="bool")

site_map = np.zeros(
    shape=(grid.width_parts, grid.depth_parts, grid.height_parts, 3))
site_map_visible = np.zeros(
    shape=(grid.width_parts, grid.depth_parts, grid.height_parts), dtype="bool")

value_map = np.zeros(
    shape=(grid.width_parts, grid.depth_parts, grid.height_parts))
value_map_visible = np.zeros(
    shape=(grid.width_parts, grid.depth_parts, grid.height_parts), dtype="bool")

for w in range(grid.width_parts):
  for d in range(grid.depth_parts):
    for h in range(grid.height_parts):
      # beta = grid.beta[w, h, d]
      if grid.geometry[w, h, d] & WALL_FLAG > 0:
        base_map[w, d, h] = grid.beta[w, h, d]
        base_map_visible[w, d, h] = True
        continue
      is_source = False
      is_listener = False
      if grid.geometry[w, h, d] & LISTENER_FLAG > 0:
        is_listener = True
      if grid.geometry[w, h, d] & SOURCE_REGION_FLAG > 0:
        is_source = True
        has_source = True

      if is_source and is_listener:
        site_map[w, d, h, 1] = 1.0
      if is_source:
        site_map[w, d, h, 2] = 1.0
      elif is_listener:
        site_map[w, d, h, 0] = 1.0
      site_map_visible[w, d, h] = is_source or is_listener
      value_map[w, d, h] = math.nan
      value_map_visible[w, d, h] = is_source

# chart
plt.style.use(os.path.join(file_dir, './styles/poster.mplstyle'))
fig = plt.gcf()
fig.set_dpi(300)
fig.set_size_inches(1920/fig.get_dpi(), 860/fig.get_dpi(), forward=True)
# axes_shape = (2, 3)
# axes_shape = (4, 3)

# axis_spl_all = plt.subplot2grid(axes_shape, (0, 0))
# axis_scores = plt.subplot2grid(axes_shape, (0, 0))
axis_best_worst = plt.subplot2grid((1, 1), (0, 0), colspan=4, rowspan=3)
# axis_boxplot_spl = plt.subplot2grid(axes_shape, (0, 2))
# axis_floor_map = plt.subplot2grid(axes_shape, (3, 0), projection='3d')
# axis_site_map = plt.subplot2grid(axes_shape, (3, 1), projection='3d')
# axis_value_map = plt.subplot2grid(
#     axes_shape, (1, 2), projection='3d', rowspan=3, colspan=3)
# axis_value_map = plt.subplot2grid(axes_shape, (3, 2), projection='3d')

# axis_spl_all.set_title("SPL (dB) values for all positions")
# axis_scores.set_title("Scores per index")
axis_best_worst.set_title(
    "Comparison of optimal and least optimal frequency response")
# axis_boxplot_spl.set_title("Range of SPL values")
# axis_floor_map.set_title("Solid objects")
# axis_site_map.set_title("Source and listener locations")
# axis_value_map.set_title("Flatness of frequency response")

# for ax in [axis_floor_map, axis_site_map, axis_value_map]:
#   ax.set_ylabel("Depth index")
#   ax.set_xlabel("Width index")
# ax.set_zlabel("Height index")

frequencies: List[float] = []
frequencies_div = None
max_per_frequency = []
sum_per_frequency = []
min_per_frequency = []
values_per_frequency = []


def as_float(value: str) -> float:
  try:
    return float(value)
  except:
    return math.nan


count = 0
max_value = -float('inf')
max_spl_overall = -float('inf')
min_spl_overall = float('inf')
min_value = float('inf')
best_set = []
best_set_index = -1
worst_set = []
location_map = {}
band_map = {}
indexes = []
value_index = []

for i, row in enumerate(reader):
  float_row = list(map(as_float, row))
  _, index, w_i, w_m, h_i, h_m, d_i, d_m, dev, spl, _, *bands = float_row
  if i == 0:
    frequencies = bands
    frequencies_div = np.diff(bands, n=1)
    for i, _ in enumerate(bands):
      # min, max, sum
      max_per_frequency.append(-float('inf'))
      min_per_frequency.append(float('inf'))
      sum_per_frequency.append(0)
      values_per_frequency.append([])
    continue
  derrivative = np.diff(bands, n=1)
  power = np.power(derrivative, 2)
  power_x_df = np.divide(power, frequencies_div)
  # derrivative = np.gradient(bands, edge_order=2)
  value = np.sum(power)
  # value = np.sum(np.abs(derrivative))
  # bands = [0] + list(np.power(derrivative, 2))
  if value == 0.0:
    continue
  count += 1
  w = int(w_i)
  d = int(d_i)
  h = int(h_i)
  location_map[i] = (w, h, d)
  value_map[w, d, h] = value
  max_value = max(value, max_value)
  min_value = min(value, min_value)
  indexes.append(i)
  value_index.append(value)
  band_map[i] = bands

  if value == max_value:
    worst_set = bands

  if value == min_value:
    best_set = bands
    best_set_index = i

  # print(bands)
  # axis_spl_all.plot(frequencies, bands)
  for freq_index, frequency in enumerate(frequencies):
    value = bands[freq_index]
    max_per_frequency[freq_index] = max(max_per_frequency[freq_index], value)
    min_per_frequency[freq_index] = min(min_per_frequency[freq_index], value)
    sum_per_frequency[freq_index] += value
    values_per_frequency[freq_index].append(value)
    max_spl_overall = max(max_spl_overall, value)
    min_spl_overall = min(min_spl_overall, value)

optimal = sorted(zip(value_index, indexes))
print("BEST:")
for i in range(min(5, len(optimal))):
  value, index = optimal[i]
  pos = location_map[index]
  w, h, d = pos
  min_band = np.min(band_map[index])
  max_band = np.max(band_map[index])
  diff = max_band - min_band
  print(
      f'{i}: {index} with {value} ({max_band}-{min_band}, d:{diff}) at [{w}, {h}, {d}] = ( {(w + 0.5) * parameters.dx}, {(h + 0.5) * parameters.dx}, {(d + 0.5) * parameters.dx} )')

print("WORST:")
for i in range(min(5, len(optimal))):
  value, index = optimal[-i-1]
  pos = location_map[index]
  w, h, d = pos
  min_band = np.min(band_map[index])
  max_band = np.max(band_map[index])
  diff = max_band - min_band
  print(
      f'{i}: {index} with {value} ({max_band}-{min_band}, d:{diff}) at [{w}, {h}, {d}] = ( {(w + 0.5) * parameters.dx}, {(h + 0.5) * parameters.dx}, {(d + 0.5) * parameters.dx} )')

csv_file.close()

# axis_scores.plot(indexes, value_index, "-")

# boxplot = axis_boxplot_spl.boxplot(
#     values_per_frequency, positions=frequencies, sym="", meanline=True)

# boxmedians = []
# for medline in boxplot['medians']:
#   linedata = medline.get_ydata()
#   median = linedata[0]
#   boxmedians.append(median)
# # avg_per_frequency = list(map(lambda x: x / count, sum_per_frequency))
# # medians = list(map(lambda x: x / count, sum_per_frequency))
# axis_boxplot_spl.plot(frequencies, boxmedians, "-", label="Median SPL")
# axis_boxplot_spl.legend()

# axis_best_worst.plot(frequencies, max_per_frequency, "--",
#                      label="Maximal SPL per frequency", lw=0.5)
# axis_best_worst.plot(frequencies, min_per_frequency, "--", lw=0.5,
#                      label="Minimal SPL per frequency")
axis_best_worst.fill_between(
    frequencies, max_per_frequency, min_per_frequency, alpha=0.15, color="k")
mean_value = np.mean(best_set)

# print("best pos", location_map[best_set_index])
# avg_spl = grid.get_average_spl_for_source(location_map[best_set_index])
# axis_best_worst.axhline(avg_spl,  lw=0.5,
#                         ls='--', color='k', alpha=0.4)

axis_best_worst.axhline(mean_value,  lw=0.5,
                        ls='--', color='k', alpha=0.4)

axis_best_worst.plot(frequencies, worst_set, "-",
                     label="Least optimal frequency response", lw=0.9)
axis_best_worst.plot(frequencies, best_set, "-",
                     label="Optimal frequency response")
axis_best_worst.legend(ncol=2, fontsize="small", loc="lower left")

room_modes = scene.get_room_modes()
for (modal_frequency, axis_type) in room_modes:
  if modal_frequency > frequencies[-1]:
    continue

  axis_best_worst.axvline(modal_frequency, lw=0.5,
                          ls='--', color='k', alpha=0.4)

axis_best_worst.set_xlim(20, 200)
axis_best_worst.set_ylim(np.min(min_per_frequency) *
                         0.95, np.max(max_per_frequency) * 1.02)

# base_map_colors = plt.cm.viridis(base_map)
# axis_floor_map.voxels(base_map_visible, alpha=0.8, facecolors=base_map_colors)
# axis_site_map.voxels(site_map_visible, alpha=0.8, facecolors=site_map)

# min_max_range = max_value - min_value
# normalized_value_map = 1 - ((value_map - min_value) / min_max_range)
# value_map_colors = plt.cm.RdYlGn(normalized_value_map)

# axis_value_map.voxels(
#     value_map_visible, facecolors=value_map_colors, alpha=0.5)


# plt.colorbar(axis_floor_map.imshow(base_map, cmap="binary"), ax=axis_floor_map)
# plt.colorbar(axis_site_map.imshow(site_map), ax=axis_site_map)

# for ax in [axis_best_worst, axis_boxplot_spl]:
for ax in [axis_best_worst]:
  ax.set_xlabel("Frequency (hz)")
  ax.set_ylabel("Average Equivalent SPL (dB)")
  # ax.set_xscale('log')
  # ax.set_xticks([20, 25, 30, 40, 50, 60, 80, 100, 120, 160, 200])
  # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
  # ax.set_xlim(20, 200)
  # ax.set_ylim(min_spl_overall * 0.5, 91)
  ax.autoscale_view()

# axis_scores.autoscale_view()


plt.tight_layout(pad=2)
if should_export:
  plt.savefig(export_path, dpi=600)
# else:
plt.show()
