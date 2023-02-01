
import sys
import os
from pathlib import Path
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from lib.charts.line_chart import LineChart
from lib.util.parse_rew import parse_rew_file
from lib.util.parse_output import OutputParser

from lib.parameters import SimulationParameters
from lib.scene.RealLifeRoomScene import RealLifeRoomScene

should_export = False
# plt.style.use(os.path.join('./styles/paper.mplstyle'))
plt.style.use(os.path.join('./styles/poster.mplstyle'))
fig = plt.gcf()
fig.set_dpi(300)
fig.set_size_inches(1920/fig.get_dpi(), 1080/fig.get_dpi(), forward=True)

output_type = "measurement"
# output_type = "scheme"
# output_type = "furniture"
# output_type = "oversampling"
# output_type = "time"
# output_type = "bands"
# output_type = "other"
# output_type = "hpc"

title = ""
rew_files: list[tuple[str, str]] = []
csv_files: list[tuple[str, str]] = []

if output_type == "measurement":
  title = "Measured, analytical and simulated response for reference position"

  rew_files = [
      ("S:\\Tu Delft\\CSE3000\\results\\rew\\measurement.txt", "Measured"),
      ("S:\\Tu Delft\\CSE3000\\results\\rew\\simulated.txt", "Analytical solution"),
  ]

  csv_files = [
      # ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 19_51_40 real-reference [300ms-200f-16o-24b-1x-slf] reference-and-start.csv", "Simulated"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 16_39_39 RealLifeRoomScene [5000ms-200f-16o-48b-1x].csv", "Simulated without objects"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-29 18_20_35 real-reference-furniture [5000ms-200f-16o-48b-1x].csv", "Simulated with objects"),
  ]
elif output_type == "scheme":
  title = "Comparison of Compact Schemas"
  csv_files = [
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 19_51_40 real-reference [300ms-200f-16o-24b-1x-slf] reference-and-start.csv", "SLF"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 19_52_50 real-reference [300ms-200f-16o-24b-1x-iwb].csv", "IWB"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 19_55_50 real-reference [300ms-200f-16o-24b-1x-iso].csv", "ISO"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 19_58_05 real-reference [300ms-200f-16o-24b-1x-iso2].csv", "ISO2"),
  ]
elif output_type == "furniture":
  title = "Effect of Furniture in Room"
  csv_files = [
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 11_48_39 RealLifeRoomScene [300ms-200f-16o-96.0b-1x].csv", "Without furniture"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 11_42_06 real-reference-furniture [300ms-200f-16o-96.0b-1x-slf].csv", "With furniture"),
  ]
  # csv_files = [
  #     ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 19_51_40 real-reference [300ms-200f-16o-24b-1x-slf] reference-and-start.csv", "Without furniture"),
  #     ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-27 10_37_50 real-reference-furniture [300ms-200f-16o-24b-1x-slf] with-modes.csv", "With furniture"),
  # ]

elif output_type == "oversampling":
  title = "Comparison of Oversampling Factors"
  csv_files = [
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_02_13 real-reference [300ms-200f-10.0o-24b-1x-slf].csv", "Factor 10"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_02_42 real-reference [300ms-200f-12.0o-24b-1x-slf].csv", "Factor 12"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_03_20 real-reference [300ms-200f-14.0o-24b-1x-slf].csv", "Factor 14"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_04_11 real-reference [300ms-200f-16.0o-24b-1x-slf].csv", "Factor 16"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_05_21 real-reference [300ms-200f-18.0o-24b-1x-slf].csv", "Factor 18"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_06_52 real-reference [300ms-200f-20.0o-24b-1x-slf].csv", "Factor 20"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_08_51 real-reference [300ms-200f-24.0o-24b-1x-slf].csv", "Factor 24"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_12_09 real-reference [300ms-200f-28.0o-24b-1x-slf].csv", "Factor 28"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_17_22 real-reference [300ms-200f-32.0o-24b-1x-slf].csv", "Factor 32"),
  ]
elif output_type == "time":
  title = "Comparison of Time per Frequency"

  # rew_files = [
  #     ("S:\\Tu Delft\\CSE3000\\results\\rew\\simulated.txt", "Analytical solution"),
  # ]

  csv_files = [
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 12_46_51 RealLifeRoomScene [25ms-200f-16o-48b-1x].csv", "25 ms"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 12_48_38 RealLifeRoomScene [50ms-200f-16o-48b-1x].csv", "50 ms"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 12_50_27 RealLifeRoomScene [100ms-200f-16o-48b-1x].csv", "100 ms"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 12_52_18 RealLifeRoomScene [150ms-200f-16o-48b-1x].csv", "150 ms"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 12_54_14 RealLifeRoomScene [200ms-200f-16o-48b-1x].csv", "200 ms"),
      # ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 12_56_13 RealLifeRoomScene [250ms-200f-16o-48b-1x].csv", "250 ms"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 12_58_18 RealLifeRoomScene [300ms-200f-16o-48b-1x].csv", "300 ms"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 13_00_25 RealLifeRoomScene [400ms-200f-16o-48b-1x].csv", "400 ms"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 13_02_37 RealLifeRoomScene [600ms-200f-16o-48b-1x].csv", "600 ms"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 13_05_00 RealLifeRoomScene [800ms-200f-16o-48b-1x].csv", "800 ms"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 13_07_33 RealLifeRoomScene [1000ms-200f-16o-48b-1x].csv", "1000 ms"),
      # ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 13_10_15 RealLifeRoomScene [1500ms-200f-16o-48b-1x].csv", "1500 ms"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 13_13_26 RealLifeRoomScene [2000ms-200f-16o-48b-1x].csv", "2000 ms"),
      # ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 16_22_18 RealLifeRoomScene [2500ms-200f-16o-48b-1x].csv", "2500 ms"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 16_28_54 RealLifeRoomScene [3000ms-200f-16o-48b-1x].csv", "3000 ms"),
      # ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 16_33_54 RealLifeRoomScene [4000ms-200f-16o-48b-1x].csv", "4000 ms"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 16_39_39 RealLifeRoomScene [5000ms-200f-16o-48b-1x].csv", "5000 ms"),
  ]
elif output_type == "bands":
  title = "Comparison of Frequency Bands"
  csv_files = [
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_52_22 real-reference [300ms-200f-16o-3.0b-1x-slf].csv", "1/3 bands"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_52_43 real-reference [300ms-200f-16o-6.0b-1x-slf].csv", "1/6 bands"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_53_10 real-reference [300ms-200f-16o-9.0b-1x-slf].csv", "1/9 bands"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_53_44 real-reference [300ms-200f-16o-12.0b-1x-slf].csv", "1/12 bands"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_54_24 real-reference [300ms-200f-16o-16.0b-1x-slf].csv", "1/16 bands"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_55_14 real-reference [300ms-200f-16o-18.0b-1x-slf].csv", "1/18 bands"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_56_08 real-reference [300ms-200f-16o-20.0b-1x-slf].csv", "1/20 bands"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_57_07 real-reference [300ms-200f-16o-22.0b-1x-slf].csv", "1/22 bands"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 20_58_10 real-reference [300ms-200f-16o-24.0b-1x-slf].csv", "1/24 bands"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 21_43_07 real-reference [300ms-200f-16o-28.0b-1x-slf].csv", "1/28 bands"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 21_44_25 real-reference [300ms-200f-16o-36.0b-1x-slf].csv", "1/36 bands"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 21_46_01 real-reference [300ms-200f-16o-48.0b-1x-slf].csv", "1/48 bands"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 21_48_04 real-reference [300ms-200f-16o-60.0b-1x-slf].csv", "1/60 bands"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 21_50_35 real-reference [300ms-200f-16o-72.0b-1x-slf].csv", "1/72 bands"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 21_53_33 real-reference [300ms-200f-16o-84.0b-1x-slf].csv", "1/84 bands"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-26 21_57_01 real-reference [300ms-200f-16o-96.0b-1x-slf].csv", "1/96 bands"),
  ]
elif output_type == "other":
  title = "-"
  rew_files = [
      ("S:\\Tu Delft\\CSE3000\\results\\rew\\simulated.txt", "Analytical solution"),
      ("S:\\Tu Delft\\CSE3000\\results\\rew\\measurement.txt", "Measured"),
  ]
  csv_files = [
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 17_33_35 RealLifeRoomScene [5000ms-200f-16o-48b-1x-iwb].csv", "IWB"),
      ("S:\\Tu Delft\\CSE3000\\results\\references\\2023-01-28 16_39_39 RealLifeRoomScene [5000ms-200f-16o-48b-1x].csv", "SLF"),
  ]
elif output_type == "hpc":
  title = "-"
  csv_files = [
      ("S:\\Tu Delft\\CSE3000\\results\\dhpc\\2023-01-25 19_45_50 RealLifeRoomScene [300ms-200f-20.0o-24.0b-1x].csv", "300ms 24b 20o"),
      ("S:\\Tu Delft\\CSE3000\\results\\dhpc\\2023-01-28 16_01_52 RealLifeRoomScene [3000ms-200f-16.0o-36.0b-1x].csv", "3s 36b 16o"),
  ]


axes_shape = (1, 1)

axis = plt.subplot2grid(axes_shape, (0, 0))
spl_chart = LineChart(axis, "linhz_spl", title)

parameters = SimulationParameters()
parameters.set_oversampling(20)
parameters.set_max_frequency(200)

# scene = OfficeScene(parameters)
scene = RealLifeRoomScene(parameters)

room_modes = scene.get_room_modes()
for (modal_frequency, axis_type) in room_modes:
  if modal_frequency > 250:
    continue
  axis.axvline(modal_frequency, lw=0.5, ls='--', color='k', alpha=0.4)


for (rew_file, label) in rew_files:
  (f, v) = parse_rew_file(rew_file)
  spl_chart.add_dataset(f, v, label=label)

max_spl = None
min_spl = None

for (csv_file, label) in csv_files:
  output = OutputParser(csv_file)
  output.parse_values()
  arr = np.array(output.best_set)
  spl_chart.add_dataset(output.frequencies, arr, "-",
                        label=label)
  if max_spl is None:
    max_spl = arr
    min_spl = arr
  else:
    max_spl = np.maximum(arr, max_spl)
    min_spl = np.minimum(arr, min_spl)

delta = max_spl - min_spl
print(np.max(delta), np.average(delta))

axis.legend()
axis.set_xlim(20, 200)

spl_chart.render()

plt.tight_layout(pad=2)

plt.savefig(f"./output/ref/effect-{output_type}.png", dpi=300)

plt.show()
