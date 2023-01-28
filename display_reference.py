
import sys
import os
from pathlib import Path
import argparse
import matplotlib
import matplotlib.pyplot as plt

from lib.charts.line_chart import LineChart
from lib.util.parse_rew import parse_rew_file
from lib.util.parse_output import OutputParser

file_dir = os.path.dirname(__file__)
parser = argparse.ArgumentParser()
parser.add_argument("rew_path", type=Path)
parser.add_argument("result_path", type=Path)
parser.add_argument("result_with_path", type=Path)
parser.add_argument("--export", action=argparse.BooleanOptionalAction)

parsed = parser.parse_args()
should_export = parsed.export

measured_csv = os.path.join(parsed.rew_path, "measurement.txt")
trivial_csv = os.path.join(parsed.rew_path, "simulated.txt")
(f1, v1) = parse_rew_file(measured_csv)
(f2, v2) = parse_rew_file(trivial_csv)

simulated_csv = parsed.result_path
simulated_csv2 = parsed.result_with_path
output = OutputParser(simulated_csv)
output.parse_values()
output2 = OutputParser(simulated_csv2)
output2.parse_values()

plt.style.use(os.path.join(file_dir, './styles/paper.mplstyle'))
fig = plt.gcf()
fig.set_dpi(300)
fig.set_size_inches(1920/fig.get_dpi(), 1080/fig.get_dpi(), forward=True)
axes_shape = (1, 1)

axis = plt.subplot2grid(axes_shape, (0, 0))
spl_chart = LineChart(axis, "hz_spl", "SPL values")
spl_chart.add_dataset(f1, v1, label="Measured")
spl_chart.add_dataset(f2, v2, label="Analytic solution")

spl_chart.add_dataset(output.frequencies, output.best_set, "-",
                      label="Simulated")

spl_chart.add_dataset(output2.frequencies, output2.best_set, "-",
                      label="Simulated with furniture")
axis.legend()

spl_chart.render()

plt.tight_layout(pad=2)
if should_export:
  pass
else:
  plt.show()
