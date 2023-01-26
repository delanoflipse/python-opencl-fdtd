import matplotlib
import matplotlib.axes as mpl_axes
import matplotlib.lines as mpl_lines

from lib.charts.chart import Chart
from typing import Tuple, List

class LineChart(Chart):
  def __init__(self, axis: mpl_axes.Axes, axis_type: str, title: str) -> None:
    super().__init__(axis, title)
    self.axis_type = axis_type
    self.plots: List[Tuple[mpl_lines.Line2D, list, list]] = []
    self.set_axis()

  def set_axis(self) -> None:
    x_type, y_type = self.axis_type.split("_")
    if x_type == "hz":
      self.axis.set_xlabel("Frequency (hz)")
      self.axis.set_xscale('log')
      self.axis.set_xticks([20, 25, 30, 40, 50, 60, 80, 100, 120, 160, 200])
      self.axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if y_type == "spl":
      self.axis.set_ylabel("Sound Pressure Level (dB)")

    if y_type == "pa":
      self.axis.set_ylabel("Sound Pressure (Pa)")

  def add_dataset(self, x_list: list, y_list: list, *args, **kwargs) -> None:
    plot, = self.axis.plot(x_list, y_list, *args, **kwargs)
    self.plots.append((plot, x_list, y_list))
    self.set_axis()

  def render(self) -> None:
    for (plot, x_list, y_list) in self.plots:
      plot.set_data(x_list, y_list)
    return super().render()
