import numpy as np
import matplotlib
import matplotlib.axes as mpl_axes
import matplotlib.lines as mpl_lines
from mpl_toolkits import mplot3d

from lib.charts.chart import Chart


class VoxelChart(Chart):
  def __init__(self, axis: mplot3d.Axes3D, shape: tuple, title: str) -> None:
    super().__init__(axis, title)
    (w, h, d) = shape
    self.shape = (w, d, h)
    self.values = np.zeros(shape=(w, d, h))
    self.map = np.zeros(shape=(w, d, h, 4))
    self.visible = np.zeros(shape=(w, d, h), dtype="bool")
    # self.voxel = self.axis.voxels(self.visible, facecolors=self.map)
    self.axis.set_xlabel("Width index")
    self.axis.set_ylabel("Depth index")
    self.axis.set_zlabel("Height index")

  def set_voxel_visible(self, w: int, h: int, d: int, on=True) -> None:
    self.visible[w, d, h] = on

  def set_voxel_color(self, w: int, h: int, d: int, color: list) -> None:
    self.map[w, d, h] = color

  def set_voxel_value(self, w: int, h: int, d: int, color: list) -> None:
    self.value[w, d, h] = color

  def color_map(self, cm) -> None:
    pass
    # self.map. = cm(self.values)

  def render(self) -> None:
    self.axis.voxels(self.visible, facecolors=self.map)
    self.axis.set_xlim([0, self.shape[0]])
    self.axis.set_ylim([0, self.shape[1]])
    self.axis.set_zlim([0, self.shape[2]])
    # return super().render()
