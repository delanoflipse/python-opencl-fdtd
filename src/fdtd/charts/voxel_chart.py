from typing import Optional

import numpy as np
from matplotlib.colors import to_rgba_array
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from fdtd.charts.chart import Chart

# corners of a unit square, as offsets along the two axes spanning a face
_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


def draw_voxels(axis: mplot3d.Axes3D, filled: np.ndarray, facecolors=None,
                alpha: Optional[float] = None, shade: bool = True) -> Optional[Poly3DCollection]:
  """Fast replacement for ``axis.voxels(filled, facecolors=...)``.

  Matplotlib's ``voxels`` loops over every grid cell in Python, which takes
  minutes for simulation-sized grids. This builds only the exposed faces
  (between a filled and an empty cell) with numpy and draws them as a single
  collection. ``facecolors`` is one colour or an array of colours per cell.
  """
  filled = np.asarray(filled, dtype=bool)
  if facecolors is None:
    facecolors = "C0"
  colors = np.asarray(facecolors)
  if colors.ndim == filled.ndim + 1:
    colors = to_rgba_array(colors.reshape(-1, colors.shape[-1]))
    colors = colors.reshape(filled.shape + (4,))
  else:
    colors = np.broadcast_to(to_rgba_array(facecolors)[0], filled.shape + (4,))

  padded = np.pad(filled, 1)
  quads, face_colors = [], []
  for axis_index in range(3):
    u, v = [a for a in range(3) if a != axis_index]
    for direction in (-1, 1):
      neighbour = np.roll(padded, -direction, axis=axis_index)[1:-1, 1:-1, 1:-1]
      cells = np.argwhere(filled & ~neighbour)
      if len(cells) == 0:
        continue
      corners = np.repeat(cells[:, None, :], 4, axis=1).astype(float)
      if direction == 1:
        corners[:, :, axis_index] += 1
      corners[:, :, u] += _SQUARE[:, 0]
      corners[:, :, v] += _SQUARE[:, 1]
      quads.append(corners)
      face_colors.append(colors[cells[:, 0], cells[:, 1], cells[:, 2]])

  axis.set_xlim(0, filled.shape[0])
  axis.set_ylim(0, filled.shape[1])
  axis.set_zlim(0, filled.shape[2])
  if not quads:
    return None

  verts = np.concatenate(quads)
  face_colors = np.concatenate(face_colors).copy()
  if alpha is not None:
    face_colors[:, 3] = alpha

  collection = Poly3DCollection(
      verts, facecolors=face_colors, linewidths=0, shade=shade)
  axis.add_collection3d(collection)
  return collection


class VoxelChart(Chart):
  def __init__(self, axis: mplot3d.Axes3D, shape: tuple, title: str) -> None:
    super().__init__(axis, title)
    (w, h, d) = shape
    self.shape = (w, d, h)
    self.values = np.zeros(shape=(w, d, h))
    self.map = np.zeros(shape=(w, d, h, 4))
    self.visible = np.zeros(shape=(w, d, h), dtype="bool")
    self.axis.set_xlabel("Width index")
    self.axis.set_ylabel("Depth index")
    self.axis.set_zlabel("Height index")

  def set_voxel_visible(self, w: int, h: int, d: int, on=True) -> None:
    self.visible[w, d, h] = on

  def set_voxel_color(self, w: int, h: int, d: int, color: list) -> None:
    self.map[w, d, h] = color

  def set_voxels(self, mask: np.ndarray, colors) -> None:
    """Show and colour all cells in ``mask``, both indexed as (w, h, d).

    ``colors`` is one RGBA colour or an array of RGBA colours per cell.
    """
    mask = np.transpose(mask, (0, 2, 1))
    colors = np.asarray(colors)
    if colors.ndim == 4:
      colors = np.transpose(colors, (0, 2, 1, 3))[mask]
    self.visible[mask] = True
    self.map[mask] = colors

  def render(self) -> None:
    draw_voxels(self.axis, self.visible, facecolors=self.map)
