"""Show the voxelised geometry and beta values of a scene."""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from fdtd.charts.voxel_chart import VoxelChart
from fdtd.grid import LISTENER_FLAG, SOURCE_REGION_FLAG, WALL_FLAG
from fdtd.parameters import SimulationParameters
from fdtd.scenes import SCENES, create_scene
from fdtd.styles import use_style


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("-s", "--scene", default="real-reference", choices=SCENES)
  parser.add_argument("-o", "--oversampling", default=24, type=float)
  parser.add_argument("--style", default=None, choices=["paper", "poster"])
  arguments = parser.parse_args()

  # scene grid
  parameters = SimulationParameters()
  parameters.set_oversampling(arguments.oversampling)
  parameters.set_max_frequency(200)

  scene = create_scene(arguments.scene, parameters)
  grid = scene.build()
  shape = grid.grid_shape

  # chart
  if arguments.style is not None:
    use_style(arguments.style)
  fig = plt.gcf()
  fig.set_dpi(150)
  fig.set_size_inches(1920/fig.get_dpi(), 1080/fig.get_dpi(), forward=True)

  axes_shape = (1, 2)
  axis_floor_map = plt.subplot2grid(axes_shape, (0, 0), projection='3d')
  axis_beta = plt.subplot2grid(axes_shape, (0, 1), projection='3d')

  chart_walls = VoxelChart(axis_floor_map, shape, "Floor map")
  chart_beta = VoxelChart(axis_beta, shape, "Beta values")
  beta_colors = plt.get_cmap()(grid.beta)
  is_edge = np.zeros(grid.beta.shape, dtype=bool)
  is_edge[0, :, :] = is_edge[:, 0, :] = is_edge[:, :, 0] = True
  chart_beta.set_voxels((grid.beta > 0) & ~is_edge, beta_colors)

  walls = (grid.geometry & WALL_FLAG) > 0
  listeners = ((grid.geometry & LISTENER_FLAG) > 0) & ~walls
  sources = ((grid.geometry & SOURCE_REGION_FLAG) > 0) & ~walls
  chart_walls.set_voxels(walls, beta_colors)
  chart_walls.set_voxels(listeners, [1, 0, 0, 0.4])
  chart_walls.set_voxels(sources, [0, 0, 1, 0.4])

  chart_walls.render()
  chart_beta.render()
  plt.show()


if __name__ == "__main__":
  main()
