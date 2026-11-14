"""Show the voxelised geometry and beta values of a scene."""
import argparse

import matplotlib.pyplot as plt

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
  cmap = plt.get_cmap()
  for w in range(grid.width_parts):
    for d in range(grid.depth_parts):
      for h in range(grid.height_parts):
        beta = grid.beta[w, h, d]
        is_edge = w == 0 or h == 0 or d == 0
        c = cmap(beta)

        if beta > 0 and not is_edge:
          chart_beta.set_voxel_visible(w, h, d)
          chart_beta.set_voxel_color(w, h, d, c)

        if grid.geometry[w, h, d] & WALL_FLAG > 0:
          chart_walls.set_voxel_visible(w, h, d)
          chart_walls.set_voxel_color(w, h, d, c)
          continue

        if grid.geometry[w, h, d] & LISTENER_FLAG > 0:
          chart_walls.set_voxel_visible(w, h, d)
          chart_walls.set_voxel_color(w, h, d, [1, 0, 0, 0.4])

        if grid.geometry[w, h, d] & SOURCE_REGION_FLAG > 0:
          chart_walls.set_voxel_visible(w, h, d)
          chart_walls.set_voxel_color(w, h, d, [0, 0, 1, 0.4])

  chart_walls.render()
  chart_beta.render()
  plt.show()


if __name__ == "__main__":
  main()
