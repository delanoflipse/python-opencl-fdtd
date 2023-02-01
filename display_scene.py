
from lib.charts.voxel_chart import VoxelChart
from lib.grid import LISTENER_FLAG, SOURCE_REGION_FLAG, WALL_FLAG
from lib.parameters import SimulationParameters
from lib.scene.RealLifeRoomScene import RealLifeRoomScene
import os
import matplotlib
import matplotlib.pyplot as plt

# scene grid
parameters = SimulationParameters()
parameters.set_oversampling(20)
parameters.set_max_frequency(200)

# scene = OfficeScene(parameters)
scene = RealLifeRoomScene(parameters)
# scene = ShoeboxRoomScene(parameters)
# scene = LShapedRoom(parameters)
# scene = CuboidReferenceScene(parameters)
# scene = ConcertHallScene(parameters)

grid = scene.build()
shape = grid.grid_shape

print(f'l: {grid.listener_count}\ts:{grid.source_count}')

# chart
file_dir = os.path.dirname(__file__)
plt.style.use(os.path.join(file_dir, './styles/paper.mplstyle'))
fig = plt.gcf()
fig.set_dpi(300)
fig.set_size_inches(1920/fig.get_dpi(), 1080/fig.get_dpi(), forward=True)

axes_shape = (1, 1)
axis_floor_map = plt.subplot2grid(axes_shape, (0, 0), projection='3d')
# axis_beta = plt.subplot2grid(axes_shape, (0, 1), projection='3d')

chart_walls = VoxelChart(axis_floor_map, shape, "Floor map")
# chart_beta = VoxelChart(axis_beta, shape, "Beta values")
cmap = plt.get_cmap()
for w in range(grid.width_parts):
  for d in range(grid.depth_parts):
    for h in range(grid.height_parts):
      beta = grid.beta[w, h, d]
      is_edge = w == 0 or h == 0 or d == 0
      c = cmap(beta)

      # if beta > 0 and not is_edge:
      #   chart_beta.set_voxel_visible(w, h, d)
      #   chart_beta.set_voxel_color(w, h, d, c)

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
# chart_beta.render()
plt.show()
