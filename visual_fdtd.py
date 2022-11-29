# from mpl_toolkits import mplot3d
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lib.simulation import sim, simulation_step
from lib.parameters import HEIGHT_PARTS

fig = plt.figure()

slice_d = HEIGHT_PARTS // 2

slice = sim.pressure[:,:,slice_d]
y = np.arange(len(slice))
x = np.arange(len(slice[0]))
(x, y) = np.meshgrid(x,y)
cs = plt.imshow(slice, cmap='plasma')
cbar = plt.colorbar()

maximum = 1e-6
last_maximum = 1e-6

def animate(i):
  global maximum, last_maximum
  for _ in range(5):
      simulation_step()
  print(i, sim.time)
  slice = sim.pressure[:,:,slice_d]
  maximum = max(maximum, abs(slice.min()), abs(slice.max()))
  cs.set_data(slice)
  if last_maximum != maximum:
    cs.set_clim(-maximum, maximum)
  last_maximum = maximum
  fig.canvas.flush_events()

ani = FuncAnimation(plt.gcf(), animate, interval=0)
plt.show()