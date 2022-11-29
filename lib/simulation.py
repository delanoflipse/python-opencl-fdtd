import math
import numpy as np
from numba import jit, cuda, njit, prange

from lib.grid import create_grid
from lib.geometry import geometry
from lib.constants import KAPPA, RHO, RHO_INVERSE
from lib.parameters import DEPTH_PARTS, DT, DT_OVER_DX, HEIGHT_PARTS, WIDTH_PARTS

class SimulationState:
  pressure = create_grid("float64")
  velocity_x = create_grid("float64")
  velocity_y = create_grid("float64")
  velocity_z = create_grid("float64")
  time = 0
  iteration = 0
  
sim = SimulationState()

RHO_DT_DX = -1 * RHO_INVERSE * DT_OVER_DX
@njit(parallel=True)
def velocity_step(pressure, velocity_x, velocity_y, velocity_z):
  for w in prange(pressure.shape[0]):
    for h in prange(pressure.shape[1]):
      for d in prange(pressure.shape[2]):
        current_pressure = pressure[w,h,d]
        if (w > 0):
          dpw = current_pressure - pressure[w - 1, h, d]
          dvz = RHO_DT_DX * dpw
          current_vx = velocity_x[w, h, d]
          velocity_x[w, h, d] = current_vx + dvz
        if (h > 0):
          dph = current_pressure - pressure[w, h - 1, d]
          dvz = RHO_DT_DX * dph
          current_vy = velocity_y[w, h, d]
          velocity_y[w, h, d] = current_vy + dvz
        if (d > 0):
          dpz = current_pressure - pressure[w, h, d - 1]
          dvz = RHO_DT_DX * dpz
          current_vz = velocity_z[w, h, d]
          velocity_z[w, h, d] = current_vz + dvz

KAPPA_DT_DX = -1 * KAPPA * DT_OVER_DX    
@njit(parallel=True)
def pressure_step(pressure, velocity_x, velocity_y, velocity_z):
  for w in prange(pressure.shape[0]):
    for h in prange(pressure.shape[1]):
      for d in prange(pressure.shape[2]):
        current_pressure = pressure[w,h,d]
        dvx = velocity_x[w + 1, h, d] - velocity_x[w , h, d] if w < WIDTH_PARTS - 1 else 0
        dvy = velocity_y[w, h + 1, d] - velocity_y[w , h, d] if h < HEIGHT_PARTS - 1 else 0
        dvz = velocity_z[w, h, d + 1] - velocity_z[w , h, d] if d < DEPTH_PARTS - 1 else 0
        dv = dvx + dvy + dvz
        dp = KAPPA_DT_DX * dv
        pressure[w,h,d] = current_pressure + dp

def simulation_step():
  velocity_step(sim.pressure, sim.velocity_x, sim.velocity_y, sim.velocity_z)
  pressure_step(sim.pressure, sim.velocity_x, sim.velocity_y, sim.velocity_z)
  
  if sim.time < 1 / 200:
    sim.pressure[WIDTH_PARTS // 2, HEIGHT_PARTS // 2, DEPTH_PARTS // 2] = math.sin(sim.time)
  
  sim.time += DT
  sim.iteration += 1