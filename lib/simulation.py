import math
import numpy as np
from numba import njit, prange

from lib.grid import create_grid
from lib.geometry import populate_neighbours
from lib.constants import KAPPA, RHO, RHO_INVERSE
from lib.parameters import AIR_DAMPENING, DEPTH_PARTS, DT, DT_OVER_DX, HEIGHT_PARTS, WIDTH_PARTS

class GridPosition:
  w = 0
  h = 0
  d = 0
  def __init__(self, w, h, d):
    self.w = w
    self.h = h
    self.d = d

class Source:
  position = GridPosition(0, 0, 0)
  frequency = 0
  pulses = 1
  invert_phase = False
  start_at = 0

class SimulationState:
  geometry = create_grid("int8")
  neighbours = create_grid("int8")
  pressure = create_grid("float64")
  velocity_x = create_grid("float64")
  velocity_y = create_grid("float64")
  velocity_z = create_grid("float64")
  analysis = create_grid("float64")
  time = 0
  iteration = 0
  sources = []
  
sim = SimulationState()
s1 = Source()
s1.position = GridPosition(WIDTH_PARTS // 2, HEIGHT_PARTS // 2, DEPTH_PARTS // 2)
s1.frequency = 1000
sim.sources.append(s1)

RHO_DT_DX = -1 * RHO_INVERSE * DT_OVER_DX
@njit(parallel=True)
def velocity_step(pressure, velocity_x, velocity_y, velocity_z, geometry, neighbours):
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
def pressure_step(pressure, velocity_x, velocity_y, velocity_z, geometry, neighbours):
  for w in prange(pressure.shape[0]):
    for h in prange(pressure.shape[1]):
      for d in prange(pressure.shape[2]):
        if geometry[w,h,d] > 1:
          continue
        n = neighbours[w,h,d]
        current_pressure = pressure[w,h,d]
        dvx = velocity_x[w + 1, h, d] - velocity_x[w , h, d] if w < WIDTH_PARTS - 1 else 0
        dvy = velocity_y[w, h + 1, d] - velocity_y[w , h, d] if h < HEIGHT_PARTS - 1 else 0
        dvz = velocity_z[w, h, d + 1] - velocity_z[w , h, d] if d < DEPTH_PARTS - 1 else 0
        dv = dvx + dvy + dvz
        dp = KAPPA_DT_DX * dv
        # TODO: invert phase on boundary (& etc)
        # if n < 6:
        #   cf = 0.5*(6 - n)
        #   pressure[w,h,d] = (current_pressure + cf * dp) / (1.0 + cf)
        # else:
        #   pressure[w,h,d] = (current_pressure + dp) * AIR_DAMPENING
        pressure[w,h,d] = (current_pressure + dp) * AIR_DAMPENING
        
        
@njit(parallel=True)
def analysis_step(pressure, velocity_x, velocity_y, velocity_z, geometry, analysis):
  for w in prange(pressure.shape[0]):
    for h in prange(pressure.shape[1]):
      for d in prange(pressure.shape[2]):
        if geometry[w,h,d] > 1:
          continue
        cur = analysis[w,h,d]
        pres = pressure[w,h,d]
        analysis[w,h,d] = max(abs(pres), cur)

def simulation_setup():
  populate_neighbours(sim.geometry, sim.neighbours)
  for source in sim.sources:
    sim.geometry[source.position.w, source.position.h, source.position.d] == 2

def simulation_step():
  velocity_step(sim.pressure, sim.velocity_x, sim.velocity_y, sim.velocity_z, sim.geometry, sim.neighbours)
  pressure_step(sim.pressure, sim.velocity_x, sim.velocity_y, sim.velocity_z, sim.geometry, sim.neighbours)
  analysis_step(sim.pressure, sim.velocity_x, sim.velocity_y, sim.velocity_z, sim.geometry, sim.analysis)
  
  for source in sim.sources:
    radial_f = 2.0 * math.pi * source.frequency
    time_active = source.pulses / source.frequency
    break_off = source.start_at + time_active
    active = sim.time >= source.start_at and (source.pulses == 0 or sim.time <= break_off)
    rel_time = sim.time - source.start_at
    factor = -1 if source.invert_phase else 1 if active else 0
    radial_t = radial_f * rel_time
    signal = math.sin(radial_t) * factor
    sim.pressure[source.position.w, source.position.h, source.position.d] = signal
  
  sim.time += DT
  sim.iteration += 1