import os
import math
import numpy as np
from numba import njit, prange

import pyopencl as cl
from lib.grid import create_grid
from lib.geometry import populate_neighbours
from lib.constants import KAPPA, RHO, RHO_INVERSE
from lib.parameters import AIR_DAMPENING, ARG_D1, ARG_D2, ARG_D3, ARG_D4, DEPTH_PARTS, DT, DT_OVER_DX, HEIGHT_PARTS, WIDTH_PARTS


class GridPosition:
  w = 0
  h = 0
  d = 0

  def __init__(self, w, h, d):
    self.w = w
    self.h = h
    self.d = d


class KernelProgram(object):
  pass


class Source:
  position = GridPosition(0, 0, 0)
  frequency = 0
  pulses = 1
  invert_phase = False
  start_at = 0


class SimulationState:
  geometry: np.ndarray = create_grid("int8")
  neighbours = create_grid("int8")
  pressure = create_grid("float64")
  pressure_previous = create_grid("float64")
  pressure_next = create_grid("float64")
  analysis = create_grid("float64")
  time = 0
  iteration = 0
  sources = []

# RHO_DT_DX = -1 * RHO_INVERSE * DT_OVER_DX


# @njit(parallel=True)
# def velocity_step(pressure, velocity_x, velocity_y, velocity_z, geometry, neighbours) -> None:
#   for w in prange(pressure.shape[0]):
#     for h in prange(pressure.shape[1]):
#       for d in prange(pressure.shape[2]):
#         current_pressure = pressure[w, h, d]
#         if (w > 0):
#           dpw = current_pressure - pressure[w - 1, h, d]
#           dvz = RHO_DT_DX * dpw
#           current_vx = velocity_x[w, h, d]
#           velocity_x[w, h, d] = current_vx + dvz
#         if (h > 0):
#           dph = current_pressure - pressure[w, h - 1, d]
#           dvz = RHO_DT_DX * dph
#           current_vy = velocity_y[w, h, d]
#           velocity_y[w, h, d] = current_vy + dvz
#         if (d > 0):
#           dpz = current_pressure - pressure[w, h, d - 1]
#           dvz = RHO_DT_DX * dpz
#           current_vz = velocity_z[w, h, d]
#           velocity_z[w, h, d] = current_vz + dvz


# KAPPA_DT_DX = -1 * KAPPA * DT_OVER_DX


# @njit(parallel=True)
# def pressure_step(pressure, velocity_x, velocity_y, velocity_z, geometry, neighbours) -> None:
#   for w in prange(pressure.shape[0]):
#     for h in prange(pressure.shape[1]):
#       for d in prange(pressure.shape[2]):
#         if geometry[w, h, d] > 1:
#           continue
#         n = neighbours[w, h, d]
#         current_pressure = pressure[w, h, d]
#         dvx = velocity_x[w + 1, h, d] - \
#             velocity_x[w, h, d] if w < WIDTH_PARTS - 1 else 0
#         dvy = velocity_y[w, h + 1, d] - \
#             velocity_y[w, h, d] if h < HEIGHT_PARTS - 1 else 0
#         dvz = velocity_z[w, h, d + 1] - \
#             velocity_z[w, h, d] if d < DEPTH_PARTS - 1 else 0
#         dv = dvx + dvy + dvz
#         dp = KAPPA_DT_DX * dv
#         # TODO: invert phase on boundary (& etc)
#         # if n < 6:
#         #   cf = 0.5*(6 - n)
#         #   pressure[w,h,d] = (current_pressure + cf * dp) / (1.0 + cf)
#         # else:
#         #   pressure[w,h,d] = (current_pressure + dp) * AIR_DAMPENING
#         pressure[w, h, d] = (current_pressure + dp) * AIR_DAMPENING


# @njit(parallel=True)
# def analysis_step(pressure, velocity_x, velocity_y, velocity_z, geometry, analysis) -> None:
#   for w in prange(pressure.shape[0]):
#     for h in prange(pressure.shape[1]):
#       for d in prange(pressure.shape[2]):
#         if geometry[w, h, d] > 1:
#           continue
#         cur = analysis[w, h, d]
#         pres = pressure[w, h, d]
#         analysis[w, h, d] = max(abs(pres), cur)


def simulation_setup(sim: SimulationState) -> None:
  populate_neighbours(sim.geometry, sim.neighbours)

  for source in sim.sources:
    sim.geometry[source.position.w, source.position.h, source.position.d] = 2

  os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
  platforms = cl.get_platforms()
  ctx = cl.create_some_context(interactive=False)
  print(platforms[0].name)
  queue = cl.CommandQueue(ctx)
  mf = cl.mem_flags
  rw_flag = mf.READ_ONLY | mf.COPY_HOST_PTR
  a_buf = cl.Buffer(ctx, rw_flag, hostbuf=sim.analysis)
  p_buf = cl.Buffer(ctx, rw_flag, hostbuf=sim.pressure)
  pn_buf = cl.Buffer(ctx, rw_flag, hostbuf=sim.pressure_next)
  pv_buf = cl.Buffer(ctx, rw_flag, hostbuf=sim.pressure_previous)
  g_buf = cl.Buffer(ctx, rw_flag, hostbuf=sim.geometry)
  n_buf = cl.Buffer(ctx, rw_flag, hostbuf=sim.neighbours)

  dir = os.path.dirname(__file__)
  loc = os.path.join(dir, './kernels/fdtd.cl')
  source = open(loc, 'r').read()
  prg = cl.Program(ctx, source).build()

  compact_step_kernel = prg.compact_step
  compact_step_kernel.set_scalar_arg_dtypes(
      [None, None, None, None, None, np.uint32, np.uint32, np.uint32, np.float64, np.float64, np.float64, np.float64])
  compact_step_kernel.set_arg(0, pv_buf)
  compact_step_kernel.set_arg(1, p_buf)
  compact_step_kernel.set_arg(2, pn_buf)
  compact_step_kernel.set_arg(3, g_buf)
  compact_step_kernel.set_arg(4, n_buf)

  compact_step_kernel.set_arg(5, WIDTH_PARTS)
  compact_step_kernel.set_arg(6, HEIGHT_PARTS)
  compact_step_kernel.set_arg(7, DEPTH_PARTS)

  compact_step_kernel.set_arg(8, ARG_D1)
  compact_step_kernel.set_arg(9, ARG_D2)
  compact_step_kernel.set_arg(10, ARG_D3)
  compact_step_kernel.set_arg(11, ARG_D4)

  analysis_step_kernel = prg.analysis_step

  sim.kernel = KernelProgram()
  sim.kernel.ctx = ctx
  sim.kernel.queue = queue
  sim.kernel.compact = compact_step_kernel
  sim.kernel.analysis = analysis_step_kernel
  sim.kernel.a_buf = a_buf
  sim.kernel.p_buf = p_buf
  sim.kernel.pn_buf = pn_buf
  sim.kernel.pv_buf = pv_buf
  sim.kernel.g_buf = g_buf
  sim.kernel.n_buf = n_buf


def simulation_step(sim: SimulationState) -> None:
  # velocity_step(sim.pressure, sim.velocity_x, sim.velocity_y,
  #               sim.velocity_z, sim.geometry, sim.neighbours)
  # pressure_step(sim.pressure, sim.velocity_x, sim.velocity_y,
  #               sim.velocity_z, sim.geometry, sim.neighbours)
  # analysis_step(sim.pressure, sim.velocity_x, sim.velocity_y,
  #               sim.velocity_z, sim.geometry, sim.analysis)
  cl.enqueue_nd_range_kernel(sim.kernel.queue, sim.kernel.compact, [
                             sim.pressure.size], None)

  for source in sim.sources:
    radial_f = 2.0 * math.pi * source.frequency
    time_active = source.pulses / source.frequency
    break_off = source.start_at + time_active
    active = sim.time >= source.start_at and (
        source.pulses == 0 or sim.time <= break_off)
    rel_time = sim.time - source.start_at
    factor = -1 if source.invert_phase else 1 if active else 0
    radial_t = radial_f * rel_time
    signal = math.sin(radial_t) * factor
    sim.pressure[source.position.w,
                 source.position.h, source.position.d] = signal

  sim.time += DT
  sim.iteration += 1
