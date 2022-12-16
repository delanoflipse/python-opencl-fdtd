import os
import math
import numpy as np
from numba import njit, prange

import pyopencl as cl
from lib.geometry import populate_neighbours
from lib.parameters import DT, DX, LAMBDA_COURANT, MAX_FREQUENCY

WALL_FLAG = 1 << 0
SOURCE_FLAG = 1 << 1
LISTENER_FLAG = 1 << 2


def create_grid(shape, dtype) -> np.ndarray:
  return np.zeros(shape=shape, dtype=dtype)


class KernelProgram(object):
  pass


class SimulationState:
  def __init__(self, shape: tuple[int, int, int]):
    (width, height, depth) = shape
    self.width_parts = math.ceil(width / DX)
    self.height_parts = math.ceil(height / DX)
    self.depth_parts = math.ceil(depth / DX)
    self.grid_size = self.width_parts * self.height_parts * self.depth_parts
    self.grid_shape = (self.width_parts, self.height_parts, self.depth_parts)

    self.geometry = create_grid(self.grid_shape, "int8")
    self.neighbours = create_grid(self.grid_shape, "int8")
    self.pressure = create_grid(self.grid_shape, "float64")
    self.pressure_previous = create_grid(self.grid_shape, "float64")
    self.pressure_next = create_grid(self.grid_shape, "float64")
    self.analysis = create_grid(self.grid_shape, "float64")
    self.time = 0
    self.iteration = 0
    self.signal = 0

  def scale(self, size: float) -> int:
    return int(round(size / DX))

  def setup(self) -> None:
    populate_neighbours(self.geometry, self.neighbours)

    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    platforms = cl.get_platforms()
    ctx = cl.create_some_context(interactive=False)
    print(f'Platform: {platforms[0].name}')
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    r_flag = mf.READ_ONLY | mf.COPY_HOST_PTR
    rw_flag = mf.READ_WRITE | mf.COPY_HOST_PTR
    a_buf = cl.Buffer(ctx, rw_flag, hostbuf=self.analysis)
    pv_buf = cl.Buffer(ctx, r_flag, hostbuf=self.pressure_previous)
    p_buf = cl.Buffer(ctx, r_flag, hostbuf=self.pressure)
    pn_buf = cl.Buffer(ctx, rw_flag, hostbuf=self.pressure_next)
    g_buf = cl.Buffer(ctx, r_flag, hostbuf=self.geometry)
    n_buf = cl.Buffer(ctx, r_flag, hostbuf=self.neighbours)

    dir = os.path.dirname(__file__)
    loc = os.path.join(dir, './kernels/fdtd.cl')
    source = open(loc, 'r').read()
    prg = cl.Program(ctx, source).build()

    # compact step kernel
    compact_step_kernel = prg.compact_step
    compact_step_kernel.set_arg(0, pv_buf)
    compact_step_kernel.set_arg(1, p_buf)
    compact_step_kernel.set_arg(2, pn_buf)
    compact_step_kernel.set_arg(3, g_buf)
    compact_step_kernel.set_arg(4, n_buf)

    compact_step_kernel.set_arg(5, np.uint32(self.width_parts))
    compact_step_kernel.set_arg(6, np.uint32(self.height_parts))
    compact_step_kernel.set_arg(7, np.uint32(self.depth_parts))

    compact_step_kernel.set_arg(8, np.float64(LAMBDA_COURANT))
    compact_step_kernel.set_arg(9, np.float64(1.8))
    compact_step_kernel.set_arg(10, np.float64(0))

    # analysis step kernel
    analysis_step_kernel = prg.analysis_step

    analysis_step_kernel.set_arg(0, p_buf)
    analysis_step_kernel.set_arg(1, a_buf)
    analysis_step_kernel.set_arg(2, g_buf)
    analysis_step_kernel.set_arg(3, np.uint32(self.width_parts))
    analysis_step_kernel.set_arg(4, np.uint32(self.height_parts))
    analysis_step_kernel.set_arg(5, np.uint32(self.depth_parts))
    analysis_step_kernel.set_arg(6, np.float64(DT))

    # setup object
    self.kernel = KernelProgram()
    self.kernel.ctx = ctx
    self.kernel.queue = queue
    self.kernel.compact = compact_step_kernel
    self.kernel.analysis = analysis_step_kernel
    self.kernel.a_buf = a_buf
    self.kernel.p_buf = p_buf
    self.kernel.pn_buf = pn_buf
    self.kernel.pv_buf = pv_buf
    self.kernel.g_buf = g_buf
    self.kernel.n_buf = n_buf

  def step(self) -> None:
    # determine source value
    sigma = 0.0004
    variance = sigma * sigma
    frequency = 20 + MAX_FREQUENCY
    t0 = 0.005 + 4/frequency
    t = self.time - t0
    cos_factor = math.cos(2 * math.pi * t * frequency)
    # sqrt_variance = math.sqrt(2 * math.pi * variance)
    exp_factor = math.exp(-(t * t) / (2 * variance))
    signal = cos_factor * (exp_factor)

    cl.enqueue_copy(self.kernel.queue, self.kernel.pv_buf,
                    self.pressure_previous,
                    is_blocking=False)
    cl.enqueue_copy(self.kernel.queue, self.kernel.p_buf, self.pressure,
                    is_blocking=False)
    cl.enqueue_copy(self.kernel.queue, self.kernel.a_buf, self.analysis)

    self.kernel.compact.set_arg(10, np.float64(signal))
    # TODO: multiple steps at once, move buffers between step
    cl.enqueue_nd_range_kernel(self.kernel.queue, self.kernel.compact, [
        self.pressure.size], None)
    cl.enqueue_nd_range_kernel(self.kernel.queue, self.kernel.analysis, [
        self.pressure.size], None)
    cl.enqueue_copy(self.kernel.queue,
                    self.pressure_previous, self.kernel.p_buf,
                    is_blocking=False)
    cl.enqueue_copy(self.kernel.queue, self.pressure, self.kernel.pn_buf,
                    is_blocking=False)
    cl.enqueue_copy(self.kernel.queue, self.analysis,
                    self.kernel.a_buf)

    self.time += DT
    self.iteration += 1
    self.signal = signal
