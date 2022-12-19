from asyncio import wait_for
import os
import math
import numpy as np
from numba import njit, prange

import pyopencl as cl
from lib.geometry import populate_neighbours
from lib.impulses import ImpulseGenerator
from lib.parameters import DT, DX, LAMBDA_COURANT, MAX_FREQUENCY, MIN_FREQUENCY

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
    self.rms = create_grid(self.grid_shape, "float64")
    self.time = 0
    self.iteration = 0
    self.beta = 0.5
    self.signal_set = []
    self.time_set = []
    self.generator: ImpulseGenerator = None
    self.signal_frequency = MIN_FREQUENCY

  def scale(self, size: float) -> int:
    return int(round(size / DX))

  def set_frequency(self, frequency: float) -> None:
    self.signal_frequency = frequency

  def set_beta(self, beta: float) -> None:
    # https://www.acoustic-supplies.com/absorption-coefficient-chart/
    if hasattr(self, "kernel"):
      self.kernel.compact.set_arg(9, np.float64(beta))
    self.beta = beta

  def setup(self) -> None:
    populate_neighbours(self.geometry, self.neighbours)

    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    platforms = cl.get_platforms()
    ctx = cl.create_some_context(interactive=False)
    print(f'Platform: {platforms[0].name}')
    queue = cl.CommandQueue(ctx)
    r_flag = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
    rw_flag = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
    a_buf = cl.Buffer(ctx, rw_flag, hostbuf=self.analysis)
    rms_buf = cl.Buffer(ctx, rw_flag, hostbuf=self.rms)
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
    compact_step_kernel.set_arg(9, np.float64(self.beta))
    compact_step_kernel.set_arg(10, np.float64(0))

    # analysis step kernel
    analysis_step_kernel = prg.analysis_step

    analysis_step_kernel.set_arg(0, p_buf)
    analysis_step_kernel.set_arg(1, a_buf)
    analysis_step_kernel.set_arg(2, rms_buf)
    analysis_step_kernel.set_arg(3, g_buf)
    analysis_step_kernel.set_arg(4, np.uint32(self.width_parts))
    analysis_step_kernel.set_arg(5, np.uint32(self.height_parts))
    analysis_step_kernel.set_arg(6, np.uint32(self.depth_parts))
    analysis_step_kernel.set_arg(7, np.float64(DT))

    # setup object
    self.kernel = KernelProgram()
    self.kernel.ctx = ctx
    self.kernel.queue = queue
    self.kernel.compact = compact_step_kernel
    self.kernel.analysis = analysis_step_kernel
    self.kernel.a_buf = a_buf
    self.kernel.rms_buf = rms_buf
    self.kernel.p_buf = p_buf
    self.kernel.pn_buf = pn_buf
    self.kernel.pv_buf = pv_buf
    self.kernel.g_buf = g_buf
    self.kernel.n_buf = n_buf

  def step(self, count: int = 1) -> None:
    #  initial write from host to device
    cl.enqueue_copy(self.kernel.queue, self.kernel.pv_buf,
                    self.pressure_previous,
                    is_blocking=False)
    cl.enqueue_copy(self.kernel.queue, self.kernel.p_buf, self.pressure,
                    is_blocking=False)
    cl.enqueue_copy(self.kernel.queue, self.kernel.a_buf, self.analysis,
                    is_blocking=False)
    last_event = cl.enqueue_copy(self.kernel.queue, self.kernel.rms_buf, self.rms,
                                 is_blocking=False)
    wait_event = last_event
    for i in range(count):
      signal = 0.0

      if self.generator != None:
        signal = self.generator.generate(
            self.time, self.iteration, self.signal_frequency)

      self.kernel.compact.set_arg(10, np.float64(signal))

      # add samples
      self.signal_set.append(signal)
      self.time_set.append(self.time)

      # run compact step
      kernel_event1 = cl.enqueue_nd_range_kernel(self.kernel.queue, self.kernel.compact, [
          self.pressure.size], None, wait_for=[wait_event])

      # copy result for next kernel run
      wait_for_list = []
      if (i < count - 1):
        copy_event1 = cl.enqueue_copy(self.kernel.queue,
                                      self.kernel.pv_buf, self.kernel.p_buf, wait_for=[kernel_event1])

        copy_event2 = cl.enqueue_copy(self.kernel.queue,
                                      self.kernel.p_buf, self.kernel.pn_buf, wait_for=[kernel_event1])
        wait_for_list = [copy_event1, copy_event2]

      # set iteration argument
      self.kernel.analysis.set_arg(8, np.uint32(self.iteration + 1))

      # run analysis
      kernel_event2 = cl.enqueue_nd_range_kernel(self.kernel.queue, self.kernel.analysis, [
          self.pressure.size], None, wait_for=wait_for_list)

      wait_event = kernel_event2
      self.time += DT
      self.iteration += 1

    # write back to host
    cl.enqueue_copy(self.kernel.queue,
                    self.pressure_previous, self.kernel.p_buf,
                    is_blocking=False, wait_for=[wait_event])
    cl.enqueue_copy(self.kernel.queue, self.pressure, self.kernel.pn_buf,
                    is_blocking=False, wait_for=[wait_event])
    cl.enqueue_copy(self.kernel.queue, self.analysis,
                    self.kernel.a_buf, wait_for=[wait_event],
                    is_blocking=False)
    final_event = cl.enqueue_copy(self.kernel.queue, self.rms,
                                  self.kernel.rms_buf, wait_for=[wait_event],
                                  is_blocking=False)
    final_event.wait()
