import math
import numpy as np

import pyopencl as cl
from numba import njit, prange
from fdtd.gpu.kernel_program import SimulationKernelProgram
from fdtd.grid import WALL_FLAG, SimulationGrid
from fdtd.impulse_generators import ImpulseGenerator
from fdtd.parameters import SimulationParameters

# 1 / p_ref^2 with p_ref = 20 uPa
SQUARED_REFERENCE_SCALE = 25e8


@njit(parallel=True)
def write_analysis(squared_sum: np.ndarray, geometry: np.ndarray, analysis: np.ndarray, rms_index: int, leq_index: int, rms_scale: float, time_elapsed: float) -> None:
  """Convert the per-cell sum of squared pressures into the RMS and Leq analysis values"""
  for i in prange(squared_sum.shape[0]):
    if geometry[i] & WALL_FLAG > 0:
      analysis[i, leq_index] = math.nan
      continue

    rms_sum = squared_sum[i] * rms_scale
    analysis[i, rms_index] = rms_sum

    if time_elapsed > 0:
      rms_value = (1.0 / time_elapsed) * rms_sum
      analysis[i, leq_index] = 10.0 * \
          math.log10(rms_value) if rms_value > 0 else 0.0


class Simulation:
  """Handles the simulation state and can perform a step"""

  def __init__(self, parameters: SimulationParameters, grid: SimulationGrid):
    self.parameters = parameters
    self.grid = grid
    self.program = SimulationKernelProgram(grid)
    self.generator: ImpulseGenerator = None
    self.time = 0
    self.iteration = 0
    # time of the last analysed iteration, used to average the RMS sum
    self.analysis_time = 0.0
    self.signal_set = []
    self.time_set = []
    self.rms_scale = parameters.dt * SQUARED_REFERENCE_SCALE
    self.pressure_squared_sum = np.zeros(grid.grid_size, dtype=np.float64)
    self.pressure_readback = np.empty(
        grid.grid_shape, dtype=self.program.real_dtype)
    self.sync_pressure_buffers()

  def reset(self) -> None:
    self.grid.reset_values()
    self.time = 0
    self.iteration = 0
    self.analysis_time = 0.0
    self.signal_set = []
    self.time_set = []
    self.sync_pressure_buffers()

  def print_statistics(self) -> None:
    print(f'Kernel platform: {self.program.platforms[0].name}')
    print(
        f'[Params] w: {self.grid.width_parts}\th:{self.grid.height_parts}\td:{self.grid.depth_parts}')
    print(f'[Params] a={self.parameters.param_a:0.2f}\tb:{self.parameters.param_b:0.2f}\tlambda:{self.parameters.lambda_courant:0.2f}')
    print(f'[Params] d1={self.parameters.arg_d1:0.2f}\td2={self.parameters.arg_d2:0.2f}\td3={self.parameters.arg_d3:0.2f}\td4={self.parameters.arg_d4:0.2f}')
    print(f'[Params] {self.parameters.sampling_frequency}hz target.\t{self.parameters.dt_hz:0.0f}hz speed.\t{self.parameters.dx * 1000:.2f}mm size. ')
    print(f'[Params] precision: {self.parameters.precision}')
    print(
        f'[Storage] Cells: {self.grid.get_cell_size_str()}\tStorage: {self.grid.get_storage_str()} needed')
    print(
        f'[Grid] Listeners: {self.grid.listener_count}\tSources: {self.grid.source_count}')

  def sync_read_buffers(self) -> None:
    """Upload beta and geometry, e.g. after a scene rebuild or a source change"""
    prog = self.program
    grid = self.grid
    queue = self.program.queue
    cl.wait_for_events([
        cl.enqueue_copy(queue, prog.beta_buffer,
                        prog.to_device(grid.beta), is_blocking=False),
        cl.enqueue_copy(queue, prog.geometry_buffer,
                        grid.geometry, is_blocking=False),
    ])

  def sync_pressure_buffers(self) -> None:
    """Upload the host pressure and analysis state"""
    prog = self.program
    grid = self.grid
    queue = self.program.queue
    rms_index = grid.analysis_keys['RMS']
    self.pressure_squared_sum[:] = grid.analysis[...,
                                                 rms_index].ravel() / self.rms_scale
    cl.wait_for_events([
        cl.enqueue_copy(queue, prog.pressure_previous_buffer,
                        prog.to_device(grid.pressure_previous), is_blocking=False),
        cl.enqueue_copy(queue, prog.pressure_next_buffer,
                        prog.to_device(grid.pressure_next), is_blocking=False),
        cl.enqueue_copy(queue, prog.pressure_buffer,
                        prog.to_device(grid.pressure), is_blocking=False),
        cl.enqueue_copy(queue, prog.pressure_squared_sum_buffer,
                        self.pressure_squared_sum, is_blocking=False),
    ])

  def enqueue_copy(self, dest, src, is_blocking=False, **kwargs) -> cl.Event:
    # kwargs["is_blocking"] = is_blocking
    return cl.enqueue_copy(self.program.queue, dest, src, is_blocking=False)

  def step(self, step_count: int = 1) -> None:
    """Proceed the simulation one or more steps, note: only writes back pressure and analysis values"""
    prog = self.program
    queue = prog.queue
    kernel = prog.step_kernel
    static_args = prog.step_static_args
    global_size = (self.grid.grid_size,)
    dt = self.parameters.dt
    current_buffer = None

    # the queue is in-order, so each step waits for the previous one
    for _ in range(step_count):
      # get next signal value
      signal = 0.0

      if self.generator is not None:
        signal = self.generator.generate(
            self.time, self.iteration)

      # add samples
      self.signal_set.append(signal)
      self.time_set.append(self.time)

      previous_buffer, current_buffer, next_buffer = prog.buffer_rotations[
          self.iteration % 3]
      kernel(queue, global_size, None, previous_buffer,
             current_buffer, next_buffer, *static_args, signal)

      # finally, update iteration parameters
      self.analysis_time = self.time
      self.time += dt
      self.iteration += 1

    if current_buffer is not None:
      self.read_back(current_buffer)

  def read_back(self, current_buffer: cl.Buffer) -> None:
    """Copy pressure and analysis values back to the host grid"""
    prog = self.program
    grid = self.grid
    queue = prog.queue
    cl.wait_for_events([
        cl.enqueue_copy(queue, self.pressure_readback,
                        current_buffer, is_blocking=False),
        cl.enqueue_copy(queue, self.pressure_squared_sum,
                        prog.pressure_squared_sum_buffer, is_blocking=False),
    ])
    grid.pressure[...] = self.pressure_readback

    write_analysis(
        self.pressure_squared_sum,
        grid.geometry.reshape(-1),
        grid.analysis.reshape(grid.grid_size, grid.analysis_values),
        grid.analysis_keys['RMS'],
        grid.analysis_keys['LEQ'],
        self.rms_scale,
        # always a float: an int (time 0) would trigger a second numba compile
        float(self.analysis_time),
    )
