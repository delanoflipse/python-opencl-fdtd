from asyncio import wait_for
import os
import math
import numpy as np
from numba import njit, prange

import pyopencl as cl
from lib.gpu.kernel_program import SimulationKernelProgram
from lib.grid import SimulationGrid, populate_neighbours
from lib.impulse_generators import ImpulseGenerator
from lib.parameters import SimulationParameters


class Simulation:
  """Handles the simulation state and can perform a step"""

  def __init__(self, parameters: SimulationParameters, grid: SimulationGrid):
    self.parameters = parameters
    self.grid = grid
    self.program = SimulationKernelProgram(grid)
    self.generator: ImpulseGenerator = None
    self.time = 0
    self.iteration = 0
    self.signal_set = []
    self.time_set = []

  def reset(self) -> None:
    self.grid.reset_values()
    self.time = 0
    self.iteration = 0
    self.signal_set = []
    self.time_set = []

  def print_statistics(self) -> None:
    print(f'Kernel platform: {self.program.platforms[0].name}')
    print(
        f'[Params] w: {self.grid.width_parts}\th:{self.grid.height_parts}\td:{self.grid.depth_parts}')
    print(f'[Params] {self.parameters.sampling_frequency}hz target.\t{self.parameters.dt_hz:0.0f}hz speed.\t{self.parameters.dx * 1000:.2f}mm size. ')
    print(
        f'[Storage] Cells: {self.grid.get_cell_size_str()}\tStorage: {self.grid.get_storage_str()} needed')
    print(
        f'[Grid] Listeners: {self.grid.listener_count}\tSources: {self.grid.source_count}')

  def sync_read_buffers(self) -> None:
    cl.enqueue_copy(self.program.queue,
                    self.program.beta_buffer, self.grid.beta)
    cl.enqueue_copy(self.program.queue,
                    self.program.geometry_buffer, self.grid.geometry)

  def enqueue_copy(self, dest, src, is_blocking=False, **kwargs) -> cl.Event:
    # kwargs["is_blocking"] = is_blocking
    return cl.enqueue_copy(self.program.queue, dest, src, is_blocking=False)

  def step(self, step_count: int = 1) -> None:
    """Proceed the simulation one or more steps"""
    # initial write from host to device
    prog = self.program
    grid = self.grid
    queue = self.program.queue
    kernel_global_size = [self.grid.pressure.size]
    # kernel_global_size_shaped = self.grid.pressure.shape
    args = {
        "is_blocking": False,
    }

    wait_event = [
        cl.enqueue_copy(queue, prog.pressure_previous_buffer,
                        grid.pressure_previous, **args),
        cl.enqueue_copy(queue, prog.pressure_buffer, grid.pressure, **args),
        cl.enqueue_copy(queue, prog.analysis_buffer, grid.analysis, **args),
    ]

    for i in range(step_count):
      # get next signal value
      signal = 0.0

      if self.generator is not None:
        signal = self.generator.generate(
            self.time, self.iteration)

      # set signal value in kernel
      self.program.step_kernel.set_arg(10, np.float64(signal))
      # set iteration argument
      self.program.analysis_kernel.set_arg(9, np.float64(self.time))

      # add samples
      self.signal_set.append(signal)
      self.time_set.append(self.time)

      # run compact step
      kernel_event1 = cl.enqueue_nd_range_kernel(
          queue, prog.step_kernel, kernel_global_size, None, wait_for=wait_event)

      # stream result into right buffer for next kernel run
      kernel_wait = [kernel_event1]
      buffer_write_wait = [
          cl.enqueue_copy(queue, prog.pressure_previous_buffer,
                          prog.pressure_buffer, wait_for=kernel_wait),
          cl.enqueue_copy(queue, prog.pressure_buffer,
                          prog.pressure_next_buffer, wait_for=kernel_wait),
      ]
      
      # run analysis on previous values
      kernel_event2 = cl.enqueue_nd_range_kernel(
          queue, prog.analysis_kernel, kernel_global_size, None, wait_for=buffer_write_wait)
      wait_event = [kernel_event2]
      # finally, update iteration parameters
      self.time += self.parameters.dt
      self.iteration += 1

    # write back to host
    cl.enqueue_copy(queue, grid.pressure_previous,
                    prog.pressure_buffer, wait_for=wait_event, **args)
    cl.enqueue_copy(queue, grid.pressure, prog.pressure_next_buffer,
                    wait_for=wait_event, **args)
    final_event = cl.enqueue_copy(
        queue, grid.analysis, prog.analysis_buffer, wait_for=wait_event, **args)

    # make sure event is done before processing data further!
    final_event.wait()
