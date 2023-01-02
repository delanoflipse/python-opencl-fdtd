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

  def step(self, count: int = 1) -> None:
    """Proceed the simulation one or more steps"""
    # initial write from host to device
    cl.enqueue_copy(self.program.queue, self.program.pressure_previous_buffer,
                    self.grid.pressure_previous,
                    is_blocking=False)
    cl.enqueue_copy(self.program.queue, self.program.pressure_buffer, self.grid.pressure,
                    is_blocking=False)
    cl.enqueue_copy(self.program.queue, self.program.analysis_buffer, self.grid.analysis,
                    is_blocking=False)
    last_event = cl.enqueue_copy(
        self.program.queue, self.program.rms_buffer, self.grid.rms, is_blocking=False)

    # iteration loop
    wait_event = [last_event]

    for i in range(count):
      # get next signal value
      signal = 0.0

      if self.generator is not None:
        signal = self.generator.generate(
            self.time, self.iteration)

      # set signal value in kernel
      self.program.step_kernel.set_arg(10, np.float64(signal))

      # add samples
      self.signal_set.append(signal)
      self.time_set.append(self.time)

      # run compact step
      kernel_event1 = cl.enqueue_nd_range_kernel(self.program.queue, self.program.step_kernel, [
          self.grid.pressure.size], None, wait_for=wait_event)

      # stream result into right buffer for next kernel run
      wait_for_list = []
      if (i < count - 1):
        copy_event1 = cl.enqueue_copy(self.program.queue,
                                      self.program.pressure_previous_buffer, self.program.pressure_buffer, wait_for=[kernel_event1])

        copy_event2 = cl.enqueue_copy(self.program.queue,
                                      self.program.pressure_buffer, self.program.pressure_next_buffer, wait_for=[kernel_event1])
        wait_for_list = [copy_event1, copy_event2]

      # set iteration argument
      self.program.analysis_kernel.set_arg(10, np.uint32(self.iteration + 1))

      # run analysis
      kernel_event2 = cl.enqueue_nd_range_kernel(self.program.queue, self.program.analysis_kernel, [
          self.grid.pressure.size], None, wait_for=wait_for_list)

      wait_event = [kernel_event2]

      # finally, update iteration parameters
      self.time += self.parameters.dt
      self.iteration += 1

    # write back to host
    cl.enqueue_copy(self.program.queue,
                    self.grid.pressure_previous, self.program.pressure_buffer,
                    is_blocking=False, wait_for=wait_event)
    cl.enqueue_copy(self.program.queue, self.grid.pressure, self.program.pressure_next_buffer,
                    is_blocking=False, wait_for=wait_event)
    cl.enqueue_copy(self.program.queue, self.grid.analysis,
                    self.program.analysis_buffer, wait_for=wait_event,
                    is_blocking=False)
    final_event = cl.enqueue_copy(self.program.queue, self.grid.rms,
                                  self.program.rms_buffer, wait_for=wait_event,
                                  is_blocking=False)

    # make sure event is done before processing data further!
    final_event.wait()
