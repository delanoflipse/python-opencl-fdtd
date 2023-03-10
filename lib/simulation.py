import numpy as np

import pyopencl as cl
from lib.gpu.kernel_program import SimulationKernelProgram
from lib.grid import SimulationGrid
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
    self.sync_pressure_buffers()

  def reset(self) -> None:
    self.grid.reset_values()
    self.time = 0
    self.iteration = 0
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
    print(
        f'[Storage] Cells: {self.grid.get_cell_size_str()}\tStorage: {self.grid.get_storage_str()} needed')
    print(
        f'[Grid] Listeners: {self.grid.listener_count}\tSources: {self.grid.source_count}')

  def sync_read_buffers(self) -> None:
    args = {
        "is_blocking": False,
    }
    prog = self.program
    grid = self.grid
    queue = self.program.queue
    cl.wait_for_events([
        cl.enqueue_copy(queue, prog.beta_buffer, grid.beta, **args),
        cl.enqueue_copy(queue, prog.geometry_buffer, grid.geometry, **args),
    ])

  def sync_pressure_buffers(self) -> None:
    args = {
        "is_blocking": False,
    }
    prog = self.program
    grid = self.grid
    queue = self.program.queue
    cl.wait_for_events([
        cl.enqueue_copy(queue, prog.pressure_previous_buffer,
                        grid.pressure_previous, **args),
        cl.enqueue_copy(queue, prog.pressure_next_buffer,
                        grid.pressure_next, **args),
        cl.enqueue_copy(queue, prog.pressure_buffer, grid.pressure, **args),
        cl.enqueue_copy(queue, prog.analysis_buffer, grid.analysis, **args),
    ])

  def enqueue_copy(self, dest, src, is_blocking=False, **kwargs) -> cl.Event:
    # kwargs["is_blocking"] = is_blocking
    return cl.enqueue_copy(self.program.queue, dest, src, is_blocking=False)

  def step(self, step_count: int = 1) -> None:
    """Proceed the simulation one or more steps, note: only writes back pressure and analysis values"""
    # initial write from host to device
    prog = self.program
    grid = self.grid
    queue = self.program.queue
    kernel_global_size = [self.grid.pressure.size]
    kernel_global_shape = self.grid.pressure.shape
    # kernel_global_size_shaped = self.grid.pressure.shape
    args = {
        "is_blocking": False,
    }

    # wait_event = [
    #     cl.enqueue_copy(queue, previous_buffer,
    #                     grid.pressure_previous, **args),
    #     cl.enqueue_copy(queue, current_buffer, grid.pressure, **args),
    #     cl.enqueue_copy(queue, prog.analysis_buffer, grid.analysis, **args),
    # ]

    wait_event = []

    for i in range(step_count):
      # get next signal value
      signal = 0.0

      if self.generator is not None:
        signal = self.generator.generate(
            self.time, self.iteration)

      # add samples
      self.signal_set.append(signal)
      self.time_set.append(self.time)

      # set signal value in kernel

      if self.iteration % 3 == 0:
        next_buffer = prog.pressure_next_buffer
        current_buffer = prog.pressure_buffer
        previous_buffer = prog.pressure_previous_buffer
      elif self.iteration % 3 == 1:
        next_buffer = prog.pressure_previous_buffer
        current_buffer = prog.pressure_next_buffer
        previous_buffer = prog.pressure_buffer
      elif self.iteration % 3 == 2:
        next_buffer = prog.pressure_buffer
        current_buffer = prog.pressure_previous_buffer
        previous_buffer = prog.pressure_next_buffer

      self.program.analysis_kernel.set_arg(0, current_buffer)
      self.program.scheme_step_kernel.set_arg(0, previous_buffer)
      self.program.scheme_step_kernel.set_arg(1, current_buffer)
      self.program.scheme_step_kernel.set_arg(2, next_buffer)
      # self.program.step_kernel.set_arg(0, previous_buffer)
      # self.program.step_kernel.set_arg(1, current_buffer)
      # self.program.step_kernel.set_arg(2, next_buffer)

      # run compact step
      # -- SLF optimised
      # self.program.step_kernel.set_arg(10, np.float64(signal))
      # kernel_event1 = cl.enqueue_nd_range_kernel(
      #     queue, prog.step_kernel, kernel_global_size, None, wait_for=wait_event)

      # -- any scheme
      self.program.scheme_step_kernel.set_arg(16, np.float64(signal))

      # set analysis argument
      self.program.analysis_kernel.set_arg(9, np.float64(self.time))

      # stream result into right buffer for next kernel run
      wait_event = [
          cl.enqueue_nd_range_kernel(
              queue,
              prog.scheme_step_kernel,
              kernel_global_size,
              None,
              wait_for=wait_event
          ),
          cl.enqueue_nd_range_kernel(
              queue,
              prog.analysis_kernel,
              kernel_global_size,
              None,
              wait_for=wait_event)
      ]

      # finally, update iteration parameters
      self.time += self.parameters.dt
      self.iteration += 1

    # write back to host
    final_events = [
        cl.enqueue_copy(queue, grid.pressure, current_buffer,
                        wait_for=wait_event, **args),
        cl.enqueue_copy(
            queue, grid.analysis, prog.analysis_buffer, wait_for=wait_event, **args)
    ]

    # make sure event is done before processing data further!
    cl.wait_for_events(final_events)
