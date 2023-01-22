
import os
import math
import pyopencl as cl
import numpy as np

from lib.physical_constants import RHO
from ..grid import SimulationGrid

RELATIVE_PROGRAM_FILE = "accelerated_fdtd.cl"


class SimulationKernelProgram:
  """Loads and setup gpu accelerated code"""

  def __init__(self, grid: SimulationGrid):
    if not grid.is_build:
      raise Exception("Please build the grid before building the program")

    params = grid.parameters
    self.parameters = params

    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    self.platforms = cl.get_platforms()
    for platform in cl.get_platforms():
      print(f'platform: {platform.name}')
      for device in platform.get_devices():
        print(f'- device: {device.name}')

    self.ctx = cl.create_some_context(interactive=False)

    self.queue = cl.CommandQueue(self.ctx)
    r_flag = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
    rw_flag = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR

    self.analysis_buffer = cl.Buffer(self.ctx, rw_flag, hostbuf=grid.analysis)
    self.pressure_previous_buffer = cl.Buffer(
        self.ctx, r_flag, hostbuf=grid.pressure_previous)
    self.pressure_buffer = cl.Buffer(self.ctx, r_flag, hostbuf=grid.pressure)
    self.pressure_next_buffer = cl.Buffer(
        self.ctx, rw_flag, hostbuf=grid.pressure_next)
    self.geometry_buffer = cl.Buffer(self.ctx, r_flag, hostbuf=grid.geometry)
    self.neighbours_buffer = cl.Buffer(
        self.ctx, r_flag, hostbuf=grid.neighbours)
    self.beta_buffer = cl.Buffer(
        self.ctx, r_flag, hostbuf=grid.beta)

    file_directory = os.path.dirname(__file__)
    loc = os.path.join(file_directory, RELATIVE_PROGRAM_FILE)
    source = ""
    with open(loc, encoding="utf-8") as file:
      source = file.read()

    prg = cl.Program(self.ctx, source).build()

    # compact step kernel
    self.step_kernel = prg.compact_step
    self.step_kernel.set_arg(0, self.pressure_previous_buffer)
    self.step_kernel.set_arg(1, self.pressure_buffer)
    self.step_kernel.set_arg(2, self.pressure_next_buffer)
    self.step_kernel.set_arg(3, self.beta_buffer)
    self.step_kernel.set_arg(4, self.geometry_buffer)
    self.step_kernel.set_arg(5, self.neighbours_buffer)

    self.step_kernel.set_arg(6, np.uint32(grid.width_parts))
    self.step_kernel.set_arg(7, np.uint32(grid.height_parts))
    self.step_kernel.set_arg(8, np.uint32(grid.depth_parts))

    self.step_kernel.set_arg(9, np.float64(params.lambda_courant))
    self.step_kernel.set_arg(10, np.float64(0))

    # analysis step kernel
    self.analysis_kernel = prg.analysis_step

    self.analysis_kernel.set_arg(0, self.pressure_next_buffer)
    self.analysis_kernel.set_arg(1, self.analysis_buffer)
    self.analysis_kernel.set_arg(2, self.geometry_buffer)
    self.analysis_kernel.set_arg(3, np.uint32(grid.width_parts))
    self.analysis_kernel.set_arg(4, np.uint32(grid.height_parts))
    self.analysis_kernel.set_arg(5, np.uint32(grid.depth_parts))
    self.analysis_kernel.set_arg(6, np.uint32(grid.analysis_values))
    self.analysis_kernel.set_arg(7, np.float64(RHO))
    self.analysis_kernel.set_arg(8, np.float64(params.dt))
    self.analysis_kernel.set_arg(9, np.float64(0))
