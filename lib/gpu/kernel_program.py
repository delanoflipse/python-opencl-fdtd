import os
import pyopencl as cl
import numpy as np

from ..grid import SimulationGrid

RELATIVE_PROGRAM_FILE = "accelerated_fdtd.cl"

PRECISION_DTYPES = {
    "float": np.float32,
    "double": np.float64,
}


class SimulationKernelProgram:
  """Loads and setup gpu accelerated code"""

  def print_devices_and_platforms(self) -> None:
    for platform in cl.get_platforms():
      print(f'platform: {platform.name}')
      for device in platform.get_devices():
        print(f'- device: {device.name}')

  def __init__(self, grid: SimulationGrid):
    if not grid.is_build:
      raise Exception("Please build the grid before building the program")

    if grid.grid_size >= 2 ** 32:
      raise Exception("Grid too large: the kernel uses 32-bit cell indices")

    params = grid.parameters
    self.parameters = params
    self.real_dtype = PRECISION_DTYPES[params.precision]

    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    self.platforms = cl.get_platforms()

    self.ctx = cl.create_some_context(interactive=False)

    self.queue = cl.CommandQueue(self.ctx)
    r_flag = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
    rw_flag = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR

    # the three pressure buffers rotate roles every step, so all are written
    self.pressure_previous_buffer = cl.Buffer(
        self.ctx, rw_flag, hostbuf=self.to_device(grid.pressure_previous))
    self.pressure_buffer = cl.Buffer(
        self.ctx, rw_flag, hostbuf=self.to_device(grid.pressure))
    self.pressure_next_buffer = cl.Buffer(
        self.ctx, rw_flag, hostbuf=self.to_device(grid.pressure_next))
    self.pressure_squared_sum_buffer = cl.Buffer(
        self.ctx, rw_flag, hostbuf=np.zeros(grid.grid_size, dtype=np.float64))
    self.geometry_buffer = cl.Buffer(self.ctx, r_flag, hostbuf=grid.geometry)
    self.neighbours_buffer = cl.Buffer(
        self.ctx, r_flag, hostbuf=grid.neighbours)
    self.beta_buffer = cl.Buffer(
        self.ctx, r_flag, hostbuf=self.to_device(grid.beta))

    # (previous, current, next) buffers for iteration % 3
    self.buffer_rotations = [
        (self.pressure_previous_buffer, self.pressure_buffer, self.pressure_next_buffer),
        (self.pressure_buffer, self.pressure_next_buffer, self.pressure_previous_buffer),
        (self.pressure_next_buffer, self.pressure_previous_buffer, self.pressure_buffer),
    ]

    file_directory = os.path.dirname(__file__)
    loc = os.path.join(file_directory, RELATIVE_PROGRAM_FILE)
    source = ""
    with open(loc, encoding="utf-8") as file:
      source = file.read()

    # scheme coefficients are compile-time constants, so unused neighbour
    # sets are compiled out
    build_options = [
        f"-DREAL={params.precision}",
        f"-DD1={float(params.arg_d1)!r}",
        f"-DD2={float(params.arg_d2)!r}",
        f"-DD3={float(params.arg_d3)!r}",
        f"-DLAMBDA={float(params.lambda_courant)!r}",
        f"-DUSE_D1={int(params.arg_d1 != 0.0)}",
        f"-DUSE_D2={int(params.arg_d2 != 0.0)}",
        f"-DUSE_D3={int(params.arg_d3 != 0.0)}",
    ]
    self.cl_program = cl.Program(self.ctx, source).build(options=build_options)

    # Arguments are passed on every call; typed scalars avoid the slow
    # Kernel.set_arg path for numpy scalars (~18us per call).
    # args: previous, current, next, betas, geometry, neighbours,
    #       pressure_squared_sum, size, h_stride, w_stride, signal
    self.step_kernel = cl.Kernel(self.cl_program, "fdtd_step")
    self.step_kernel.set_scalar_arg_dtypes(
        [None] * 7 + [np.uint32] * 3 + [self.real_dtype])
    self.step_static_args = (
        self.beta_buffer,
        self.geometry_buffer,
        self.neighbours_buffer,
        self.pressure_squared_sum_buffer,
        grid.grid_size,
        grid.depth_parts,
        grid.depth_parts * grid.height_parts,
    )

  def to_device(self, array: np.ndarray) -> np.ndarray:
    """Host array in the device precision (no copy if it already matches)"""
    return np.ascontiguousarray(array, dtype=self.real_dtype)
