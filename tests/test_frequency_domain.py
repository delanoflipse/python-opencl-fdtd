import math

import numpy as np

from conftest import requires_opencl
from fdtd import frequency_domain
from fdtd.parameters import SimulationParameters
from fdtd.scenes import create_scene


def coarse_setup(scheme=(1.0, 1 / 4, 1 / 16)):
  params = SimulationParameters()
  params.set_max_frequency(200)
  params.set_oversampling(4)
  params.set_scheme(*scheme)
  scene = create_scene("shoebox", params)
  grid = scene.build()
  scene.rebuild()
  return params, scene, grid


def test_laplacian_is_symmetric_and_annihilates_constants():
  params, _, grid = coarse_setup()
  operator = frequency_domain.build_operator(grid, params)
  laplacian = operator.laplacian
  assert abs(laplacian - laplacian.T).max() < 1e-12
  assert np.abs(laplacian @ np.ones(operator.size)).max() < 1e-12


def test_only_boundary_cells_have_loss():
  params, _, grid = coarse_setup()
  operator = frequency_domain.build_operator(grid, params)
  beta_k = operator.to_grid(operator.beta_k)
  assert np.all(beta_k[1:-1, 1:-1, 1:-1] == 0.0)
  assert np.all(beta_k[0, :, :] > 0.0)


def test_hard_source_clamps_source_cell():
  params, _, grid = coarse_setup()
  operator = frequency_domain.build_operator(grid, params)
  source = grid.source_set[0]
  field = frequency_domain.solve(operator, 50.0, source, hard_source=True)
  assert abs(field[source] - 1.0) < 1e-6


@requires_opencl
def test_matches_steady_state_fdtd():
  """A sine-driven FDTD run converges to the frequency-domain solution"""
  from fdtd.impulse_generators import SimpleSinoidGenerator
  from fdtd.simulation import Simulation

  frequency = 47.0
  params, scene, grid = coarse_setup()
  source = grid.source_set[0]
  grid.select_source_locations([source])
  sim = Simulation(grid=grid, parameters=params)
  sim.generator = SimpleSinoidGenerator(frequency)
  scene.rebuild()
  sim.sync_read_buffers()
  sim.reset()

  # lock-in over the second half of the run, after the transient decayed
  total = int(2.0 / params.dt)
  accumulated = np.zeros(grid.grid_shape, dtype=np.complex128)
  while sim.iteration < total:
    sim.step(1)
    if sim.iteration >= total // 2:
      accumulated += grid.pressure * np.exp(-2j * math.pi * frequency * sim.time)
  fdtd_field = accumulated / accumulated[source]

  operator = frequency_domain.build_operator(grid, params)
  fdfd_field = frequency_domain.solve(operator, frequency, source, hard_source=True)
  mask = ~np.isnan(fdfd_field)
  error = np.linalg.norm((fdtd_field - fdfd_field)[mask]) / np.linalg.norm(fdfd_field[mask])
  assert error < 1e-2
