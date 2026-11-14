import math

import pytest

from fdtd.parameters import SimulationParameters
from fdtd.physical_constants import C_AIR


def test_defaults_derive_grid_spacing_and_time_step():
  params = SimulationParameters()
  assert params.sampling_frequency == params.max_frequency * params.oversampling
  assert params.dx == pytest.approx(C_AIR / params.sampling_frequency)
  # Courant number: c * dt / dx
  assert C_AIR * params.dt / params.dx == pytest.approx(1 / math.sqrt(3))


def test_setters_recalculate_derived_values():
  params = SimulationParameters()
  dx = params.dx
  params.set_oversampling(params.oversampling * 2)
  assert params.dx == pytest.approx(dx / 2)
  params.set_max_frequency(params.max_frequency * 2)
  assert params.dx == pytest.approx(dx / 4)


def test_standard_leapfrog_scheme_coefficients():
  params = SimulationParameters()
  params.set_scheme(1 / math.sqrt(3), 0.0, 0.0)
  assert params.arg_d1 == pytest.approx(1 / 3)
  assert params.arg_d2 == 0.0
  assert params.arg_d3 == 0.0
  assert params.arg_d4 == pytest.approx(0.0)


def test_precision_is_validated():
  params = SimulationParameters()
  params.set_precision("double")
  assert params.precision == "double"
  with pytest.raises(ValueError):
    params.set_precision("half")
