"""End-to-end tests on a real OpenCL device (skipped without one)."""
import csv

import numpy as np
import pytest

from conftest import requires_opencl
from fdtd.analysis.frequency_sweep import get_avg_spl
from fdtd.impulse_generators import SimpleSinoidGenerator
from fdtd.parameters import SimulationParameters
from fdtd.scenes import create_scene
from fdtd.sweep import SweepConfig, run_full_sweep

pytestmark = requires_opencl


def simulate_leq(precision: str, steps: int = 400) -> float:
  from fdtd.simulation import Simulation

  params = SimulationParameters()
  params.set_max_frequency(200)
  params.set_oversampling(6)
  params.set_precision(precision)
  params.set_signal_frequency(100.0)
  scene = create_scene("shoebox", params)
  grid = scene.build()
  grid.select_source_locations([grid.source_set[0]])
  sim = Simulation(params, grid)
  sim.generator = SimpleSinoidGenerator(params.signal_frequency)
  sim.reset()
  sim.sync_read_buffers()
  sim.step(steps)

  assert np.isfinite(grid.pressure).all()
  avg_spl, _, _ = get_avg_spl(grid.analysis, grid.geometry, grid.analysis_keys["LEQ"])
  return avg_spl


def test_sound_reaches_listeners():
  assert simulate_leq("float") > 0.0


def test_single_and_double_precision_agree():
  assert simulate_leq("float") == pytest.approx(simulate_leq("double"), abs=0.05)


def test_full_sweep_writes_outputs(tmp_path):
  result = run_full_sweep(SweepConfig(
      scene="shoebox", simulated_time=0.02, oversampling=6, bands=1,
      output_dir=tmp_path, write_chart=False))

  assert len(result.deviations) == len(result.spl_per_source) > 0
  (csv_file,) = tmp_path.glob("*.csv")
  with open(csv_file, encoding="utf-8", newline="") as file:
    rows = list(csv.reader(file))
  assert len(rows) == len(result.deviations) + 1
  assert len(list(tmp_path.glob("*.log"))) == 1
