
import math
import numpy as np
from lib.grid import LISTENER_FLAG
from lib.impulse_generators import GaussianModulatedImpulseGenerator
from lib.simulation import Simulation
from numba import njit, prange
from typing import Tuple


def frequency_sweep(sim: Simulation) -> np.ndarray:
  sweep_analysis = sim.grid.create_grid("float64")
  runtime_steps = int(0.5 / sim.parameters.dt)
  for f in range(sim.parameters.min_frequency, sim.parameters.max_frequency):
    sim.generator = GaussianModulatedImpulseGenerator(f)
    sim.grid.reset_values()
    sim.step(runtime_steps)
    # TODO process values
  return sweep_analysis


@njit(parallel=True)
def run_sweep_analysis(step_analysis: np.ndarray, summation: np.ndarray, sum_sqr: np.ndarray, deviation: np.ndarray, ranking: np.ndarray, analysis_value: int, n: int) -> None:
  """Set neighbour flags for geometry"""
  _max = -1e99
  _min = 1e99
  for w in prange(step_analysis.shape[0]):
    for h in prange(step_analysis.shape[1]):
      for d in prange(step_analysis.shape[2]):
        spl_value = step_analysis[w, h, d, analysis_value]
        if math.isnan(spl_value):
          deviation[w, h, d] = math.nan
          continue
        _m = summation[w, h, d]
        _new_m = _m + (spl_value - _m)/n
        summation[w, h, d] = _new_m
        sum_sqr[w, h, d] += (spl_value - _new_m) * (spl_value - _m)
        _dev = sum_sqr[w, h, d] / n
        _max = max(_dev, _max)
        _min = min(_dev, _min)
        deviation[w, h, d] = _dev

  _range = _max - _min
  for w in prange(step_analysis.shape[0]):
    for h in prange(step_analysis.shape[1]):
      for d in prange(step_analysis.shape[2]):
        standart_dev_leq = deviation[w, h, d]
        if math.isnan(standart_dev_leq):
          ranking[w, h, d] = 0
          continue
        diff = (standart_dev_leq - _min) / _range
        r = 1 - diff
        ranking[w, h, d] = r


@njit(parallel=True)
def get_avg_dev(standard_deviation: np.ndarray, flags: np.ndarray) -> float:
  """Set neighbour flags for geometry"""
  _sum = 0.0
  _count = 0
  for w in prange(standard_deviation.shape[0]):
    for h in prange(standard_deviation.shape[1]):
      for d in prange(standard_deviation.shape[2]):
        cell_flags = flags[w, h, d]
        if cell_flags & LISTENER_FLAG == 0:
          continue

        deviation = standard_deviation[w, h, d]
        if math.isnan(deviation):
          continue
        _sum += deviation
        _count += 1
  if _count == 0:
    return 0.0

  return _sum / _count


@njit(parallel=True)
def get_avg_spl(analytical_values: np.ndarray, flags: np.ndarray, leq_index: int) -> Tuple[float, float, float]:
  """Set neighbour flags for geometry"""
  _sum = 0.0
  _count = 0.0
  _min: float = 1e999
  _max: float = -1e999

  for w in prange(analytical_values.shape[0]):
    for h in prange(analytical_values.shape[1]):
      for d in prange(analytical_values.shape[2]):
        cell_flags = flags[w, h, d]
        if cell_flags & LISTENER_FLAG == 0:
          continue

        v_l_eq: float = analytical_values[w, h, d, leq_index]
        if math.isnan(v_l_eq):
          continue
        _sum += v_l_eq
        _min = min(_min, v_l_eq)
        _max = max(_max, v_l_eq)
        _count += 1.0
  if _count == 0.0:
    return (0.0, 0.0, 0.0)

  return (_sum / _count, _min, _max)
