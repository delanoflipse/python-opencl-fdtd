"""Module Parameters contains all possible impulse generators"""
import math

from lib.physical_constants import C_AIR

SQRT_3 = math.sqrt(3)
EPSILON = 1e-12


def eps(value: float) -> float:
  if abs(value) < EPSILON:
    return 0.0
  return value


class SimulationParameters:
  def __init__(self):
    self.oversampling = 16
    self.max_frequency = 200.0
    self.lambda_courant = 1 / math.sqrt(3)
    self.param_a = 0.0
    self.param_b = 0.0
    self.signal_frequency = 200.0
    self.recalc()

  def recalc(self) -> None:
    self.lambda_2 = self.lambda_courant * self.lambda_courant
    self.arg_d1 = eps(self.lambda_2 *
                      (1.0 - 4.0 * self.param_a + 4.0 * self.param_b))
    self.arg_d2 = eps(self.lambda_2 * (self.param_a - 2.0 * self.param_b))
    self.arg_d3 = eps(self.lambda_2 * self.param_b)
    self.arg_d4 = eps(2.0 - 6.0 * self.lambda_2 + 12.0 * self.lambda_2 *
                      self.param_a - 8.0 * self.param_b * self.lambda_2)

    self.sampling_frequency = self.max_frequency * self.oversampling
    self.min_wavelength = C_AIR / self.sampling_frequency
    self.dx = self.min_wavelength
    self.dt = (self.dx * self.lambda_courant) / C_AIR
    self.dt_hz = 1 / self.dt

  def set_oversampling(self, oversampling: float) -> None:
    self.oversampling = oversampling
    self.recalc()

  def set_free_parameters(self, param_a: float, param_b: float) -> None:
    self.param_a = param_a
    self.param_b = param_b
    self.recalc()

  def set_signal_frequency(self, signal_frequency: float) -> None:
    self.signal_frequency = signal_frequency

  def set_scheme(self, lambda_courant: float, param_a: float, param_b: float) -> None:
    self.lambda_courant = lambda_courant
    self.param_a = param_a
    self.param_b = param_b
    self.recalc()

  def set_max_frequency(self, max_frequency: float) -> None:
    self.max_frequency = max_frequency
    self.recalc()
