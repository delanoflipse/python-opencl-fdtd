"""Module Parameters contains all possible impulse generators"""
import math

from lib.physical_constants import C_AIR

SQRT_3 = math.sqrt(3)


class SimulationParameters:
  def __init__(self):
    self.oversampling = 16
    self.set_min_frequency(20.0)
    self.set_max_frequency(200.0)
    self.signal_frequency = 200.0
    # TODO: getters/setters/use free parameters
    self.param_a = 0.0
    self.param_b = 0.0
    self.arg_d1 = self.lambda_2 * \
        (1.0 - 4.0 * self.param_a + 4.0 * self.param_b)
    self.arg_d2 = self.lambda_2 * (self.param_a - 2.0 * self.param_b)
    self.arg_d3 = self.lambda_2 * self.param_b
    self.arg_d4 = 2.0 * (1.0 - 3.0 * self.lambda_2 + 6.0 * self.lambda_2 *
                         self.param_a - 4.0 * self.param_b * self.lambda_2)

  def set_oversampling(self, oversampling: int) -> None:
    self.oversampling = oversampling
    self.set_max_frequency(self.max_frequency)

  def set_min_frequency(self, min_frequency: float) -> None:
    self.min_frequency = min_frequency

  def set_signal_frequency(self, signal_frequency: float) -> None:
    self.signal_frequency = signal_frequency

  def set_max_frequency(self, max_frequency: float) -> None:
    self.max_frequency = max_frequency
    self.sampling_frequency = self.max_frequency
    # TODO: adjust to be more coherent
    self.min_wavelength = C_AIR / self.sampling_frequency
    self.dx = self.min_wavelength / self.oversampling
    self.dt = self.dx / (SQRT_3 * C_AIR)
    self.dt_hz = 1 / self.dt
    self.lambda_courant = (C_AIR * self.dt) / self.dx
    self.lambda_2 = self.lambda_courant * self.lambda_courant
