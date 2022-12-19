"""Module Impulse contains all possible impulse generators"""
import math


class ImpulseGenerator:
  """Interface for generators"""
  def generate(self, time: float, iteration: int) -> float:
    """Generate a value given the current state of the simulation"""
    return 0.0


class GaussianModulatedImpulseGenerator(ImpulseGenerator):
  """Generates a Gaussian modulated cosine impulse"""
  def __init__(self, frequency: float, sigma: float = 0.002) -> None:
    super().__init__()
    self.frequency = frequency
    self.sigma = sigma
    self.variance = sigma * sigma
    self.t_0 = 8 * sigma

  def generate(self, time: float, iteration: int) -> float:
    t_now = time - self.t_0
    cos_factor = math.cos(2 * math.pi * t_now * self.frequency)
    exp_factor = math.exp(-(t_now * t_now) / (2 * self.variance))
    return cos_factor * (exp_factor)


class DiracImpulseGenerator(ImpulseGenerator):
  """Generates a dirac impulse"""
  def generate(self, time: float, iteration: int) -> float:
    return 1.0 if iteration == 0 else 0.0
