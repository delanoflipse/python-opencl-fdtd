"""Module Impulse contains all possible impulse generators"""
import math


class ImpulseGenerator:
  """Interface for generators"""

  def generate(self, time: float, iteration: int) -> float:
    """Generate a value given the current state of the simulation"""
    return 0.0


class GaussianModulatedImpulseGenerator(ImpulseGenerator):
  """Generates a Gaussian modulated cosine impulse"""
  # https://github.com/scipy/scipy/blob/v1.9.3/scipy/signal/_waveforms.py#L161-L258

  def __init__(self, frequency: float, bandwidth: float = 0.5, bwr=-6, tpr=-60) -> None:
    super().__init__()
    self.frequency = frequency
    self.ref = pow(10.0, bwr / 20.0)
    self.bandwidth = bandwidth
    self.a = -(math.pi * frequency * bandwidth) ** 2 / \
        (4.0 * math.log(self.ref))
    self.tref = pow(10.0, tpr / 20.0)
    self.cuttoff = math.sqrt(-math.log(self.tref) / self.a)
    self.t_0 = self.cuttoff * 2

  def generate(self, time: float, iteration: int) -> float:
    t_now = time - self.t_0
    exp_envolope = math.exp(-self.a * t_now * t_now)
    cos_factor = math.sin(2 * math.pi * t_now * self.frequency)
    return exp_envolope * cos_factor


class GaussianMonopulseGenerator(ImpulseGenerator):
  """Generates a Gaussian modulated cosine impulse"""

  def __init__(self, frequency: float) -> None:
    super().__init__()
    self.frequency = frequency
    self.sigma = 1 / (2 * math.pi * frequency)
    self.exp1_2 = math.exp(1/2)
    self.t_0 = self.sigma * math.pi * 1.5

  def generate(self, time: float, iteration: int) -> float:
    # https://nl.mathworks.com/help/signal/ref/gmonopuls.html
    t_now = time - self.t_0
    t_over_sg = (t_now / self.sigma)
    signal = self.exp1_2 * t_over_sg * math.exp(-0.5*t_over_sg*t_over_sg)
    return signal


class DiracImpulseGenerator(ImpulseGenerator):
  """Generates a dirac impulse"""

  def generate(self, time: float, iteration: int) -> float:
    return 1.0 if iteration == 0 else 0.0
