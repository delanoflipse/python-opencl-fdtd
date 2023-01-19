"""Module Impulse contains all possible impulse generators"""
import math


class ImpulseGenerator:
  """Interface for generators"""

  def generate(self, time: float, iteration: int) -> float:
    """Generate a value given the current state of the simulation"""
    return 0.0


class HannWindow(ImpulseGenerator):
  def __init__(self, width: float = 0.5, end_signal=0.0) -> None:
    self.width = width
    self.end_signal = end_signal
    self.factor = math.pi / self.width

  def generate(self, time: float, iteration: int) -> float:
    t_now = time
    if t_now >= self.width:
      return self.end_signal
    sin_factor = math.sin(t_now * self.factor)
    window_value = sin_factor * sin_factor
    return window_value


class GaussianWindow(ImpulseGenerator):
  def __init__(self, bandwidth: float = 16, bandwidth_reference=-6, cutoff_limit_db=-120) -> None:
    self.bandwidth = bandwidth
    self.ref = pow(10.0, bandwidth_reference / 20.0)
    self.a = -(math.pi * bandwidth) ** 2 / (4.0 * math.log(self.ref))
    self.tref = pow(10.0, cutoff_limit_db / 20.0)
    self.cuttoff = math.sqrt(-math.log(self.tref) / self.a)
    self.t_0 = self.cuttoff * 1.0

  def generate(self, time: float, iteration: int) -> float:
    t_now = time - self.t_0
    exp_envolope = math.exp(-self.a * t_now * t_now)
    return exp_envolope


class WindowModulatedSinoidImpulse(ImpulseGenerator):
  """Generates a windowed sinoid impulse"""

  def __init__(self, frequency: float, window_generator: ImpulseGenerator = None, use_cosine: bool = True) -> None:
    self.frequency = frequency
    if window_generator is None:
      self.window_generator = HannWindow(0.5)
    else:
      self.window_generator = window_generator
    self.use_cosine = use_cosine

  def generate(self, time: float, iteration: int) -> float:
    envelope_factor = self.window_generator.generate(time, iteration)
    if math.isnan(envelope_factor):
      return math.nan

    sinoid_param = time * self.frequency * 2 * math.pi
    sinoid_value = math.cos(
        sinoid_param) if self.use_cosine else math.sin(sinoid_param)
    return envelope_factor * sinoid_value


class SimpleSinoidGenerator(ImpulseGenerator):
  """Generates a windowed sinoid impulse"""

  def __init__(self, frequency: float, use_cosine: bool = False) -> None:
    self.frequency = frequency
    self.use_cosine = use_cosine

  def generate(self, time: float, iteration: int) -> float:
    sinoid_param = time * self.frequency * 2 * math.pi
    sinoid_value = math.cos(
        sinoid_param) if self.use_cosine else math.sin(sinoid_param)
    return sinoid_value


class GaussianModulatedImpulseGenerator(ImpulseGenerator):
  """Generates a Gaussian modulated cosine impulse"""
  # https://github.com/scipy/scipy/blob/v1.9.3/scipy/signal/_waveforms.py#L161-L258

  def __init__(self, frequency: float, bandwidth: float = 0.8, bwr=-6, tpr=-120) -> None:
    super().__init__()
    self.frequency = frequency
    self.ref = pow(10.0, bwr / 20.0)
    self.bandwidth = bandwidth
    # self.a = -(math.pi * frequency * bandwidth) ** 2 / \
    #     (4.0 * math.log(self.ref))
    self.a = -(math.pi * bandwidth * 20) ** 2 / \
        (4.0 * math.log(self.ref))

    self.tref = pow(10.0, tpr / 20.0)
    self.cuttoff = math.sqrt(-math.log(self.tref) / self.a)
    self.t_0 = self.cuttoff * 1.0

  def generate(self, time: float, iteration: int) -> float:
    t_now = time - self.t_0
    exp_envolope = math.exp(-self.a * t_now * t_now)
    cos_factor = math.sin(t_now * self.frequency * 2 * math.pi)
    return exp_envolope * cos_factor


class GaussianMonopulseGenerator(ImpulseGenerator):
  """Generates a Gaussian modulated cosine impulse"""

  def __init__(self, frequency: float) -> None:
    super().__init__()
    self.frequency = frequency
    self.sigma = 1 / (2 * math.pi * frequency)
    self.exp1_2 = math.exp(1/2)
    self.t_0 = self.sigma * math.pi * 1.8

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
