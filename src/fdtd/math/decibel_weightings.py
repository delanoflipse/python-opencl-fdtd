
import math


def get_reference_a_weighting(frequency: float) -> float:
  # https://en.wikipedia.org/wiki/A-weighting
  f2 = frequency * frequency
  f4 = f2 * f2
  r_a1 = (12194 ** 2) * f4
  sqrt_parts = (f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)
  r_a2 = (f2 + 20.6 ** 2) * (f2 + 12194 ** 2) * math.sqrt(sqrt_parts)
  return r_a1 / r_a2


RA_1000 = get_reference_a_weighting(1000.0)
RA_1000_LOG = 20 * math.log10(RA_1000)


def get_a_weighting(frequency: float) -> float:
  return 20 * math.log10(get_reference_a_weighting(frequency)) - RA_1000_LOG
  # return 20 * math.log10(get_reference_a_weighting(frequency)) + 2.0
