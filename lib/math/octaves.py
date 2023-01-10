"""Math utilites for octave bands"""

import math
import numpy as np

CENTER_FREQUENCY = 1000

# http://blog.prosig.com/2006/02/17/standard-octave-bands/
def get_octaval_center_frequencies(lower=20, upper=200, fraction=24, exclusive=False) -> np.ndarray:
  factor = 2 ** (1 / fraction)
  lower_it = fraction * -math.log2(lower / CENTER_FREQUENCY)
  upper_it = fraction * -math.log2(upper / CENTER_FREQUENCY)
  start_band = math.floor(lower_it) if exclusive else math.ceil(lower_it)
  end_band = math.ceil(upper_it) if exclusive else math.floor(upper_it)
  current_frequency = CENTER_FREQUENCY * 2 ** (-start_band / fraction)
  
  values: list[float] = []
  
  for _ in range(start_band - end_band + 1):
    values.append(current_frequency)
    current_frequency *= factor
  
  return np.array(values)
  