import math
import numpy as np
import matplotlib.pyplot as plt

from lib.impulse_generators import GaussianModulatedImpulseGenerator, GaussianMonopulseGenerator, WindowModulatedSinoidImpulse, SimpleSinoidGenerator, ImpulseGenerator
from lib.math.decibel_weightings import get_a_weighting
from lib.math.octaves import get_octaval_center_frequencies

frequencies = get_octaval_center_frequencies(20, 200)
full_frequencies = get_octaval_center_frequencies(20, 20e3, fraction=3)
dt = 1/10000
time = np.arange(0.0, 2.0, dt)
generators = [GaussianMonopulseGenerator,
              GaussianModulatedImpulseGenerator, WindowModulatedSinoidImpulse, SimpleSinoidGenerator]

axes_shape = (8, len(generators))  # (Rows, Colums)
fig = plt.gcf()

P_0 = 20e-6
P_0_2 = P_0 * P_0
P_0_2INV = 1 / P_0_2


def array_map(x, f):
  return np.array(list(map(f, x)))


axis_a_weight = plt.subplot2grid(axes_shape, (7, 0), colspan=len(generators))
axis_a_weight.set_title("A weighting (dB)")
axis_a_weight.set_xscale('log')
axis_a_weight.grid(True)
weight_set = array_map(full_frequencies, get_a_weighting)
axis_a_weight.plot(full_frequencies, weight_set)


for index, generator in enumerate(generators):
  axis = plt.subplot2grid(axes_shape, (0, index), rowspan=2)
  axis.set_title(f"{generator.__name__}")
  axis_zero = plt.subplot2grid(axes_shape, (2, index))
  axis_zero.set_title("Zero value")
  axis_sum = plt.subplot2grid(axes_shape, (3, index))
  axis_sum.set_title("Sum value")
  axis_rms = plt.subplot2grid(axes_shape, (4, index))
  axis_rms.set_title("RMS value")
  axis_dbs = plt.subplot2grid(axes_shape, (5, index), rowspan=2)
  axis_dbs.set_title("SPL (dB) value")
  zero_set = []
  sum_set = []
  dbs_set = []

  for f in frequencies:
    gen: ImpulseGenerator = generator(f)
    value_set = []
    db_value_set = []
    sum = 0.0
    rms_sum = 0.0
    rms = 0.0
    rms_set = []
    i = 0
    for t in time:
      i += 1
      v = gen.generate(t, 0)
      value_set.append(v)
      sum += v * dt
      rms_sum += dt * (v * v) * P_0_2INV
      rms = (1 / t) * rms_sum if t > 0 else 0.0
      spl = 10 * math.log10(rms) if rms != 0.0 else 0.0
      db_value_set.append(spl)
      rms_set.append(rms)
    zero_set.append(gen.generate(0.0, 0))
    sum_set.append(sum)
    axis.plot(time, value_set)
    axis_rms.plot(time, rms_set)
    axis_dbs.plot(time, db_value_set)
  axis_zero.plot(frequencies, zero_set)
  axis_sum.plot(frequencies, sum_set)

fig.tight_layout()
plt.show()
