import math
import numpy as np
import matplotlib.pyplot as plt

from lib.impulse_generators import GaussianModulatedImpulseGenerator, GaussianMonopulseGenerator

frequencies = np.arange(20.0, 200.0, 1.0)
time = np.arange(0.0, 2.0, 1/10000)
generators = [GaussianMonopulseGenerator, GaussianModulatedImpulseGenerator]

axes_shape = (5, len(generators))  # (Rows, Colums)
fig = plt.gcf()


def array_map(x, f):
  return np.array(list(map(f, x)))


for index, generator in enumerate(generators):
  axis = plt.subplot2grid(axes_shape, (0, index), rowspan=2)
  axis.set_title(f"{generator.__name__}")
  axis_zero = plt.subplot2grid(axes_shape, (2, index))
  axis_zero.set_title(f"Zero value")
  axis_sum = plt.subplot2grid(axes_shape, (3, index))
  axis_sum.set_title(f"Sum value")
  axis_rms = plt.subplot2grid(axes_shape, (4, index))
  axis_rms.set_title(f"RMS value")
  zero_set = []
  sum_set = []
  rms_set = []
  for f in frequencies:
    gen = generator(f)
    value_set = []
    sum = 0.0
    rms_sum = 0.0
    for t in time:
      v = gen.generate(t, 0)
      value_set.append(v)
      sum += v
      rms_sum += v * v
    rms = math.sqrt((1/time.size)*rms_sum)
    zero_set.append(gen.generate(0.0, 0))
    sum_set.append(sum)
    rms_set.append(rms)
    axis.plot(time, value_set)
  axis_zero.plot(frequencies, zero_set)
  axis_sum.plot(frequencies, sum_set)
  axis_rms.plot(frequencies, rms_set)

fig.tight_layout()
plt.show()
