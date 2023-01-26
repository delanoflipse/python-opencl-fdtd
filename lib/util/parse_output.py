import csv
import math
from lib.scene.scene import Scene
from typing import List
import numpy as np


def as_float(value: str) -> float:
  try:
    return float(value)
  except:
    return math.nan


class OutputParser:

  def __init__(self, file_path: str) -> None:
    self.file_path = file_path
    self.reset_values()

  def reset_values(self) -> None:
    self.count = 0
    self.max_spl_overall = -float('inf')
    self.min_spl_overall = float('inf')
    self.max_value = -float('inf')
    self.min_value = float('inf')
    self.best_set = []
    self.worst_set = []
    self.location_map = {}
    self.band_map = {}
    self.indexes = []
    self.value_index = []
    self.frequencies: List[float] = []
    self.max_per_frequency = []
    self.sum_per_frequency = []
    self.min_per_frequency = []
    self.values_per_frequency = []
    self.optimal = []

  def parse_values(self) -> None:
    csv_file = open(self.file_path, 'r', encoding="utf-8", newline='')
    reader = csv.reader(csv_file, delimiter=',', quotechar='|')
    self.reset_values()

    for i, row in enumerate(reader):
      float_row = list(map(as_float, row))
      _, index, w_i, w_m, h_i, h_m, d_i, d_m, dev, spl, _, *bands = float_row
      if i == 0:
        self.frequencies = bands
        for i, _ in enumerate(bands):
          # min, max, sum
          self.max_per_frequency.append(-float('inf'))
          self.min_per_frequency.append(float('inf'))
          self.sum_per_frequency.append(0)
          self.values_per_frequency.append([])
        continue
      derrivative2 = np.diff(bands, n=1)
      value = np.sum(np.power(derrivative2, 2))
      if value == 0.0:
        continue
      self.count += 1
      w = int(w_i)
      d = int(d_i)
      h = int(h_i)
      self.location_map[i] = (w, h, d)
      self.max_value = max(value, self.max_value)
      self.min_value = min(value, self.min_value)
      self.indexes.append(i)
      self.value_index.append(value)
      self.band_map[i] = bands

      if value == self.max_value:
        self.worst_set = bands

      if value == self.min_value:
        self.best_set = bands

      # print(bands)
      # axis_spl_all.plot(frequencies, bands)
      for freq_index, frequency in enumerate(self.frequencies):
        value = bands[freq_index]
        self.max_per_frequency[freq_index] = max(
            self.max_per_frequency[freq_index], value)
        self.min_per_frequency[freq_index] = min(
            self.min_per_frequency[freq_index], value)
        self.sum_per_frequency[freq_index] += value
        self.values_per_frequency[freq_index].append(value)
        self.max_spl_overall = max(self.max_spl_overall, value)
        self.min_spl_overall = min(self.min_spl_overall, value)

    self.optimal = sorted(zip(self.value_index, self.indexes))
    csv_file.close()
