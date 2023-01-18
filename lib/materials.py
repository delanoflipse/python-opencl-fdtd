import numpy as np
import scipy as sp
import scipy.interpolate


def log_interp1d(xx, yy, kind='linear'):
  # https://stackoverflow.com/questions/29346292/logarithmic-interpolation-in-python
  logx = np.log10(xx)
  logy = np.log10(yy)
  boundary_vals = (logy[0], logy[-1])
  lin_interp = sp.interpolate.interp1d(
      logx, logy, kind=kind, bounds_error=False, fill_value=boundary_vals)

  def log_interp(zz): return np.power(10.0, lin_interp(np.log10(zz)),)
  return log_interp


class SimulatedMaterial:
  # https://calculla.com/sound_absorption_coefficients
  # https://www.acoustic.ua/st/web_absorption_data_eng.pdf
  # https://cds.cern.ch/record/1251519/files/978-3-540-48830-9_BookBackMatter.pdf
  # 1 - alpha = beta
  def __init__(self, material_name: str):
    self.interp_function = None
    self.get_interp_from_material(material_name)

  def get_interp_from_material(self, name: str) -> None:
    if name == "plaster":
      self.interp_function = log_interp1d([125, 250], [0.29, 0.1])
      return
    if name == "painted_concrete":
      self.interp_function = log_interp1d([125, 250], [0.01, 0.01])
      return
    if name == "laminate":
      self.interp_function = log_interp1d([125, 250], [0.04, 0.04])
      return
    if name == "wood":
      self.interp_function = log_interp1d([125, 250], [0.1, 0.07])
      return
    if name == "carpet":
      self.interp_function = log_interp1d([125, 250], [0.1, 0.15])
    if name == "cellulose":
      self.interp_function = log_interp1d([125, 250], [0.05, 0.16])
      return
    if name == "whiteboard":
      # TODO
      self.interp_function = log_interp1d([125, 250], [0.1, 0.16])
      return
    if name == "metal":
      # TODO
      self.interp_function = log_interp1d([125, 250], [0.35, 0.39])
      return
    if name == "hard_wood":
      self.interp_function = log_interp1d([125, 250], [0.1, 0.07])
      return
    if name == "glass":
      self.interp_function = log_interp1d([125, 250], [0.15, 0.05])
      return

  def get_beta(self, frequency: float = 125) -> float:
    if self.interp_function is None:
      return 0.5
    return self.interp_function(frequency)
