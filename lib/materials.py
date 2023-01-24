import numpy as np
import scipy as sp

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
  def __init__(self, interp_function):
    self.interp_function = interp_function

  def get_beta(self, frequency: float = 125) -> float:
    if self.interp_function is None:
      return 0.5
    return self.interp_function(frequency)


wood_material = SimulatedMaterial(log_interp1d([125, 250], [0.1, 0.07]))
painted_concrete_material = SimulatedMaterial(
    log_interp1d([125, 250], [0.01, 0.01]))
laminate_material = SimulatedMaterial(log_interp1d([125, 250], [0.04, 0.04]))
plaster_material = SimulatedMaterial(log_interp1d([125, 250], [0.29, 0.1]))
glass_material = SimulatedMaterial(log_interp1d([125, 250], [0.15, 0.05]))
double_glass_material = SimulatedMaterial(log_interp1d([125, 250], [0.15, 0.05]))
carpet_material = SimulatedMaterial(log_interp1d([125, 250], [0.1, 0.15]))
cellulose_material = SimulatedMaterial(log_interp1d([125, 250], [0.05, 0.16]))

# TODO: find reference values
metal_material = SimulatedMaterial(log_interp1d([125, 250], [0.35, 0.39]))
hard_wood_material = SimulatedMaterial(log_interp1d([125, 250], [0.1, 0.07]))
whiteboard_material = SimulatedMaterial(log_interp1d([125, 250], [0.1, 0.16]))
suspended_ceiling_material = SimulatedMaterial(log_interp1d([125, 250], [0.15, 0.11]))
hard_wall_material = SimulatedMaterial(log_interp1d([125, 250], [0.04, 0.05]))
cushion_material = SimulatedMaterial(log_interp1d([125, 250], [0.32, 0.4]))
