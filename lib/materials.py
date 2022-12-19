class SimulatedMaterial:
  # https://calculla.com/sound_absorption_coefficients
  # 1 - alpha = beta
  # TODO: frequency dependant materials
  def __init__(self, material_name: str):
    self.beta = self.get_beta_from_material(material_name)

  def get_beta_from_material(self, name: str) -> float:
    if name == "wood":
      return 0.5
    return 0.5
