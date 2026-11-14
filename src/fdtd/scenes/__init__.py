"""Predefined scenes that can be voxelised into a simulation grid."""
from fdtd.scenes.base import Scene
from fdtd.scenes.bedroom import BedroomScene
from fdtd.scenes.bell_box import BellBoxScene
from fdtd.scenes.concert_hall import ConcertHallScene
from fdtd.scenes.cuboid_reference import CuboidReferenceScene
from fdtd.scenes.l_shaped_room import LShapedRoomScene
from fdtd.scenes.office import OfficeScene
from fdtd.scenes.real_life_room import RealLifeRoomScene
from fdtd.scenes.shoebox_reference import ShoeboxReferenceScene
from fdtd.scenes.studio_room import StudioRoomScene
from fdtd.scenes.volumetric import VolumetricScene

__all__ = [
    "SCENES",
    "create_scene",
    "Scene",
    "BedroomScene",
    "BellBoxScene",
    "ConcertHallScene",
    "CuboidReferenceScene",
    "LShapedRoomScene",
    "OfficeScene",
    "RealLifeRoomScene",
    "ShoeboxReferenceScene",
    "StudioRoomScene",
    "VolumetricScene",
]

# Scenes selectable by name (e.g. from the command line). Each entry builds a
# scene from simulation parameters.
SCENES = {
    "shoebox": ShoeboxReferenceScene,
    "cuboid": CuboidReferenceScene,
    "lshape": LShapedRoomScene,
    "bedroom": BedroomScene,
    "bellbox": BellBoxScene,
    "concert": ConcertHallScene,
    "office": OfficeScene,
    "studio": StudioRoomScene,
    "real-scene": RealLifeRoomScene,
    "real-reference": lambda parameters: RealLifeRoomScene(parameters, True),
}


def create_scene(name: str, parameters) -> Scene:
  """Instantiate a registered scene by name."""
  try:
    factory = SCENES[name]
  except KeyError:
    raise ValueError(
        f"Unknown scene '{name}', choose from: {', '.join(SCENES)}") from None
  return factory(parameters)
