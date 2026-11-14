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
