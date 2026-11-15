"""Scenes: rooms with region markers, defined in JSON (see fdtd.scenes.base)."""
from pathlib import Path
from typing import Dict, Union

from fdtd.parameters import SimulationParameters
from fdtd.scenes.base import Scene

__all__ = [
    "SCENES",
    "SCENE_DIR",
    "Scene",
    "create_scene",
    "volumetric_scene",
]

SCENE_DIR = Path(__file__).parent / "data"

# order in which the bundled scenes are listed
_PREFERRED_ORDER = ["shoebox", "shoebox-reverberant", "cuboid", "lshape", "bedroom", "bellbox",
                    "concert", "office", "studio", "real-scene", "real-reference"]


def _find_scenes() -> Dict[str, Path]:
  paths = {path.stem: path for path in SCENE_DIR.glob("*.json")}
  ordered = [name for name in _PREFERRED_ORDER if name in paths]
  ordered += sorted(set(paths) - set(ordered))
  return {name: paths[name] for name in ordered}


# Scenes selectable by name (e.g. from the command line): name -> JSON file
SCENES = _find_scenes()


def create_scene(name: Union[str, Path], parameters: SimulationParameters) -> Scene:
  """Load a bundled scene by name, or a scene from a path to a .json file."""
  if name in SCENES:
    return Scene.from_file(SCENES[name], parameters)
  path = Path(name)
  if path.suffix == ".json" and path.is_file():
    return Scene.from_file(path, parameters)
  raise ValueError(
      f"Unknown scene '{name}', choose from: {', '.join(SCENES)} or a .json file")


def volumetric_scene(parameters: SimulationParameters, volume: float) -> Scene:
  """A volume x 1 x 1 m tube, e.g. for benchmarking"""
  return Scene(parameters, {
      "name": f"{volume:g} m^3 tube",
      "size": [volume, 1.0, 1.0],
      "boundaries": {"all": 0.1, "height_min": 0.05},
      "regions": [
          {"name": "source", "type": "source", "at": [0, 0, 0]},
          {"name": "centre", "type": "listener", "at": ["width / 2", "height / 2", "depth / 2"]},
      ],
  }, key="volumetric")
