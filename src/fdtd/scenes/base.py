"""A scene: a box shaped room with region markers, loaded from JSON.

Scene file format (all lengths in metres)::

  {
    "name": "Bedroom",
    "description": "optional text",
    "size": [width, height, depth],
    "boundaries": {                       # absorption of the six room faces
      "all": 0.1,                         # optional default for every face
      "width_min": "plaster",             # material name, beta, or a list of
      "height_min": ["glass", 0.05],      #   those that is averaged
      ...                                 # width/height/depth _min/_max
    },
    "regions": [
      {"name": "closet", "type": "wall", "material": "wood",
       "w": ["width - 0.6", null], "h": [null, 2.0], "d": [0.9, 1.9]},
      {"name": "speaker", "type": "source", "at": [0.3, 0.15, 0.3]},
      {"name": "seats", "type": "listener", "h": [1.0, 1.4]}
    ]
  }

Region types are ``wall`` (solid, with a ``material`` or ``beta``),
``source`` (candidate source positions) and ``listener`` (cells that are
analysed). A region is either a box, given per axis as ``[min, max]`` where a
missing axis or ``null`` bound extends to the room edge, or a single cell
``"at": [w, h, d]``. Coordinates are numbers or expressions over ``width``,
``height``, ``depth`` and ``dx`` (see :mod:`fdtd.scenes.expressions`).
Regions are applied in order, so a later wall overrides an earlier beta.
Materials are resolved at the signal frequency, every (re)build.
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from fdtd.grid import BASE_BETA, LISTENER_FLAG, SOURCE_REGION_FLAG, WALL_FLAG, SimulationGrid
from fdtd.materials import MATERIALS
from fdtd.parameters import SimulationParameters
from fdtd.physical_constants import C_AIR
from fdtd.scenes.expressions import evaluate

REGION_FLAGS = {
    "wall": WALL_FLAG,
    "source": SOURCE_REGION_FLAG,
    "listener": LISTENER_FLAG,
}
AXES = ("w", "h", "d")
BOUNDARY_FACES = ("width_min", "width_max", "height_min",
                  "height_max", "depth_min", "depth_max")
# beta of a wall region without material or beta (as in SimulationGrid.fill_region)
DEFAULT_WALL_BETA = 0.5

_SCENE_KEYS = {"name", "description", "size", "boundaries", "regions"}
_REGION_KEYS = {"name", "type", "material", "beta", "at", *AXES}

MaterialSpec = Union[str, float, List[Union[str, float]]]


def resolve_material(spec: MaterialSpec, frequency: float) -> float:
  """Beta of a material name, a beta value or the average of a list of those"""
  if isinstance(spec, list):
    if not spec:
      raise ValueError("an averaged material list must not be empty")
    return sum(resolve_material(item, frequency) for item in spec) / len(spec)
  if isinstance(spec, str):
    try:
      material = MATERIALS[spec]
    except KeyError:
      raise ValueError(
          f"unknown material '{spec}', choose from: {', '.join(MATERIALS)}") from None
    return float(material.get_beta(frequency))
  if isinstance(spec, (int, float)) and not isinstance(spec, bool):
    return float(spec)
  raise ValueError(f"invalid material {spec!r}")


class Scene:
  def __init__(self, parameters: SimulationParameters, definition: Dict[str, Any], key: Optional[str] = None) -> None:
    self.parameters = parameters
    self.definition = definition
    self.key = key
    _validate(definition)
    self.name: str = definition.get("name", key or "scene")
    self.description: str = definition.get("description", "")
    self.width, self.height, self.depth = (float(v) for v in definition["size"])
    self.shape = (self.width, self.height, self.depth)
    self.grid: SimulationGrid = None

  @classmethod
  def from_file(cls, path: Union[str, Path], parameters: SimulationParameters) -> "Scene":
    path = Path(path)
    with open(path, encoding="utf-8") as file:
      definition = json.load(file)
    try:
      return cls(parameters, definition, key=path.stem)
    except ValueError as error:
      raise ValueError(f"{path.name}: {error}") from None

  def build(self) -> SimulationGrid:
    self.grid = SimulationGrid(self.shape, self.parameters)
    self.mark_regions()
    self.grid.build()
    return self.grid

  def rebuild(self) -> None:
    if self.grid is None:
      raise Exception("Grid is set. Please call build before calling rebuild!")
    self.mark_regions()
    self.grid.rebuild()
    return None

  def get_room_modes(self) -> List[Tuple[float, int]]:
    """Analytic modes (frequency, number of axes) of the bounding box, up to order 3"""
    frequencies = []
    for i in range(4):
      i_active = 1 if i > 0 else 0
      for j in range(4):
        j_active = 1 if j > 0 else 0
        for k in range(4):
          k_active = 1 if k > 0 else 0
          nw = i / self.width
          nh = j / self.height
          nd = k / self.depth
          axis_type = i_active + j_active + k_active
          if axis_type == 0:
            continue
          s_part = nw * nw + nh * nh + nd * nd
          f = (C_AIR / 2) * math.sqrt(s_part)
          if f == 0.0:
            continue
          frequencies.append((f, axis_type))
    return frequencies

  # ---- resolving the definition ----

  def _names(self) -> Dict[str, float]:
    return {"width": self.width, "height": self.height,
            "depth": self.depth, "dx": self.parameters.dx}

  def boundary_betas(self, frequency: Optional[float] = None) -> Dict[str, float]:
    if frequency is None:
      frequency = self.parameters.signal_frequency
    boundaries = self.definition.get("boundaries", {})
    default = resolve_material(boundaries["all"], frequency) \
        if "all" in boundaries else BASE_BETA
    return {face: resolve_material(boundaries[face], frequency) if face in boundaries else default
            for face in BOUNDARY_FACES}

  def region_bounds(self, region: Dict[str, Any]) -> Dict[str, float]:
    """fill_region keyword arguments (w_min, ..., d_max) of a region"""
    names = self._names()
    bounds: Dict[str, float] = {}
    if "at" in region:
      for axis, value in zip(AXES, region["at"]):
        bounds[f"{axis}_min"] = bounds[f"{axis}_max"] = evaluate(value, names)
      return bounds
    for axis in AXES:
      low, high = region.get(axis, (None, None))
      if low is not None:
        bounds[f"{axis}_min"] = evaluate(low, names)
      if high is not None:
        bounds[f"{axis}_max"] = evaluate(high, names)
    return bounds

  def region_beta(self, region: Dict[str, Any], frequency: Optional[float] = None) -> float:
    if frequency is None:
      frequency = self.parameters.signal_frequency
    if "beta" in region:
      return resolve_material(region["beta"], frequency)
    if "material" in region:
      return resolve_material(region["material"], frequency)
    return DEFAULT_WALL_BETA

  def mark_regions(self) -> None:
    if self.grid is None:
      return
    for face, beta in self.boundary_betas().items():
      setattr(self.grid.edge_betas, face, beta)
    for region in self.definition.get("regions", []):
      flag = REGION_FLAGS[region["type"]]
      bounds = self.region_bounds(region)
      if flag == WALL_FLAG:
        self.grid.fill_region(**bounds, geometry_flag=flag,
                              beta=self.region_beta(region))
      else:
        self.grid.fill_region(**bounds, geometry_flag=flag)

  def default_source(self) -> Tuple[float, float, float]:
    """Centre (m) of the first source region, or of the room without one"""
    for region in self.definition.get("regions", []):
      if region["type"] == "source":
        bounds = self.region_bounds(region)
        size = dict(zip(AXES, self.shape))
        return tuple(
            (min(max(bounds.get(f"{a}_min", 0.0), 0.0), size[a]) +
             min(max(bounds.get(f"{a}_max", size[a]), 0.0), size[a])) / 2
            for a in AXES)
    return (self.width / 2, self.height / 2, self.depth / 2)

  def describe(self) -> Dict[str, Any]:
    """The scene as voxelised by the grid: region boxes snapped to cell edges (m)"""
    if self.grid is None:
      raise Exception("Please call build before describing the scene")
    grid = self.grid
    dx = self.parameters.dx
    regions = []
    for index, region in enumerate(self.definition.get("regions", [])):
      cells = grid.region_slices(**self.region_bounds(region))
      entry = {
          "name": region.get("name", f"{region['type']} {index}"),
          "type": region["type"],
          "min": [s.start * dx for s in cells],
          "max": [s.stop * dx for s in cells],
      }
      if region["type"] == "wall":
        entry["beta"] = self.region_beta(region)
        if "material" in region:
          entry["material"] = region["material"]
      regions.append(entry)
    return {
        "key": self.key,
        "name": self.name,
        "description": self.description,
        "size": list(self.shape),
        "extent": [parts * dx for parts in grid.grid_shape],
        "boundaries": self.boundary_betas(),
        "regions": regions,
    }


def _validate(definition: Dict[str, Any]) -> None:
  if not isinstance(definition, dict):
    raise ValueError("a scene must be a JSON object")
  unknown = set(definition) - _SCENE_KEYS
  if unknown:
    raise ValueError(f"unknown scene keys: {', '.join(sorted(unknown))}")
  size = definition.get("size")
  if not (isinstance(size, list) and len(size) == 3 and
          all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in size)):
    raise ValueError("'size' must be [width, height, depth] with positive numbers")

  boundaries = definition.get("boundaries", {})
  if not isinstance(boundaries, dict):
    raise ValueError("'boundaries' must be an object")
  unknown = set(boundaries) - {"all", *BOUNDARY_FACES}
  if unknown:
    raise ValueError(f"unknown boundary faces: {', '.join(sorted(unknown))}")
  for face, spec in boundaries.items():
    try:
      resolve_material(spec, 125.0)
    except ValueError as error:
      raise ValueError(f"boundary '{face}': {error}") from None

  regions = definition.get("regions", [])
  if not isinstance(regions, list):
    raise ValueError("'regions' must be a list")
  names = {"width": 1.0, "height": 1.0, "depth": 1.0, "dx": 1.0}
  for index, region in enumerate(regions):
    label = f"region {index}" if not isinstance(region, dict) \
        else f"region '{region.get('name', index)}'"
    if not isinstance(region, dict):
      raise ValueError(f"{label} must be an object")
    unknown = set(region) - _REGION_KEYS
    if unknown:
      raise ValueError(f"{label}: unknown keys: {', '.join(sorted(unknown))}")
    if region.get("type") not in REGION_FLAGS:
      raise ValueError(f"{label}: 'type' must be one of {', '.join(REGION_FLAGS)}")
    if region["type"] != "wall" and ("material" in region or "beta" in region):
      raise ValueError(f"{label}: only walls have a material or beta")
    if "at" in region:
      if any(axis in region for axis in AXES):
        raise ValueError(f"{label}: use either 'at' or w/h/d bounds")
      if not (isinstance(region["at"], list) and len(region["at"]) == 3):
        raise ValueError(f"{label}: 'at' must be [w, h, d]")
      values = list(region["at"])
    else:
      values = []
      for axis in AXES:
        if axis not in region:
          continue
        bounds = region[axis]
        if not (isinstance(bounds, list) and len(bounds) == 2):
          raise ValueError(f"{label}: '{axis}' must be [min, max] (null for the room edge)")
        values += [v for v in bounds if v is not None]
    try:
      for value in values:
        evaluate(value, names)
      for key in ("material", "beta"):
        if key in region:
          resolve_material(region[key], 125.0)
    except ValueError as error:
      raise ValueError(f"{label}: {error}") from None
