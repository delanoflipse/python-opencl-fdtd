import json

import numpy as np
import pytest

from fdtd.grid import LISTENER_FLAG, SOURCE_REGION_FLAG, WALL_FLAG
from fdtd.materials import carpet_material
from fdtd.parameters import SimulationParameters
from fdtd.scenes import SCENES, Scene, create_scene, volumetric_scene
from fdtd.scenes.expressions import evaluate


def coarse_parameters() -> SimulationParameters:
  params = SimulationParameters()
  params.set_max_frequency(200)
  params.set_oversampling(4)
  return params


def minimal(**overrides):
  definition = {"size": [4.0, 3.0, 5.0], "regions": []}
  definition.update(overrides)
  return definition


# ---- expressions ----

@pytest.mark.parametrize("expression, expected", [
    (1.5, 1.5),
    ("width - 0.6", 3.4),
    ("depth / 2 + 0.23", 2.73),
    ("-height + 4", 1.0),
    ("max(dx * 2, 0.03)", 0.2),
    ("min(1, 2, 0.5)", 0.5),
])
def test_expressions(expression, expected):
  names = {"width": 4.0, "height": 3.0, "depth": 5.0, "dx": 0.1}
  assert evaluate(expression, names) == pytest.approx(expected)


@pytest.mark.parametrize("expression", [
    "__import__('os')", "width ** 2", "unknown + 1", "abs(-1)", "1 if width else 2", "", True, None, [1],
])
def test_expressions_reject_anything_else(expression):
  with pytest.raises(ValueError):
    evaluate(expression, {"width": 4.0})


# ---- bundled scene files ----

@pytest.mark.parametrize("name", sorted(SCENES))
def test_bundled_scene_files_are_valid(name):
  with open(SCENES[name], encoding="utf-8") as file:
    definition = json.load(file)
  Scene(coarse_parameters(), definition, key=name)


# ---- validation ----

@pytest.mark.parametrize("definition, message", [
    (minimal(size=[1, 2]), "size"),
    (minimal(size=[1, 0, 2]), "size"),
    (minimal(extra=1), "unknown scene keys"),
    (minimal(boundaries={"ceiling": 0.1}), "unknown boundary faces"),
    (minimal(boundaries={"all": "cheese"}), "unknown material"),
    (minimal(regions=[{"type": "door"}]), "'type' must be one of"),
    (minimal(regions=[{"type": "wall", "colour": "red"}]), "unknown keys"),
    (minimal(regions=[{"type": "wall", "material": "cheese"}]), "unknown material"),
    (minimal(regions=[{"type": "source", "beta": 0.1}]), "only walls"),
    (minimal(regions=[{"type": "source", "at": [1, 1]}]), "'at' must be"),
    (minimal(regions=[{"type": "source", "at": [1, 1, 1], "w": [0, 1]}]), "either 'at'"),
    (minimal(regions=[{"type": "wall", "w": [0]}]), "must be \\[min, max\\]"),
    (minimal(regions=[{"type": "wall", "w": [0, "width * height ** 2"]}]), "unsupported expression"),
])
def test_invalid_scenes_are_rejected(definition, message):
  with pytest.raises(ValueError, match=message):
    Scene(coarse_parameters(), definition)


def test_unknown_scene_file_names_the_file(tmp_path):
  path = tmp_path / "broken.json"
  path.write_text(json.dumps(minimal(regions=[{"name": "door", "type": "door"}])), encoding="utf-8")
  with pytest.raises(ValueError, match="broken.json: region 'door'"):
    create_scene(str(path), coarse_parameters())


# ---- behaviour ----

def test_regions_are_voxelised_in_order():
  params = coarse_parameters()  # dx ~ 0.43 m
  scene = Scene(params, minimal(
      boundaries={"all": 0.2, "height_min": "carpet"},
      regions=[
          {"name": "block", "type": "wall", "beta": 0.3, "w": [None, "width / 2"]},
          {"name": "overlay", "type": "wall", "beta": 0.7, "w": [None, 0.5]},
          {"name": "spk", "type": "source", "at": ["width - 0.5", 0.5, 0.5]},
          {"name": "ears", "type": "listener", "h": [1.0, 2.0]},
      ]))
  grid = scene.build()
  half = grid.scale(2.0)
  assert (grid.geometry[:half] & WALL_FLAG).all()
  assert not (grid.geometry[half:] & WALL_FLAG).any()
  assert grid.beta[0, 0, 0] == 0.7  # the later region wins
  assert grid.beta[half - 1, 0, 0] == 0.3
  assert grid.source_set == [grid.pos(3.5, 0.5, 0.5)]
  assert ((grid.geometry & LISTENER_FLAG) > 0).sum() > 0
  assert grid.edge_betas.width_min == 0.2
  assert grid.edge_betas.height_min == pytest.approx(carpet_material.get_beta(params.signal_frequency))


def test_material_lists_are_averaged():
  scene = Scene(coarse_parameters(), minimal(boundaries={"all": ["painted_concrete", 0.21]}))
  assert scene.boundary_betas(125.0)["depth_max"] == pytest.approx((0.01 + 0.21) / 2)


def test_point_regions_are_clamped_into_the_grid():
  scene = Scene(coarse_parameters(), minimal(regions=[
      {"type": "source", "at": ["width + 3", -1, "depth"]}]))
  grid = scene.build()
  assert grid.source_set == [(grid.width_parts - 1, 0, grid.depth_parts - 1)]


def test_describe_snaps_regions_to_cells():
  params = coarse_parameters()
  scene = Scene(params, minimal(regions=[
      {"name": "box", "type": "wall", "material": "wood", "w": [1.0, 2.0], "d": [None, 1.0]},
      {"type": "listener", "at": [2, 1, 2]},
  ]))
  scene.build()
  info = scene.describe()
  dx = params.dx
  box, point = info["regions"]
  assert box["name"] == "box" and box["material"] == "wood"
  assert box["min"] == [round(1.0 / dx) * dx, 0.0, 0.0]
  assert box["max"][1] == pytest.approx(scene.grid.height_parts * dx)
  assert point["name"] == "listener 1"
  assert np.allclose(np.subtract(point["max"], point["min"]), dx)
  assert info["extent"] == [parts * dx for parts in scene.grid.grid_shape]


def test_default_source_is_the_centre_of_the_first_source_region():
  scene = Scene(coarse_parameters(), minimal(regions=[
      {"type": "source", "w": [1.0, None], "h": [0.2, 0.4]},
      {"type": "source", "at": [0, 0, 0]},
  ]))
  assert scene.default_source() == pytest.approx((2.5, 0.3, 2.5))


def test_volumetric_scene():
  grid = volumetric_scene(coarse_parameters(), 10).build()
  assert grid.real_shape == (10, 1.0, 1.0)
  assert ((grid.geometry & SOURCE_REGION_FLAG) > 0).sum() == 1
