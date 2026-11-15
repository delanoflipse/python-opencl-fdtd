import pytest

from fdtd.analysis.source_pairs import get_n_pairs_with_min_distance
from fdtd.grid import LISTENER_FLAG, SOURCE_REGION_FLAG
from fdtd.parameters import SimulationParameters
from fdtd.scenes import SCENES, Scene, create_scene


def coarse_parameters() -> SimulationParameters:
  params = SimulationParameters()
  params.set_max_frequency(200)
  params.set_oversampling(4)
  return params


def test_unknown_scene_is_rejected():
  with pytest.raises(ValueError, match="Unknown scene"):
    create_scene("does-not-exist", coarse_parameters())


# ConcertHallScene only marks a source position so far (see TODO in the scene)
INCOMPLETE_SCENES = {"concert"}


@pytest.mark.parametrize("name", [
    pytest.param(name, marks=pytest.mark.xfail(
        reason="scene has no listener region yet", strict=True))
    if name in INCOMPLETE_SCENES else name
    for name in sorted(SCENES)
])
def test_registered_scene_builds(name):
  scene = create_scene(name, coarse_parameters())
  assert isinstance(scene, Scene)
  grid = scene.build()
  assert grid.is_build
  assert ((grid.geometry & LISTENER_FLAG) > 0).any(), "scene has no listeners"
  assert ((grid.geometry & SOURCE_REGION_FLAG) > 0).any(), "scene has no source region"
  assert len(grid.source_set) > 0


def test_source_pairs_respect_minimum_distance():
  locations = [(0, 0, 0), (1, 0, 0), (5, 0, 0)]
  # cells are 1 m apart, speakers must be at least 2 m apart
  pairs = get_n_pairs_with_min_distance(locations, 2, 1.0, 2.0)
  assert [tuple(p) for p in pairs] == [((0, 0, 0), (5, 0, 0)), ((1, 0, 0), (5, 0, 0))]


def test_single_sources_are_returned_as_sets():
  locations = [(0, 0, 0), (1, 0, 0)]
  assert get_n_pairs_with_min_distance(locations, 1, 1.0, 2.0) == [[(0, 0, 0)], [(1, 0, 0)]]
