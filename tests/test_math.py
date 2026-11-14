import numpy as np
import pytest

from fdtd.impulse_generators import GENERATORS, create_generator
from fdtd.math.decibel_weightings import get_a_weighting
from fdtd.math.octaves import get_octaval_center_frequencies


def test_octave_bands_are_spaced_by_the_fraction():
  bands = get_octaval_center_frequencies(20, 200, fraction=3)
  np.testing.assert_allclose(bands[1:] / bands[:-1], 2 ** (1 / 3))


def test_octave_bands_cover_the_range():
  bands = get_octaval_center_frequencies(20, 200, fraction=3)
  assert bands[0] <= 20
  assert bands[-1] >= 200


def test_octave_bands_contain_1khz_reference():
  bands = get_octaval_center_frequencies(500, 2000, fraction=1)
  np.testing.assert_allclose(bands, [500, 1000, 2000])


def test_exclusive_bands_stay_within_the_range():
  inclusive = get_octaval_center_frequencies(20, 200, fraction=1)
  exclusive = get_octaval_center_frequencies(20, 200, fraction=1, exclusive=True)
  np.testing.assert_allclose(exclusive, [31.25, 62.5, 125])
  np.testing.assert_allclose(inclusive, [15.625, 31.25, 62.5, 125, 250])


@pytest.mark.parametrize("frequency, expected_db", [
    (1000, 0.0),
    (100, -19.1),
    (10000, -2.5),
])
def test_a_weighting_matches_reference_values(frequency, expected_db):
  assert get_a_weighting(frequency) == pytest.approx(expected_db, abs=0.1)


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_registered_generator_produces_a_signal(name):
  generator = create_generator(name, 400.0)
  dt = 1 / 20000
  values = [generator.generate(i * dt, i) for i in range(int(0.5 / dt))]
  assert max(abs(v) for v in values) == pytest.approx(1.0, abs=0.01)


def test_unknown_generator_is_rejected():
  with pytest.raises(ValueError, match="Unknown generator"):
    create_generator("does-not-exist", 400.0)
