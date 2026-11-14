import pytest

from fdtd.cli import _build_parser


def test_run_defaults():
  args = _build_parser().parse_args(["run"])
  assert args.scene == "shoebox"
  assert args.deviation == "spectral"
  assert isinstance(args.distance, float)


def test_run_rejects_unknown_scene():
  with pytest.raises(SystemExit):
    _build_parser().parse_args(["run", "--scene", "nope"])


def test_benchmark_options():
  args = _build_parser().parse_args(
      ["benchmark", "--scene", "office", "--precision", "double"])
  assert args.scene == "office"
  assert args.precision == "double"
