import time

import numpy as np
import pytest

from conftest import requires_opencl
from fdtd.cli import _build_parser
from fdtd.web.session import (SessionError, SlowMotionSession, SweepSession, options_payload,
                              room_mode_table, take_slice)

COARSE = {"scene": "shoebox", "max_frequency": 100, "oversampling": 6}


def test_take_slice_orientation():
  volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)  # (w, h, d)
  plan, horizontal, vertical = take_slice(volume, "h", 1)
  assert (horizontal, vertical) == ("w", "d")
  assert plan.shape == (4, 2)
  assert plan[3, 1] == volume[1, 1, 3]  # row = d (downwards), column = w

  side, horizontal, vertical = take_slice(volume, "w", 1)
  assert (horizontal, vertical) == ("d", "h")
  assert side.shape == (3, 4)
  assert side[0, 2] == volume[1, 2, 2]  # top row is the highest h

  front, horizontal, vertical = take_slice(volume, "d", 3)
  assert (horizontal, vertical) == ("w", "h")
  assert front[2, 1] == volume[1, 0, 3]

  with pytest.raises(SessionError):
    take_slice(volume, "x", 0)


def test_room_mode_table():
  modes = room_mode_table((5.0, 2.5, 4.0), 100.0)  # first oblique mode: 88 Hz
  assert modes[0]["order"] == [1, 0, 0]
  assert modes[0]["frequency"] == pytest.approx(343.21 / 10)
  assert all(a["frequency"] <= b["frequency"] for a, b in zip(modes, modes[1:]))
  assert {m["kind"] for m in modes} == {"axial", "tangential", "oblique"}


def test_options_list_the_bundled_scenes():
  options = options_payload()
  keys = [scene["key"] for scene in options["scenes"]]
  assert keys[0] == "shoebox" and "real-scene" in keys
  assert len(options["scenes"][0]["default_source"]) == 3


def test_web_command():
  args = _build_parser().parse_args(["web", "--port", "9000", "--no-browser"])
  assert args.port == 9000 and args.no_browser


@pytest.mark.parametrize("options, message", [
    ({"scene": "nope"}, "scene must be one of"),
    ({"oversampling": "lots"}, "oversampling must be a number"),
    ({"max_frequency": 20000, "oversampling": 64}, "M cells"),
    ({"source_type": "loud"}, "source_type"),
])
def test_invalid_session_options(options, message):
  with pytest.raises(SessionError, match=message):
    SlowMotionSession({**COARSE, **options})


@requires_opencl
def test_slow_motion_session():
  session = SlowMotionSession({**COARSE, "generator": "monopulse", "signal_frequency": 50})
  info = session.info()
  assert info["state"] == {"time": 0.0, "iteration": 0}
  assert len(info["scene"]["regions"]) == 2

  session.step(30)
  axis_index = info["source"]["cell"][1]
  data, horizontal, vertical = session.pressure_slice("h", axis_index)
  shape = info["grid"]["shape"]
  assert data.dtype == np.float32 and data.shape == (shape[2], shape[0])
  assert np.nanmax(np.abs(data)) > 0
  assert session.state()["iteration"] == 30

  flags, _, _ = session.flags_slice("h", axis_index)
  assert ((flags & 2) > 0).sum() == 1  # the active source

  session.reset()
  assert session.state()["iteration"] == 0
  with pytest.raises(SessionError):
    session.pressure_slice("h", shape[1])


@requires_opencl
def test_source_inside_a_wall_is_rejected():
  with pytest.raises(SessionError, match="inside a wall"):
    SlowMotionSession({**COARSE, "scene": "bedroom", "source": [3.4, 0.2, 1.4]})


@requires_opencl
def test_sweep_session():
  session = SweepSession({**COARSE, "sweep_min": 30, "sweep_max": 60, "bands": 3, "sweep_time": 0.05})
  try:
    session.thread.join(timeout=60)
    status = session.sweep_status()
    assert status["status"] == "done"
    assert status["completed"] == status["total"] == len(session.frequencies)
    assert all(np.isfinite(status["spl_avg"]))
    data, _, _, (low, high) = session.leq_slice(0, "d", 2)
    assert low < high and np.nanmax(data) > 0
  finally:
    session.close()


@requires_opencl
def test_sweep_can_be_cancelled():
  session = SweepSession({**COARSE, "sweep_min": 20, "sweep_max": 100, "bands": 24, "sweep_time": 2})
  time.sleep(0.2)
  session.close()
  assert session.sweep_status()["status"] == "cancelled"
  with pytest.raises(SessionError):
    session.leq_slice(session.frequencies.size - 1, "h", 0)
