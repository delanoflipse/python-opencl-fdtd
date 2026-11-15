"""Simulation sessions behind the web viewer.

A session owns one scene, grid and OpenCL simulation. ``SlowMotionSession``
advances an impulse/sine response step by step; ``SweepSession`` runs a sine
per frequency in a background thread and keeps the Leq field of every
frequency, so (mode) patterns can be inspected per frequency.
"""
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fdtd.grid import LISTENER_FLAG, SOURCE_FLAG, WALL_FLAG
from fdtd.impulse_generators import GENERATORS, SimpleSinoidGenerator, create_generator
from fdtd.math.octaves import get_octaval_center_frequencies
from fdtd.parameters import SimulationParameters
from fdtd.physical_constants import C_AIR
from fdtd.scenes import SCENES, create_scene
from fdtd.simulation import Simulation

# name -> (label, lambda (courant number), a, b)
SCHEMES = {
    "slf": ("Standard leapfrog (SLF)", 1 / math.sqrt(3), 0.0, 0.0),
    "iwb": ("Interpolated wideband (IWB)", 1.0, 1 / 4, 1 / 16),
    "ccp": ("Close-cubic packed (CCP)", 1.0, 1 / 4, 0.0),
    "idwm": ("Interpolated DWM (IDWM)", 1 / math.sqrt(3), 0.2034, 0.0438),
}
PRECISIONS = ("float", "double")
SOURCE_TYPES = ("hard", "soft")
AXES = ("w", "h", "d")

MAX_CELLS = 64_000_000
MAX_SWEEP_BYTES = 2 * 1024 ** 3
SWEEP_CHUNK_STEPS = 128

DEFAULTS: Dict[str, Any] = {
    "scene": "shoebox",
    "max_frequency": 200.0,
    "oversampling": 16.0,
    "scheme": "slf",
    "precision": "float",
    "source_type": "hard",
    # slow motion
    "generator": "monopulse",
    "signal_frequency": 100.0,
    # sweep
    "sweep_min": 20.0,
    "sweep_max": 200.0,
    "bands": 24.0,
    "sweep_time": 0.3,
}


class SessionError(ValueError):
  """Invalid request parameters, reported back to the browser"""


def _number(options: Dict[str, Any], key: str, low: float, high: float) -> float:
  value = options.get(key, DEFAULTS.get(key))
  try:
    value = float(value)
  except (TypeError, ValueError):
    raise SessionError(f"{key} must be a number") from None
  if not (low <= value <= high) or math.isnan(value):
    raise SessionError(f"{key} must be between {low:g} and {high:g}")
  return value


def _choice(options: Dict[str, Any], key: str, choices) -> str:
  value = options.get(key, DEFAULTS.get(key))
  if value not in choices:
    raise SessionError(f"{key} must be one of: {', '.join(choices)}")
  return value


def scene_catalogue() -> List[Dict[str, Any]]:
  """Bundled scenes with their nominal size and default source position"""
  parameters = SimulationParameters()
  entries = []
  for key in SCENES:
    scene = create_scene(key, parameters)
    entries.append({
        "key": key,
        "name": scene.name,
        "description": scene.description,
        "size": list(scene.shape),
        "default_source": [round(v, 3) for v in scene.default_source()],
    })
  return entries


def options_payload() -> Dict[str, Any]:
  return {
      "scenes": scene_catalogue(),
      "generators": list(GENERATORS),
      "schemes": [{"key": k, "label": v[0], "lambda": v[1]} for k, v in SCHEMES.items()],
      "precisions": list(PRECISIONS),
      "source_types": list(SOURCE_TYPES),
      "defaults": DEFAULTS,
      "c_air": C_AIR,
      "max_cells": MAX_CELLS,
  }


def room_mode_table(size: Tuple[float, float, float], max_frequency: float, max_order: int = 8) -> List[Dict[str, Any]]:
  """Analytic modes of a rigid box up to max_frequency"""
  width, height, depth = size
  modes = []
  for i in range(max_order + 1):
    for j in range(max_order + 1):
      for k in range(max_order + 1):
        if i == j == k == 0:
          continue
        frequency = (C_AIR / 2) * math.sqrt((i / width) ** 2 + (j / height) ** 2 + (k / depth) ** 2)
        if frequency > max_frequency:
          continue
        axes = (i > 0) + (j > 0) + (k > 0)
        modes.append({
            "frequency": frequency,
            "order": [i, j, k],
            "kind": ("axial", "tangential", "oblique")[axes - 1],
        })
  modes.sort(key=lambda mode: mode["frequency"])
  return modes


def take_slice(volume: np.ndarray, axis: str, index: int) -> Tuple[np.ndarray, str, str]:
  """2D slice of a (w, h, d) volume as displayed: rows top to bottom, columns left to right.

  Returns (slice, horizontal axis, vertical axis). The plan view (h) shows w to
  the right and d downwards; the elevations (w, d) show h upwards.
  """
  if axis == "h":
    return volume[:, index, :].T, "w", "d"
  if axis == "w":
    return volume[index, :, :][::-1, :], "d", "h"
  if axis == "d":
    return volume[:, :, index].T[::-1, :], "w", "h"
  raise SessionError(f"axis must be one of: {', '.join(AXES)}")


class Session:
  mode = "base"

  def __init__(self, options: Dict[str, Any]) -> None:
    scene_key = _choice(options, "scene", list(SCENES))
    scheme_key = _choice(options, "scheme", list(SCHEMES))
    precision = _choice(options, "precision", PRECISIONS)
    source_type = _choice(options, "source_type", SOURCE_TYPES)
    max_frequency = _number(options, "max_frequency", 1.0, 20000.0)
    oversampling = _number(options, "oversampling", 2.0, 256.0)

    parameters = SimulationParameters()
    _label, courant, param_a, param_b = SCHEMES[scheme_key]
    parameters.set_scheme(courant, param_a, param_b)
    parameters.set_max_frequency(max_frequency)
    parameters.set_oversampling(oversampling)
    parameters.set_precision(precision)
    parameters.hard_source = source_type == "hard"
    parameters.set_signal_frequency(self.initial_signal_frequency(options))
    self.parameters = parameters
    self.scheme = scheme_key

    self.scene = create_scene(scene_key, parameters)
    cells = math.prod(max(1, round(size / parameters.dx)) for size in self.scene.shape)
    if cells > MAX_CELLS:
      raise SessionError(
          f"grid would have {cells / 1e6:.1f}M cells (limit {MAX_CELLS / 1e6:.0f}M), "
          "lower the max frequency or oversampling")
    self.grid = self.scene.build()

    source = options.get("source") or self.scene.default_source()
    try:
      source = [float(v) for v in source]
      assert len(source) == 3
    except (TypeError, ValueError, AssertionError):
      raise SessionError("source must be [w, h, d] in metres") from None
    self.source_cell = tuple(
        int(min(max(self.grid.scale(v), 0), parts - 1))
        for v, parts in zip(source, self.grid.grid_shape))
    if self.grid.geometry[self.source_cell] & WALL_FLAG:
      raise SessionError("the source position is inside a wall or object")
    self.grid.select_source_locations([self.source_cell])

    self.lock = threading.Lock()
    self.sim = Simulation(parameters, self.grid)
    self.wall_mask = (self.grid.geometry & WALL_FLAG) > 0

  def initial_signal_frequency(self, options: Dict[str, Any]) -> float:
    return DEFAULTS["signal_frequency"]

  def info(self) -> Dict[str, Any]:
    params = self.parameters
    return {
        "mode": self.mode,
        "scene": self.scene.describe(),
        "grid": {
            "shape": list(self.grid.grid_shape),
            "cells": self.grid.grid_size,
            "dx": params.dx,
            "dt": params.dt,
        },
        "parameters": {
            "max_frequency": params.max_frequency,
            "oversampling": params.oversampling,
            "scheme": self.scheme,
            "precision": params.precision,
            "source_type": "hard" if params.hard_source else "soft",
            "signal_frequency": params.signal_frequency,
        },
        "source": {
            "cell": list(self.source_cell),
            "position": [(i + 0.5) * params.dx for i in self.source_cell],
        },
        "room_modes": room_mode_table(self.scene.shape, params.max_frequency),
        "device": self.sim.program.ctx.devices[0].name.strip(),
    }

  def flags_slice(self, axis: str, index: int) -> Tuple[np.ndarray, str, str]:
    """1 = wall, 2 = source, 4 = listener"""
    index = self._check_index(axis, index)
    geometry = self.grid.geometry
    flags = ((geometry & WALL_FLAG) > 0).astype(np.uint8)
    flags |= ((geometry & SOURCE_FLAG) > 0).astype(np.uint8) << 1
    flags |= ((geometry & LISTENER_FLAG) > 0).astype(np.uint8) << 2
    return take_slice(flags, axis, index)

  def _check_index(self, axis: str, index: int) -> int:
    if axis not in AXES:
      raise SessionError(f"axis must be one of: {', '.join(AXES)}")
    parts = self.grid.grid_shape[AXES.index(axis)]
    if not 0 <= index < parts:
      raise SessionError(f"slice index must be between 0 and {parts - 1}")
    return index

  def close(self) -> None:
    pass


class SlowMotionSession(Session):
  mode = "slowmotion"

  def __init__(self, options: Dict[str, Any]) -> None:
    super().__init__(options)
    self.generator_name = _choice(options, "generator", list(GENERATORS))
    self.sim.generator = create_generator(self.generator_name, self.parameters.signal_frequency)
    self.sim.sync_read_buffers()
    self.sim.reset()

  def initial_signal_frequency(self, options: Dict[str, Any]) -> float:
    return _number(options, "signal_frequency", 1.0, 20000.0)

  def info(self) -> Dict[str, Any]:
    info = super().info()
    info["generator"] = self.generator_name
    info["state"] = self.state()
    return info

  def state(self) -> Dict[str, Any]:
    return {"time": float(self.sim.time), "iteration": self.sim.iteration}

  def step(self, count: int) -> None:
    if not 1 <= count <= 100_000:
      raise SessionError("step count must be between 1 and 100000")
    with self.lock:
      self.sim.step(count)

  def reset(self) -> None:
    with self.lock:
      self.sim.reset()

  def pressure_slice(self, axis: str, index: int) -> Tuple[np.ndarray, str, str]:
    index = self._check_index(axis, index)
    with self.lock:
      data, horizontal, vertical = take_slice(self.grid.pressure, axis, index)
      data = data.astype(np.float32)
    walls, _, _ = take_slice(self.wall_mask, axis, index)
    data[walls] = np.nan
    return data, horizontal, vertical


class SweepSession(Session):
  """Steady-state sine response per frequency (as in ``fdtd run``) for one source"""
  mode = "sweep"

  def __init__(self, options: Dict[str, Any]) -> None:
    super().__init__(options)
    sweep_min = _number(options, "sweep_min", 1.0, 20000.0)
    sweep_max = _number(options, "sweep_max", sweep_min, 20000.0)
    if sweep_max > self.parameters.max_frequency * 1.0001:
      raise SessionError("the sweep cannot go above the grid's max frequency")
    bands = _number(options, "bands", 1.0, 1000.0)
    self.sweep_time = _number(options, "sweep_time", 0.001, 60.0)
    self.frequencies = get_octaval_center_frequencies(sweep_min, sweep_max, fraction=bands)
    if self.frequencies.size == 0:
      raise SessionError("no band centre frequencies in this range")
    stored = self.frequencies.size * self.grid.grid_size * 4
    if stored > MAX_SWEEP_BYTES:
      raise SessionError(
          f"storing every frequency needs {stored / 1024 ** 3:.1f} GiB, "
          "use fewer bands or a coarser grid")
    self.steps_per_frequency = max(1, int(self.sweep_time / self.parameters.dt))

    self.leq = np.full((self.frequencies.size, *self.grid.grid_shape), np.nan, dtype=np.float32)
    self.leq_range = np.full((self.frequencies.size, 2), np.nan)
    self.spl = np.full((self.frequencies.size, 3), np.nan)  # avg, min, max over listeners
    listeners = ((self.grid.geometry & LISTENER_FLAG) > 0) & ~self.wall_mask
    self.listener_mask = listeners if listeners.any() else ~self.wall_mask

    self.completed = 0
    self.progress = 0.0
    self.status = "running"
    self.error: Optional[str] = None
    self.started = time.time()
    self.finished: Optional[float] = None
    self.cancel_event = threading.Event()
    self.thread = threading.Thread(target=self._run, name="fdtd-sweep", daemon=True)
    self.thread.start()

  def initial_signal_frequency(self, options: Dict[str, Any]) -> float:
    return _number(options, "sweep_min", 1.0, 20000.0)

  def _run(self) -> None:
    try:
      total = self.frequencies.size
      for index, frequency in enumerate(self.frequencies):
        with self.lock:
          self.parameters.set_signal_frequency(float(frequency))
          self.sim.generator = SimpleSinoidGenerator(self.parameters.signal_frequency)
          self.scene.rebuild()
          self.sim.reset()
          self.sim.sync_read_buffers()
        done = 0
        while done < self.steps_per_frequency:
          if self.cancel_event.is_set():
            self.status = "cancelled"
            return
          count = min(SWEEP_CHUNK_STEPS, self.steps_per_frequency - done)
          done += count
          with self.lock:
            self.sim.step(count, read_back=done >= self.steps_per_frequency)
          self.progress = (index + done / self.steps_per_frequency) / total

        leq = self.grid.analysis[..., self.grid.analysis_keys["LEQ"]]
        self.leq[index] = leq
        # colour range: robust percentiles, so the few cells next to the
        # (hard) source do not squash the pattern in the rest of the room
        air = leq[~self.wall_mask]
        self.leq_range[index] = np.percentile(air, [2, 98])
        listeners = leq[self.listener_mask]
        self.spl[index] = (np.mean(listeners), np.min(listeners), np.max(listeners))
        self.completed = index + 1
      self.status = "done"
    except Exception as error:  # reported to the browser
      self.status = "error"
      self.error = f"{type(error).__name__}: {error}"
      raise
    finally:
      self.finished = time.time()

  def info(self) -> Dict[str, Any]:
    info = super().info()
    info["sweep"] = {
        "frequencies": self.frequencies.tolist(),
        "time_per_frequency": self.sweep_time,
        "steps_per_frequency": self.steps_per_frequency,
        "listener_cells": int(self.listener_mask.sum()),
    }
    return info

  def sweep_status(self) -> Dict[str, Any]:
    end = self.finished if self.finished is not None else time.time()
    elapsed = end - self.started
    remaining = elapsed * (1 - self.progress) / self.progress \
        if self.progress > 0 and self.status == "running" else None
    completed = self.completed

    def column(values: np.ndarray) -> List[float]:
      return [float(v) if np.isfinite(v) else None for v in values[:completed]]
    return {
        "status": self.status,
        "error": self.error,
        "progress": self.progress,
        "completed": completed,
        "total": int(self.frequencies.size),
        "elapsed": elapsed,
        "remaining": remaining,
        "spl_avg": column(self.spl[:, 0]),
        "spl_min": column(self.spl[:, 1]),
        "spl_max": column(self.spl[:, 2]),
    }

  def leq_slice(self, frequency_index: int, axis: str, index: int) -> Tuple[np.ndarray, str, str, Tuple[float, float]]:
    index = self._check_index(axis, index)
    if not 0 <= frequency_index < self.completed:
      raise SessionError("this frequency has not been simulated yet")
    data, horizontal, vertical = take_slice(self.leq[frequency_index], axis, index)
    low, high = self.leq_range[frequency_index]
    return data, horizontal, vertical, (float(low), float(high))

  def close(self) -> None:
    self.cancel_event.set()
    self.thread.join(timeout=30)
