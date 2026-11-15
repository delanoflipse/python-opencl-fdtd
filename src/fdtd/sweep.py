"""Full frequency sweep over all source positions of a scene.

For every (set of) source position(s), a sine is simulated for each octave
band centre frequency. The resulting SPL per band is logged, written to CSV
and summarised in a chart, so source positions can be compared.
"""
import csv
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import time
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from fdtd.analysis.frequency_sweep import get_avg_dev, get_avg_spl, run_sweep_analysis
from fdtd.analysis.source_pairs import get_n_pairs_with_min_distance
from fdtd.impulse_generators import SimpleSinoidGenerator
from fdtd.math.octaves import get_octaval_center_frequencies
from fdtd.parameters import SimulationParameters
from fdtd.scenes import create_scene
from fdtd.simulation import Simulation
from fdtd.styles import use_style

DEVIATION_METRICS = ("spectral", "spatial")


@dataclass
class SweepConfig:
  scene: str = "shoebox"
  simulated_time: float = 0.3
  max_frequency: float = 200.0
  oversampling: float = 16
  bands: float = 24
  speakers: int = 1
  min_distance: float = 2.0
  precision: Optional[str] = None
  # "spectral": flatness of the averaged SPL over the bands
  # "spatial": average per-cell deviation over the listener region
  deviation: str = "spectral"
  output_dir: Path = Path("output")
  write_logs: bool = True
  write_csv: bool = True
  write_chart: bool = True
  live_chart: bool = False


@dataclass
class SweepResult:
  frequencies: np.ndarray
  deviations: List[float] = field(default_factory=list)
  spl_per_source: List[List[float]] = field(default_factory=list)


def format_duration(seconds: float) -> str:
  hours = math.floor(seconds / 3600)
  minutes = math.floor((seconds - hours * 3600) / 60)
  rest = seconds - hours * 3600 - minutes * 60
  return f'{hours}h {minutes}m {rest:.2f}s'


def _create_logger(log_file: Optional[Path]) -> logging.Logger:
  log = logging.getLogger("FDTD")
  log.setLevel(logging.DEBUG)
  log.handlers.clear()
  formatter = logging.Formatter(
      "%(asctime)s [%(levelname)-5.5s] - %(message)s")

  if log_file is not None:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.setFormatter(formatter)
  log.addHandler(console_handler)
  return log


class _SweepCsv:
  def __init__(self, path: Path, speakers: int, frequencies: np.ndarray, dx: float):
    self.file = open(path, 'w', encoding="utf-8", newline='')
    self.writer = csv.writer(self.file)
    self.speakers = speakers
    self.dx = dx

    header = ["Time", "Index"]
    for i in range(speakers):
      header += [f"W{i} idx", f"w{i} (m)", f"H{i} idx",
                 f"h{i} (m)", f"D{i} idx", f"d{i} (m)"]
    header += ["Deviation", "SPL (avg dB)", "Bands (SPL dB):",
               *(f'{x:.2f}' for x in frequencies.tolist())]
    self._write(header)

  def write_source(self, source_index: int, source_set, deviation: float, spl_values: List[float]) -> None:
    row = [time(), source_index]
    for i in range(self.speakers):
      (w, h, d) = source_set[i]
      row += [w, f'{(w + 0.5) * self.dx:.2f}',
              h, f'{(h + 0.5) * self.dx:.2f}',
              d, f'{(d + 0.5) * self.dx:.2f}']
    row += [f'{deviation:.4f}', f'{np.average(spl_values):.4f}', "",
            *(f'{x:.2f}' for x in spl_values)]
    self._write(row)

  def _write(self, row) -> None:
    self.writer.writerow(row)
    self.file.flush()

  def close(self) -> None:
    self.file.close()


class _SweepChart:
  def __init__(self, frequencies: np.ndarray, room_modes):
    self.frequencies = frequencies
    self.min_dev = float("inf")
    self.max_dev = -float("inf")

    use_style("poster")
    self.fig = plt.gcf()
    self.fig.set_dpi(150)
    self.fig.set_size_inches(1920 / self.fig.get_dpi(),
                             1080 / self.fig.get_dpi(), forward=True)
    axes_shape = (2, 2)
    self.axis_deviation = plt.subplot2grid(axes_shape, (0, 0))
    self.axis_spl = plt.subplot2grid(axes_shape, (0, 1))
    self.axis_best_spl = plt.subplot2grid(axes_shape, (1, 0), colspan=2)
    self.axes = [self.axis_best_spl, self.axis_deviation, self.axis_spl]

    self.axis_deviation.set_title("Deviation per sweep")
    self.axis_deviation.set_xlabel("Sweep Index")
    self.axis_deviation.set_ylabel("Deviation")

    self.axis_spl.set_title("SPL values per frequency band")
    self.axis_best_spl.set_title("-")
    for ax in [self.axis_best_spl, self.axis_spl]:
      ax.set_xlabel("Frequency (hz)")
      ax.set_ylabel("SPL Leq (dB)")
      ax.set_xscale('log')
      ax.set_xticks([20, 25, 30, 40, 50, 60, 80, 100, 120, 160, 200])
      ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    self.deviation_plot, = self.axis_deviation.plot([], [], "-")
    self.best_spl_plot, self.worst_spl_plot = self.axis_best_spl.plot(
        [], [], [], "-")
    self.max_spl_plot, = self.axis_best_spl.plot([], [], "--")
    self.min_spl_plot, = self.axis_best_spl.plot([], [], "--")

    for (modal_frequency, _axis_type) in room_modes:
      if modal_frequency <= frequencies[-1]:
        self.axis_best_spl.axvline(
            modal_frequency, linestyle='--', color='k', alpha=0.5)

  def add_source(self, source_index: int, deviation: float, spl_values, min_spl_values, max_spl_values) -> None:
    if deviation == 0.0:
      return
    self.axis_spl.plot(self.frequencies, spl_values)
    if deviation < self.min_dev:
      self.min_dev = deviation
      self.axis_best_spl.set_title(
          f"best/worst ({source_index}) SPL values per frequency band")
      self.best_spl_plot.set_data(self.frequencies, spl_values)
      self.min_spl_plot.set_data(self.frequencies, min_spl_values)
      self.max_spl_plot.set_data(self.frequencies, max_spl_values)
    if deviation > self.max_dev:
      self.max_dev = deviation
      self.worst_spl_plot.set_data(self.frequencies, spl_values)

  def set_deviations(self, sources, deviations) -> None:
    self.deviation_plot.set_data(sources, deviations)

  def redraw(self) -> None:
    for ax in self.axes:
      ax.relim()
      ax.autoscale_view()
    plt.tight_layout()
    self.fig.canvas.draw()
    self.fig.canvas.flush_events()

  def save(self, path: Path) -> None:
    for ax in self.axes:
      ax.relim()
      ax.autoscale_view()
    plt.tight_layout()
    plt.savefig(path, dpi=300)


def run_full_sweep(config: SweepConfig) -> SweepResult:
  if config.deviation not in DEVIATION_METRICS:
    raise ValueError(f"deviation must be one of {DEVIATION_METRICS}")
  if not config.live_chart:
    matplotlib.use('Agg')

  parameters = SimulationParameters()
  parameters.set_oversampling(config.oversampling)
  parameters.set_max_frequency(config.max_frequency)
  if config.precision is not None:
    parameters.set_precision(config.precision)

  runtime_steps = int(config.simulated_time / parameters.dt)
  frequencies = get_octaval_center_frequencies(20, 200, fraction=config.bands)

  scene = create_scene(config.scene, parameters)
  grid = scene.build()
  sim = Simulation(grid=grid, parameters=parameters)
  sim.print_statistics()
  leq_index = grid.analysis_keys["LEQ"]
  position_sets = get_n_pairs_with_min_distance(
      grid.source_set, config.speakers, parameters.dx, config.min_distance)

  scene_name = scene.__class__.__name__
  output_uid = (f'{datetime.now().strftime("%Y-%m-%d %H_%M_%S")} {scene_name} '
                f'[{config.simulated_time * 1000:.0f}ms-{config.max_frequency}f-'
                f'{config.oversampling}o-{config.bands}b-{config.speakers}x]')
  output_dir = Path(config.output_dir)
  if config.write_logs or config.write_csv or config.write_chart:
    output_dir.mkdir(parents=True, exist_ok=True)

  log = _create_logger(output_dir / f"{output_uid}.log"
                       if config.write_logs else None)
  log.info('---- Starting simulation ----')
  log.info('%d pair(s)', len(position_sets))
  log.info('%d steps per sim', runtime_steps)
  log.info('%d frequencies', frequencies.size)
  log.info('Scene: %s', scene_name)
  log.info('Deviation metric: %s', config.deviation)
  log.info('-----------------------------')

  csv_out = _SweepCsv(output_dir / f"{output_uid}.csv", config.speakers,
                      frequencies, parameters.dx) if config.write_csv else None
  chart = _SweepChart(frequencies, scene.get_room_modes()) \
      if config.write_chart or config.live_chart else None
  if config.live_chart:
    plt.show(block=False)

  if config.deviation == "spatial":
    sweep_sum = grid.create_grid("float64")
    sweep_sum_sqr = grid.create_grid("float64")
    sweep_deviation = grid.create_grid("float64")
    sweep_ranking = grid.create_grid("float64")

  result = SweepResult(frequencies=frequencies)
  timings = []
  start_time = time()

  for source_index, source_set in enumerate(position_sets):
    start = time()
    grid.select_source_locations(source_set)
    log.info('Picked source set %d/%d with positions:',
             source_index + 1, len(position_sets))
    for pos in source_set:
      log.info('%d, %d, %d', *pos)

    if config.deviation == "spatial":
      for array in (sweep_sum, sweep_sum_sqr, sweep_deviation, sweep_ranking):
        array.fill(0.0)

    spl_values, min_spl_values, max_spl_values = [], [], []
    for (index,), frequency in np.ndenumerate(frequencies):
      parameters.set_signal_frequency(frequency)
      sim.generator = SimpleSinoidGenerator(parameters.signal_frequency)
      scene.rebuild()
      sim.reset()
      sim.sync_read_buffers()
      sim.step(runtime_steps)

      if config.deviation == "spatial":
        run_sweep_analysis(grid.analysis, sweep_sum, sweep_sum_sqr,
                           sweep_deviation, sweep_ranking, leq_index, index + 1)
      avg_spl, min_spl, max_spl = get_avg_spl(
          grid.analysis, grid.geometry, leq_index)
      spl_values.append(avg_spl)
      min_spl_values.append(min_spl)
      max_spl_values.append(max_spl)
      log.info(f'[{source_index}] {frequency:.2f}hz: {avg_spl:.2f} SPL (dB)')

    if config.deviation == "spatial":
      deviation = get_avg_dev(sweep_deviation, grid.geometry)
    else:
      deviation = float(np.sum(np.power(np.diff(spl_values, n=1), 2)))
    result.deviations.append(deviation)
    result.spl_per_source.append(spl_values)
    log.info(f'Deviation: {deviation}')

    if chart is not None:
      chart.add_source(source_index, deviation, spl_values,
                       min_spl_values, max_spl_values)
      chart.set_deviations(range(source_index + 1), result.deviations)
      if config.live_chart:
        chart.redraw()
    if csv_out is not None:
      csv_out.write_source(source_index, source_set, deviation, spl_values)

    elapsed = time() - start
    timings.append(elapsed)
    avg_timing = np.average(timings)
    time_left = (len(position_sets) - source_index - 1) * avg_timing
    log.info(f'Elapsed: {elapsed:.1f}s, est: {avg_timing:.1f}s/run = '
             f'{format_duration(time_left)} left')
    log.info('----- End iteration %d -----', source_index)

  log.info(
      f'Time elapsed for full sweep: {format_duration(time() - start_time)}')

  if chart is not None and config.write_chart:
    chart.save(output_dir / f'{output_uid}.png')
  if csv_out is not None:
    csv_out.close()
  if config.live_chart:
    plt.show(block=True)

  return result
