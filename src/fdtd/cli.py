"""Command line interface: ``fdtd run ...`` and ``fdtd benchmark ...``."""
import argparse
from pathlib import Path
from typing import List, Optional

from fdtd.scenes import SCENES

PRECISIONS = ["float", "double"]


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="fdtd", description="OpenCL accelerated acoustic FDTD simulation")
  commands = parser.add_subparsers(dest="command", required=True)

  run = commands.add_parser(
      "run", help="frequency sweep over all source positions of a scene")
  run.add_argument("-s", "--scene", default="shoebox", choices=SCENES)
  run.add_argument("-t", "--time", default=0.3, type=float,
                   help="simulated time per frequency (s)")
  run.add_argument("-o", "--oversampling", default=16, type=float)
  run.add_argument("-f", "--frequency", default=200, type=float,
                   help="maximum frequency (Hz)")
  run.add_argument("-b", "--bands", default=24, type=float,
                   help="bands per octave")
  run.add_argument("-x", "--speakers", default=1, type=int)
  run.add_argument("--distance", default=2.0, type=float,
                   help="minimum distance between speakers (m)")
  run.add_argument("--deviation", default="spectral",
                   choices=["spectral", "spatial"],
                   help="metric used to rank source positions")
  run.add_argument("--precision", default=None, choices=PRECISIONS)
  run.add_argument("--output", default=Path("output"), type=Path,
                   help="directory for logs, CSV and charts")
  run.add_argument("--live", action="store_true",
                   help="show the chart while the sweep runs")
  run.add_argument("--novisuals", action="store_true")
  run.add_argument("--nologs", action="store_true")
  run.add_argument("--nocsv", action="store_true")

  bench = commands.add_parser("benchmark", help="time kernel throughput")
  bench.add_argument("-i", "--iterations", default=2 ** 13, type=int,
                     help="iterations per step")
  bench.add_argument("-s", "--steps", default=2 ** 3, type=int)
  bench.add_argument("-o", "--oversampling", default=16, type=float)
  bench.add_argument("--scene", default=None, choices=SCENES,
                     help="benchmark a named scene instead of an empty cube")
  bench.add_argument("-v", "--volume", default=50, type=float,
                     help="volume of the empty cube (m^3)")
  bench.add_argument("--precision", default=None, choices=PRECISIONS)
  return parser


def main(argv: Optional[List[str]] = None) -> None:
  arguments = _build_parser().parse_args(argv)
  print(arguments)

  # imported lazily so `--help` does not compile numba/OpenCL code
  if arguments.command == "run":
    from fdtd.sweep import SweepConfig, run_full_sweep
    run_full_sweep(SweepConfig(
        scene=arguments.scene,
        simulated_time=arguments.time,
        max_frequency=arguments.frequency,
        oversampling=arguments.oversampling,
        bands=arguments.bands,
        speakers=arguments.speakers,
        min_distance=arguments.distance,
        precision=arguments.precision,
        deviation=arguments.deviation,
        output_dir=arguments.output,
        write_logs=not arguments.nologs,
        write_csv=not arguments.nocsv,
        write_chart=not arguments.novisuals,
        live_chart=arguments.live,
    ))
  elif arguments.command == "benchmark":
    from fdtd.benchmark import run_benchmark
    run_benchmark(
        iterations_per_step=arguments.iterations,
        step_count=arguments.steps,
        oversampling=arguments.oversampling,
        scene=arguments.scene,
        volume=arguments.volume,
        precision=arguments.precision,
    )


if __name__ == "__main__":
  main()
