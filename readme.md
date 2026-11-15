# OpenCL accelerated Acoustic Finite Difference Simulation

This is project is part of my bachelor thesis for the Computer Science and Engineering degree of the TU Delft. For more information about the CSE3000 course, visit [https://github.com/TU-Delft-CSE/Research-Project](https://github.com/TU-Delft-CSE/Research-Project).

## Requirements

Python 3.9 or newer and an OpenCL device. Development uses Python 3.12 (`.python-version`).

To install with [uv](https://docs.astral.sh/uv/) (installs Python, the `fdtd` package and all dependencies into `.venv`):

```bash
uv sync
```

Then run any script with `uv run`, e.g. `uv run scripts/visual_fdtd.py`.

Alternatively, to install with pip, use:

```bash
pip install -e .
```

## Development

```bash
uv run pytest        # tests; the simulation tests are skipped without an OpenCL device
uv run ruff check    # lint
```

## Run experiment(s)

The simulation code lives in the `fdtd` package (`src/fdtd`). Headless experiments go through the `fdtd` command:

```bash
uv run fdtd run --scene office --time 0.3 --bands 24   # full sweep over all source positions
uv run fdtd run --scene real-scene --deviation spatial --live
uv run fdtd benchmark --scene shoebox                  # time kernel throughput
uv run fdtd run --help                                 # all options and available scenes
```

`fdtd run` writes a log, CSV and chart per run to `output/` (override with `--output`). Source positions are ranked by `--deviation spectral` (flatness of the averaged SPL over the bands, the default) or `--deviation spatial` (average per-cell deviation over the listener region).

Interactive scripts live in `scripts/` (run with e.g. `uv run scripts/visual_fdtd.py`):

- `visual_fdtd.py` runs a scene and visualises a slice of the 3d space.
- `mode_visual.py` runs a scene and shows the field at a single frequency (`-f`) as it builds up, via a running DFT of the FDTD pressure.
- `frequency_visual.py` solves the steady-state field at a single frequency directly in the frequency domain (`fdtd.frequency_domain`, no time stepping) and shows the room mode shape. Add `--sweep FMIN FMAX` for a clickable frequency response at the receiver; `--beta` overrides the wall absorption.
- `sweep_visual.py` runs a single frequency sweep to determine the frequency response over the listener region.
- `sources_visual.py` plots the available impulse generators.
- `display_scene.py` shows a voxelized 3d scene (`--scene`, `--oversampling`).
- `parse_output.py` summarises the CSV of an `fdtd run` (`uv run scripts/parse_output.py output/<run>.csv --scene office`).
- `comparison_chart.py` compares simulated and measured responses (needs `fdtd.util`, which is not on this branch).

### Precision

The pressure field is simulated in single precision (`float`) by default; the RMS / Leq accumulation is always double precision. Single precision stays within ~0.004 dB Leq of double precision up to an oversampling of 128. In decaying fields it leaves a round-off noise floor of roughly 1e-7 times the source amplitude, which shows up when a plot autoscales to a field that has decayed below it. For full double precision, call `parameters.set_precision("double")`, or pass `--precision double` to `fdtd run` / `fdtd benchmark`.

### Run on the DHPC

`jobs/run_script.sh` runs a command on the DHPC:

```
sbatch jobs/run_script.sh -m fdtd run [...]
```
