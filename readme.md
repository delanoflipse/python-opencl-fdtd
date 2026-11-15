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

### Web viewer

```bash
uv run fdtd web            # opens http://localhost:8765/ (--port, --no-browser, --verbose)
```

Pick a view, then the parameters (scene, grid resolution, scheme, source position and type, signal or sweep range), then explore the result in a split view: the room's regions in 3D with a movable slice (plan, side or front), and that slice in 2D.

- **Slow motion** shows the pressure in the slice. Play/pause, step forward (with a step size) and reset; space and → also work.
- **Full sweep** runs a steady sine per band centre frequency for the chosen source (the same method as `fdtd run`) with a progress bar, then shows the listener level per frequency against the analytic box modes, and the Leq pattern in the slice for any simulated frequency. Modes show up best in reverberant rooms; try `shoebox-reverberant` with ~1.5 s per frequency.

The browser loads three.js from cdn.jsdelivr.net, so the viewer needs internet access.

### Scenes

Scenes are JSON files in `src/fdtd/scenes/data/`: a room size plus region markers (not voxels), which are voxelised for every grid resolution. `--scene` accepts a bundled name or a path to a `.json` file. The full format is documented in `src/fdtd/scenes/base.py`. A short example:

```json
{
  "name": "Bedroom",
  "size": [3.6, 2.6, 4.2],
  "boundaries": {"all": "painted_concrete", "height_min": "laminate", "width_min": ["glass", 0.05]},
  "regions": [
    {"name": "closet", "type": "wall", "material": "wood", "w": ["width - 0.6", null], "h": [null, 2.0], "d": [0.9, 1.9]},
    {"name": "speaker", "type": "source", "at": [0.3, 0.2, 0.3]},
    {"name": "bed", "type": "listener", "w": [1.75, "width - 0.45"], "h": [0.8, 1.3], "d": ["depth - 2.0", null]}
  ]
}
```

Region types are `wall`, `source` (candidate source positions) and `listener` (analysed cells). A box is given per axis (w = width, h = height, d = depth) as `[min, max]` in metres, where a missing axis or `null` bound runs to the room edge; `at` marks a single cell. Coordinates can be expressions over `width`, `height`, `depth` and `dx`. Materials are names from `fdtd.materials.MATERIALS`, a beta value, or a list that is averaged. Regions are applied in order.

### Scripts

Interactive scripts live in `scripts/` (run with e.g. `uv run scripts/visual_fdtd.py`):

- `visual_fdtd.py` runs a scene and visualises a slice of the 3d space.
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
