# OpenCL accelerated Acoustic Finite Difference Simulation

This is project is part of my bachelor thesis for the Computer Science and Engineering degree of the TU Delft. For more information about the CSE3000 course, visit [https://github.com/TU-Delft-CSE/Research-Project](https://github.com/TU-Delft-CSE/Research-Project).

## Requirements

This repository was written for python version 3.9.x

These packages are required:

- numpy
- numba
- scipy
- matplotlib
- pyopencl

To install with pip, use:

```bash
pip install numpy numba scipy pyopencl matplotlib
```

## Run experiment(s)

There are a number of scripts in the root folder that apply the simulation for different purposes. The names are not very consistant, but the goal was to name every script different so they could be easily run. The different scripts have these goals:

- `benchmark_fdtd.py` is a benchmark utility that times the performance of the current implementation. I used this to gauge the performance of my implementation.
- `visual_fdtd.py` run a scene and visualises a slice of the 3d space.
- `sweep_visual.py` runs a single frequency sweep to determine the frequency response over the listener region.
- `full_sweep_visual.py` performs a full sweep for all locations and frequencies.
- **Utility scripts:**
- `parse_output.py` is used to reformat the output of a full run.
- `display_scene.py` shows a voxelized 3d scene.
- **Headless scripts:**
- `cli_benchmark.py` and `cli_run.py` are used to run a headless experiment with different parameters.

### Run on the DPHC

The script in `cli_run.py` can be used with `jobs/run_script.sh` to run on the DHPC. To run the script:

```
sbatch ./job/run_script.sh cli_run.py [...]
```
