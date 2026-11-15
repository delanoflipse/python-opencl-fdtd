"""
Frequency-domain (FDFD) solver: the steady-state field at a single frequency.

Instead of stepping the FDTD scheme in time, substitute a time-harmonic field
p^n = P exp(j w n dt) into the exact same discrete update (see
gpu/accelerated_fdtd.cl). With z = exp(j w dt) every non-wall cell gives

  interior:  (z - 2 + 1/z) P - L P = 0
  boundary:  (z - 2 + 1/z) P + beta_k (z - 1/z) P - L P = 0

  where L P = D1 (S1 - k1 P) + D2 (S2 - k2 P) + D3 (S3 - k3 P) is the compact
  scheme's discrete Laplacian (only existing neighbours), and
  beta_k = (26 - neighbours) * lambda * beta / 2 is the kernel's boundary loss.

So A(w) = diag(2 cos(w dt) - 2 + 2j beta_k sin(w dt)) - L, a complex symmetric
sparse matrix, and the field for a source is one sparse solve A P = b. Because
it is the same discretisation, a sine-driven FDTD run converges to this field
(including its numerical dispersion), only without the time stepping.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fdtd.grid import WALL_FLAG, SimulationGrid
from fdtd.parameters import SimulationParameters

# neighbour offsets (w, h, d) in the order of the bit flags of populate_neighbours
NEIGHBOUR_BIT_OFFSETS = (
    # D1, bits 0-5
    (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
    # D2, bits 6-17
    (-1, -1, 0), (-1, 1, 0), (1, 1, 0), (1, -1, 0),
    (0, -1, -1), (0, 1, -1), (0, 1, 1), (0, -1, 1),
    (-1, 0, -1), (-1, 0, 1), (1, 0, 1), (1, 0, -1),
    # D3, bits 18-25
    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
    (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
)
K1_BITMASK = (1 << 6) - 1
K2_BITMASK = ((1 << 18) - 1) & ~K1_BITMASK
K3_BITMASK = ((1 << 26) - 1) & ~((1 << 18) - 1)


def _popcount(values: np.ndarray) -> np.ndarray:
  counts = np.zeros(values.shape, dtype=np.int64)
  for bit in range(26):
    counts += (values >> bit) & 1
  return counts


@dataclass
class HelmholtzOperator:
  """The frequency independent parts of A(w) for one grid"""
  laplacian: sp.csr_matrix   # L over the non-wall cells
  beta_k: np.ndarray         # boundary loss per non-wall cell (0 for interior cells)
  cell_index: np.ndarray     # grid shaped, index into the unknowns or -1 for walls
  grid_shape: Tuple[int, int, int]
  dt: float

  @property
  def size(self) -> int:
    return self.laplacian.shape[0]

  def matrix(self, frequency: float) -> sp.csc_matrix:
    """A(w) for the given frequency in Hz"""
    w_dt = 2.0 * np.pi * frequency * self.dt
    diagonal = (2.0 * np.cos(w_dt) - 2.0) + 2j * self.beta_k * np.sin(w_dt)
    return (sp.diags(diagonal) - self.laplacian).tocsc()

  def to_grid(self, values: np.ndarray) -> np.ndarray:
    """Scatter a vector over the unknowns into a grid, walls become NaN"""
    dtype = np.complex128 if np.iscomplexobj(values) else np.float64
    field = np.full(self.grid_shape, np.nan, dtype=dtype)
    mask = self.cell_index >= 0
    field[mask] = values[self.cell_index[mask]]
    return field


def build_operator(grid: SimulationGrid, parameters: SimulationParameters) -> HelmholtzOperator:
  """Build L and beta_k from a built grid (scene.build() / rebuild() must have run)"""
  geometry = grid.geometry
  neighbours = grid.neighbours.astype(np.int64)
  active = (geometry & WALL_FLAG) == 0
  cell_index = np.full(grid.grid_shape, -1, dtype=np.int64)
  cell_index[active] = np.arange(int(active.sum()))
  size = int(active.sum())

  coefficients = [parameters.arg_d1] * 6 + [parameters.arg_d2] * 12 + [parameters.arg_d3] * 8
  rows, cols, values = [], [], []
  positions = np.argwhere(active)
  flags = neighbours[active]
  for bit, (offset, coefficient) in enumerate(zip(NEIGHBOUR_BIT_OFFSETS, coefficients)):
    if coefficient == 0.0:
      continue
    has_neighbour = ((flags >> bit) & 1) == 1
    source = positions[has_neighbour]
    target = source + np.array(offset)
    rows.append(cell_index[tuple(source.T)])
    cols.append(cell_index[tuple(target.T)])
    values.append(np.full(len(source), coefficient))

  rows = np.concatenate(rows)
  cols = np.concatenate(cols)
  values = np.concatenate(values)
  off_diagonal = sp.csr_matrix((values, (rows, cols)), shape=(size, size))
  laplacian = off_diagonal - sp.diags(np.asarray(off_diagonal.sum(axis=1)).ravel())

  # boundary loss, exactly as the kernel: cells missing any used neighbour
  interior_mask = (K1_BITMASK if parameters.arg_d1 != 0.0 else 0) | \
      (K2_BITMASK if parameters.arg_d2 != 0.0 else 0) | \
      (K3_BITMASK if parameters.arg_d3 != 0.0 else 0)
  is_boundary = (flags & interior_mask) != interior_mask
  beta = grid.beta[active]
  beta_k = np.where(is_boundary,
                    (26 - _popcount(flags)) * parameters.lambda_courant * beta * 0.5, 0.0)

  return HelmholtzOperator(laplacian.tocsr(), beta_k, cell_index, grid.grid_shape, parameters.dt)


def solve_vector(operator: HelmholtzOperator, frequency: float, source: Tuple[int, int, int],
                 hard_source: bool = False, amplitude: complex = 1.0, x0: Optional[np.ndarray] = None,
                 rtol: float = 1e-8) -> np.ndarray:
  """
  Steady-state complex pressure for a point source, as a vector over the unknowns.

  soft source (default): an injection term, like `next += signal`; the field is
    the discrete Green's function, so room resonances show up freely.
  hard source: the source cell is clamped to `amplitude`, like the FDTD kernel
    (USE_HYBRID_HARD_SOURCE), handy to compare against an FDTD run.

  Uses BiCGSTAB (about 1-2 s for ~50k cells), falling back to LGMRES and finally
  a direct solve. Pass the solution of a nearby frequency as x0 to warm-start.
  """
  matrix = operator.matrix(frequency)
  source_index = operator.cell_index[source]
  if source_index < 0:
    raise ValueError(f"source {source} is inside a wall")
  rhs = np.zeros(operator.size, dtype=np.complex128)
  if hard_source:
    # replace the source row by the identity: P_source = amplitude
    keep = np.ones(operator.size)
    keep[source_index] = 0.0
    matrix = (sp.diags(keep) @ matrix + sp.diags(1.0 - keep)).tocsc()
    rhs[source_index] = amplitude
  else:
    # the kernel adds the signal after the boundary factor: next = update / (1 + beta_k) + S
    rhs[source_index] = amplitude * (1.0 + operator.beta_k[source_index])

  solution, info = spla.bicgstab(matrix, rhs, x0=x0, rtol=rtol, maxiter=20000)
  if info != 0:
    solution, info = spla.lgmres(matrix, rhs, x0=solution, rtol=rtol, maxiter=5000)
  if info != 0:
    solution = spla.spsolve(matrix, rhs)
  return solution


def solve(operator: HelmholtzOperator, frequency: float, source: Tuple[int, int, int],
          hard_source: bool = False, amplitude: complex = 1.0) -> np.ndarray:
  """Steady-state complex pressure on the grid (NaN in walls), see solve_vector"""
  return operator.to_grid(solve_vector(operator, frequency, source, hard_source, amplitude))


def frequency_response(operator: HelmholtzOperator, frequencies: np.ndarray, source: Tuple[int, int, int],
                       receivers: List[Tuple[int, int, int]], hard_source: bool = False,
                       progress: Optional[Callable[[int, float], None]] = None) -> np.ndarray:
  """Complex pressure at each receiver for every frequency, shape (len(frequencies), len(receivers))"""
  receiver_index = [operator.cell_index[r] for r in receivers]
  result = np.zeros((len(frequencies), len(receivers)), dtype=np.complex128)
  previous = None
  for i, frequency in enumerate(frequencies):
    previous = solve_vector(operator, float(frequency), source, hard_source, x0=previous)
    result[i] = previous[receiver_index]
    if progress is not None:
      progress(i, float(frequency))
  return result
