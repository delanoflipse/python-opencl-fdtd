#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Build options, set by kernel_program.py:
//   REAL                float | double: precision of the pressure and beta buffers
//   D1, D2, D3          scheme coefficients (D4 = 2 - 6 D1 - 12 D2 - 8 D3 is
//                       implied by the Laplacian form used below)
//   LAMBDA              courant number
//   USE_D1/D2/D3        0 | 1: whether that neighbour set contributes (coefficient != 0)
#ifndef REAL
#define REAL double
#endif

#define USE_HYBRID_HARD_SOURCE 1

#define C_D1 ((REAL)(D1))
#define C_D2 ((REAL)(D2))
#define C_D3 ((REAL)(D3))
#define C_LAMBDA ((REAL)(LAMBDA))

#define K1_BITMASK ((1u << 6) - 1u)
#define K2_BITMASK (((1u << 18) - 1u) & ~K1_BITMASK)
#define K3_BITMASK (((1u << 26) - 1u) & ~((1u << 18) - 1u))
// a cell with all of these neighbours present is an interior cell
#define INTERIOR_MASK                                                          \
  ((USE_D1 ? K1_BITMASK : 0u) | (USE_D2 ? K2_BITMASK : 0u) |                   \
   (USE_D3 ? K3_BITMASK : 0u))

// One time step of the compact explicit scheme, plus the analysis
// accumulation: pressure_squared_sum[i] += pressure[i]^2.
// The squared sum is scaled and converted to RMS / Leq on the host.
//
// The update is written in Laplacian form,
//   next = 2 p - p_prev + D1 (S1 - k1 p) + D2 (S2 - k2 p) + D3 (S3 - k3 p),
// which equals D1 S1 + D2 S2 + D3 S3 + D4 p - p_prev but stays exact for a
// uniform field when the coefficients are rounded (e.g. D1 = 1/3 in float).
__kernel void fdtd_step(__global const REAL *restrict previous_pressure,
                        __global const REAL *restrict pressure,
                        __global REAL *restrict pressure_next,
                        __global const REAL *restrict betas,
                        __global const uchar *restrict geometry,
                        __global const uint *restrict neighbours,
                        __global double *restrict pressure_squared_sum,
                        uint size, uint h_stride, uint w_stride, REAL signal) {
  const uint i = get_global_id(0);
  const uint d_stride = 1;

  if (i >= size) {
    return;
  }

  const uchar geometry_type = geometry[i];
  const bool is_wall = geometry_type & 1;
  const bool is_source = geometry_type >> 1 & 1;

  if (is_wall) {
    pressure_next[i] = NAN;
    return;
  }

  const uint neighbour_flag = neighbours[i];
  const REAL current = pressure[i];
  const REAL previous = previous_pressure[i];

  pressure_squared_sum[i] += (double)current * (double)current;

  REAL next_value;

  if ((neighbour_flag & INTERIOR_MASK) == INTERIOR_MASK) {
    // interior: every neighbour exists, no boundary loss
    next_value = 2 * current - previous;
#if USE_D1
    next_value += C_D1 * (pressure[i - w_stride] + pressure[i + w_stride] +
                          pressure[i - h_stride] + pressure[i + h_stride] +
                          pressure[i - d_stride] + pressure[i + d_stride] -
                          6 * current);
#endif
#if USE_D2
    next_value += C_D2 * (pressure[i - w_stride - h_stride] +
                           pressure[i - w_stride + h_stride] +
                           pressure[i + w_stride + h_stride] +
                           pressure[i + w_stride - h_stride] +
                           pressure[i - d_stride - h_stride] +
                           pressure[i - d_stride + h_stride] +
                           pressure[i + d_stride + h_stride] +
                           pressure[i + d_stride - h_stride] +
                           pressure[i - w_stride - d_stride] +
                           pressure[i - w_stride + d_stride] +
                           pressure[i + w_stride + d_stride] +
                           pressure[i + w_stride - d_stride] -
                           12 * current);
#endif
#if USE_D3
    next_value += C_D3 * (pressure[i - w_stride - h_stride - d_stride] +
                           pressure[i - w_stride - h_stride + d_stride] +
                           pressure[i - w_stride + h_stride - d_stride] +
                           pressure[i - w_stride + h_stride + d_stride] +
                           pressure[i + w_stride - h_stride - d_stride] +
                           pressure[i + w_stride - h_stride + d_stride] +
                           pressure[i + w_stride + h_stride - d_stride] +
                           pressure[i + w_stride + h_stride + d_stride] -
                           8 * current);
#endif
  } else {
    // boundary: only existing neighbours contribute, with boundary loss
    const REAL beta = betas[i];
    const REAL beta_k = (26 - (REAL)popcount(neighbour_flag)) * C_LAMBDA *
                        beta * (REAL)0.5;
    const REAL beta_2_factor = 1 - beta_k;
    const REAL beta_1_factor = 1 / (1 + beta_k);

    REAL d1_sum = 0;
    REAL d2_sum = 0;
    REAL d3_sum = 0;

#if USE_D1
    if (neighbour_flag >> 0 & 1)
      d1_sum += pressure[i - w_stride];
    if (neighbour_flag >> 1 & 1)
      d1_sum += pressure[i + w_stride];
    if (neighbour_flag >> 2 & 1)
      d1_sum += pressure[i - h_stride];
    if (neighbour_flag >> 3 & 1)
      d1_sum += pressure[i + h_stride];
    if (neighbour_flag >> 4 & 1)
      d1_sum += pressure[i - d_stride];
    if (neighbour_flag >> 5 & 1)
      d1_sum += pressure[i + d_stride];
#endif

#if USE_D2
    if (neighbour_flag >> 6 & 1)
      d2_sum += pressure[i - w_stride - h_stride];
    if (neighbour_flag >> 7 & 1)
      d2_sum += pressure[i - w_stride + h_stride];
    if (neighbour_flag >> 8 & 1)
      d2_sum += pressure[i + w_stride + h_stride];
    if (neighbour_flag >> 9 & 1)
      d2_sum += pressure[i + w_stride - h_stride];
    if (neighbour_flag >> 10 & 1)
      d2_sum += pressure[i - d_stride - h_stride];
    if (neighbour_flag >> 11 & 1)
      d2_sum += pressure[i - d_stride + h_stride];
    if (neighbour_flag >> 12 & 1)
      d2_sum += pressure[i + d_stride + h_stride];
    if (neighbour_flag >> 13 & 1)
      d2_sum += pressure[i + d_stride - h_stride];
    if (neighbour_flag >> 14 & 1)
      d2_sum += pressure[i - w_stride - d_stride];
    if (neighbour_flag >> 15 & 1)
      d2_sum += pressure[i - w_stride + d_stride];
    if (neighbour_flag >> 16 & 1)
      d2_sum += pressure[i + w_stride + d_stride];
    if (neighbour_flag >> 17 & 1)
      d2_sum += pressure[i + w_stride - d_stride];
#endif

#if USE_D3
    if (neighbour_flag >> 18 & 1)
      d3_sum += pressure[i - w_stride - h_stride - d_stride];
    if (neighbour_flag >> 19 & 1)
      d3_sum += pressure[i - w_stride - h_stride + d_stride];
    if (neighbour_flag >> 20 & 1)
      d3_sum += pressure[i - w_stride + h_stride - d_stride];
    if (neighbour_flag >> 21 & 1)
      d3_sum += pressure[i - w_stride + h_stride + d_stride];
    if (neighbour_flag >> 22 & 1)
      d3_sum += pressure[i + w_stride - h_stride - d_stride];
    if (neighbour_flag >> 23 & 1)
      d3_sum += pressure[i + w_stride - h_stride + d_stride];
    if (neighbour_flag >> 24 & 1)
      d3_sum += pressure[i + w_stride + h_stride - d_stride];
    if (neighbour_flag >> 25 & 1)
      d3_sum += pressure[i + w_stride + h_stride + d_stride];
#endif

    REAL update = 2 * current - beta_2_factor * previous;
#if USE_D1
    update += C_D1 * (d1_sum - (REAL)popcount(neighbour_flag & K1_BITMASK) * current);
#endif
#if USE_D2
    update += C_D2 * (d2_sum - (REAL)popcount(neighbour_flag & K2_BITMASK) * current);
#endif
#if USE_D3
    update += C_D3 * (d3_sum - (REAL)popcount(neighbour_flag & K3_BITMASK) * current);
#endif
    next_value = beta_1_factor * update;
  }

  if (is_source && !isnan(signal)) {
    if (USE_HYBRID_HARD_SOURCE) {
      next_value = signal;
    } else {
      next_value += signal;
    }
  }

  pressure_next[i] = next_value;
}
