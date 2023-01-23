#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define USE_HYBRID_HARD_SOURCE true
// #define ALPHA_TIMING 0.05
#define ALPHA_TIMING 0.01

__constant uint K1_BITMASK = (1 << 6) - 1;
__constant uint K2_BITMASK = (1 << 18) - 1 - K1_BITMASK;
__constant uint K3_BITMASK = (1 << 26) - 1 - K1_BITMASK - K2_BITMASK;
__constant uint K_FULL = (1 << 26) - 1;

bool in_range(uint size, uint index) { return index > 0 && index < size; }

__kernel void compact_step(__global double *previous_pressure,
                           __global double *pressure,
                           __global double *pressure_next,
                           __global double *betas, __global char *geometry,
                           __global uint *neighbours, uint size_w, uint size_h,
                           uint size_d, double lambda, double signal) {
  size_t i = get_global_id(0);
  size_t w = (i / (size_h * size_d)) % size_w;
  size_t h = (i / (size_d)) % size_h;
  size_t d = i % size_d;

  size_t d_stride = 1;
  size_t h_stride = size_d;
  size_t w_stride = size_d * size_h;
  size_t size = size_d * size_h * size_w;
  char geometry_type = geometry[i];

  bool is_wall = geometry_type & 1;
  bool is_source = geometry_type >> 1 & 1;

  uint neighbour_flag = neighbours[i];

  if (i >= size || i < 0) {
    return;
  }

  if (is_wall) {
    pressure_next[i] = NAN;
    // pressure_next[i] = 0.001;
    return;
  }

  if (neighbour_flag == 0) {
    pressure_next[i] = 0.0;
    return;
  }

  double lambda2 = lambda * lambda;
  uint neighbour_count = popcount(neighbour_flag & K1_BITMASK);
  double neighbour_factor = 2.0 - (double)(neighbour_count)*lambda2;

  double current = pressure[i];
  double previous = previous_pressure[i];

  double beta_1_factor = 1.0;
  double beta_2_factor = 1.0;

  if (neighbour_count < 6) {
    double beta = betas[i];
    beta_1_factor = 1.0 / (1.0 + lambda * beta);
    beta_2_factor = 1.0 - lambda * beta;
  }

  double stencil_sum = 0.0;
  if (neighbour_flag >> 0 & 1) stencil_sum += pressure[i - w_stride];
  if (neighbour_flag >> 1 & 1) stencil_sum += pressure[i + w_stride];
  if (neighbour_flag >> 2 & 1) stencil_sum += pressure[i - h_stride];
  if (neighbour_flag >> 3 & 1) stencil_sum += pressure[i + h_stride];
  if (neighbour_flag >> 4 & 1) stencil_sum += pressure[i - d_stride];
  if (neighbour_flag >> 5 & 1) stencil_sum += pressure[i + d_stride];

  double next_value = beta_1_factor * (neighbour_factor * current + lambda2 * stencil_sum -
                       beta_2_factor * previous);

  if (is_source && !isnan(signal)) {
    if (USE_HYBRID_HARD_SOURCE) {
      next_value = signal;
    } else {
      next_value += signal;
    }
  }

  pressure_next[i] = next_value;
}

__kernel void compact_schema_step(__global double *previous_pressure,
                           __global double *pressure,
                           __global double *pressure_next,
                           __global double *betas, __global char *geometry,
                           __global uint *neighbours, uint size_w, uint size_h,
                           uint size_d, double lambda, double pa, double pb, double d1, double d2, double d3, double d4, double signal) {
  size_t i = get_global_id(0);
  size_t w = (i / (size_h * size_d)) % size_w;
  size_t h = (i / (size_d)) % size_h;
  size_t d = i % size_d;

  size_t d_stride = 1;
  size_t h_stride = size_d;
  size_t w_stride = size_d * size_h;
  size_t size = size_d * size_h * size_w;
  char geometry_type = geometry[i];

  bool is_wall = geometry_type & 1;
  bool is_source = geometry_type >> 1 & 1;

  uint neighbour_flag = neighbours[i];

  if (i >= size || i < 0) {
    return;
  }

  if (is_wall) {
    pressure_next[i] = NAN;
    // pressure_next[i] = 0.001;
    return;
  }

  // if (neighbour_flag == 0) {
  //   pressure_next[i] = 0.0;
  //   return;
  // }

  uint k_neighbour_1 = popcount(neighbour_flag & K1_BITMASK);
  uint k_neighbour_2 = popcount(neighbour_flag & K2_BITMASK);
  uint k_neighbour_3 = popcount(neighbour_flag & K3_BITMASK);
  bool has_wall_neighbours = neighbour_flag != K_FULL;

  // TODO:
  double neighbour_factor = d4;
  double beta_1_factor = 1.0;
  double beta_2_factor = 1.0;
  double lambda2 = lambda * lambda;

  if (has_wall_neighbours) {
    neighbour_factor = 2 -(double)(k_neighbour_1) * lambda2
      + (double)(k_neighbour_2) * pa * lambda2
      - (double)(k_neighbour_3) * pb * lambda2;
    
    double beta = betas[i];
    beta_1_factor = 1.0 / (1.0 + lambda * beta);
    beta_2_factor = 1.0 - lambda * beta;
  }

  double current = pressure[i];
  double previous = previous_pressure[i];
  double d1_sum = 0.0;
  double d2_sum = 0.0;
  double d3_sum = 0.0;

  // D1 - 1x neighbours
  if (d1 != 0.0) {
    if (neighbour_flag >> 0 & 1) d1_sum += pressure[i - w_stride];
    if (neighbour_flag >> 1 & 1) d1_sum += pressure[i + w_stride];
    if (neighbour_flag >> 2 & 1) d1_sum += pressure[i - h_stride];
    if (neighbour_flag >> 3 & 1) d1_sum += pressure[i + h_stride];
    if (neighbour_flag >> 4 & 1) d1_sum += pressure[i - d_stride];
    if (neighbour_flag >> 5 & 1) d1_sum += pressure[i + d_stride];
  }

  // D2 - 2x neighbours
  if (d2 != 0.0) {
    if (neighbour_flag >> 6 & 1) d2_sum += pressure[i - w_stride - h_stride];
    if (neighbour_flag >> 7 & 1) d2_sum += pressure[i - w_stride + h_stride];
    if (neighbour_flag >> 8 & 1) d2_sum += pressure[i + w_stride + h_stride];
    if (neighbour_flag >> 9 & 1) d2_sum += pressure[i + w_stride - h_stride];
    if (neighbour_flag >> 10 & 1) d2_sum += pressure[i - d_stride - h_stride];
    if (neighbour_flag >> 11 & 1) d2_sum += pressure[i - d_stride + h_stride];
    if (neighbour_flag >> 12 & 1) d2_sum += pressure[i + d_stride + h_stride];
    if (neighbour_flag >> 13 & 1) d2_sum += pressure[i + d_stride - h_stride];
    if (neighbour_flag >> 14 & 1) d2_sum += pressure[i - w_stride - d_stride];
    if (neighbour_flag >> 15 & 1) d2_sum += pressure[i - w_stride + d_stride];
    if (neighbour_flag >> 16 & 1) d2_sum += pressure[i + w_stride + d_stride];
    if (neighbour_flag >> 17 & 1) d2_sum += pressure[i + w_stride - d_stride];
    // if (neighbour_flag >> 6 & 1) d2_sum += pressure[i + w_stride + h_stride];
    // if (neighbour_flag >> 7 & 1) d2_sum += pressure[i + w_stride - h_stride];
    // if (neighbour_flag >> 8 & 1) d2_sum += pressure[i - w_stride - h_stride];
    // if (neighbour_flag >> 9 & 1) d2_sum += pressure[i - w_stride + h_stride];
    // if (neighbour_flag >> 10 & 1) d2_sum += pressure[i + d_stride + h_stride];
    // if (neighbour_flag >> 11 & 1) d2_sum += pressure[i + d_stride - h_stride];
    // if (neighbour_flag >> 12 & 1) d2_sum += pressure[i - d_stride - h_stride];
    // if (neighbour_flag >> 13 & 1) d2_sum += pressure[i - d_stride + h_stride];
    // if (neighbour_flag >> 14 & 1) d2_sum += pressure[i + w_stride + d_stride];
    // if (neighbour_flag >> 15 & 1) d2_sum += pressure[i + w_stride - d_stride];
    // if (neighbour_flag >> 16 & 1) d2_sum += pressure[i - w_stride - d_stride];
    // if (neighbour_flag >> 17 & 1) d2_sum += pressure[i - w_stride + d_stride];
  }

  // D3 - 3x neighbours
  if (d3 != 0.0) {
    // if (neighbour_flag >> 18 & 1) d3_sum += pressure[i + w_stride + h_stride + d_stride];
    // if (neighbour_flag >> 19 & 1) d3_sum += pressure[i + w_stride + h_stride - d_stride];
    // if (neighbour_flag >> 20 & 1) d3_sum += pressure[i + w_stride - h_stride + d_stride];
    // if (neighbour_flag >> 21 & 1) d3_sum += pressure[i + w_stride - h_stride - d_stride];
    // if (neighbour_flag >> 22 & 1) d3_sum += pressure[i - w_stride + h_stride + d_stride];
    // if (neighbour_flag >> 23 & 1) d3_sum += pressure[i - w_stride + h_stride - d_stride];
    // if (neighbour_flag >> 24 & 1) d3_sum += pressure[i - w_stride - h_stride + d_stride];
    // if (neighbour_flag >> 25 & 1) d3_sum += pressure[i - w_stride - h_stride - d_stride];
    
    if (neighbour_flag >> 18 & 1) d3_sum += pressure[i - w_stride - h_stride - d_stride];
    if (neighbour_flag >> 19 & 1) d3_sum += pressure[i - w_stride - h_stride + d_stride];
    if (neighbour_flag >> 20 & 1) d3_sum += pressure[i - w_stride + h_stride - d_stride];
    if (neighbour_flag >> 21 & 1) d3_sum += pressure[i - w_stride + h_stride + d_stride];
    if (neighbour_flag >> 22 & 1) d3_sum += pressure[i + w_stride - h_stride - d_stride];
    if (neighbour_flag >> 23 & 1) d3_sum += pressure[i + w_stride - h_stride + d_stride];
    if (neighbour_flag >> 24 & 1) d3_sum += pressure[i + w_stride + h_stride - d_stride];
    if (neighbour_flag >> 25 & 1) d3_sum += pressure[i + w_stride + h_stride + d_stride];
  }

  double stencil_sum = d1 * d1_sum + d2 * d2_sum + d3 * d3_sum;
  double current_sum = neighbour_factor * current;
  double next_value = beta_1_factor * (stencil_sum + current_sum - beta_2_factor * previous);
  // double next_value = d1_sum;

  if (is_source && !isnan(signal)) {
    if (USE_HYBRID_HARD_SOURCE) {
      next_value = signal;
    } else {
      next_value += signal;
    }
  }

  pressure_next[i] = next_value;
}

__kernel void analysis_step(__global double *pressure,
                            __global double *analysis, __global char *geometry,
                            uint size_w, uint size_h, uint size_d, uint size_a,
                            double rho, double dt, double time_elapsed) {
  size_t i = get_global_id(0);
  size_t w = (i / (size_h * size_d)) % size_w;
  size_t h = (i / (size_d)) % size_h;
  size_t d = i % size_d;

  size_t size = size_d * size_h * size_w * size_a;

  if (i >= size || i < 0) {
    return;
  }

  // size_t pres_i = i * size_a + 0;
  size_t rms_i = i * size_a + 1;
  size_t leq_i = i * size_a + 2;
  // size_t ewma_i = i * size_a + 3;
  // size_t ewma_db_i = i * size_a + 4;

  char geometry_type = geometry[i];

  bool is_wall = geometry_type & 1;
  bool is_source = geometry_type >> 1 & 1;
  bool is_listener = geometry_type >> 3 & 1;
  double alpha = dt / ALPHA_TIMING;

  if (is_wall) {
    // if (!is_listener) {
    // analysis[ewma_db_i] = NAN;
    analysis[leq_i] = NAN;
    return;
  }

  double current_pressure = pressure[i];
  // double previous_pressure = pressure_previous[i];
  // double delta_pressure = current_pressure - previous_pressure;
  // double actual_pressure = rho * delta_pressure;
  double actual_pressure = current_pressure;
    // TODO: enable again if needed
  // analysis[pres_i] = actual_pressure;

  double rms_addition = actual_pressure * actual_pressure * 25e8;

  double rms_sum = analysis[rms_i] + dt * rms_addition;
  analysis[rms_i] = rms_sum;

  if (time_elapsed > 0) {
    double iteration_factor = 1.0 / time_elapsed;

    // note: log sqrt x = 0.5 * log x
    double rms_value = iteration_factor * rms_sum;
    analysis[leq_i] = rms_value > 0 ? 10.0 * log10(rms_value) : 0;

    // TODO: enable again if needed
    // double current_ewma = analysis[ewma_i];
    // double ewma = alpha * rms_addition + (1 - alpha) * current_ewma;
    // analysis[ewma_i] = ewma;
    // analysis[ewma_db_i] = ewma > 0 ? 10.0 * log10(ewma) : 0;
  }
}
