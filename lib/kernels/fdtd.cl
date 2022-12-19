#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define D1_SIZE 6
#define D2_SIZE 12
#define D3_SIZE 8

__constant unsigned char BitsSetTable256[256] = {
#define B2(n) n, n + 1, n + 1, n + 2
#define B4(n) B2(n), B2(n + 1), B2(n + 1), B2(n + 2)
#define B6(n) B4(n), B4(n + 1), B4(n + 1), B4(n + 2)
    B6(0), B6(1), B6(1), B6(2)};

bool in_range(uint size, uint index) { return index > 0 && index < size; }

__kernel void compact_step(__global double *previous_pressure,
                           __global double *pressure,
                           __global double *pressure_next,
                           __global char *geometry, __global char *neighbours,
                           uint size_w, uint size_h, uint size_d, double lambda,
                           double beta, double signal) {
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

  char neighbour_flag = neighbours[i];

  if (i >= size || i < 0) {
    return;
  }

  if (is_wall || neighbour_flag == 0) {
    pressure_next[i] = 0.0;
    // pressure_next[i] = 0.001;
    return;
  }

  double lambda2 = lambda * lambda;
  char neighbour_count = BitsSetTable256[neighbour_flag];
  double neighbour_factor = 2.0 - (double)(neighbour_count)*lambda2;

  // bool w_plus = in_range(size_w, w + 1);
  // bool w_min = in_range(size_w, w - 1);
  // bool h_plus = in_range(size_h, h + 1);
  // bool h_min = in_range(size_h, h - 1);
  // bool d_plus = in_range(size_d, d + 1);
  // bool d_min = in_range(size_d, d - 1);

  bool n_w_min = neighbour_flag >> 0 & 1;
  bool n_w_plus = neighbour_flag >> 1 & 1;
  bool n_h_min = neighbour_flag >> 2 & 1;
  bool n_h_plus = neighbour_flag >> 3 & 1;
  bool n_d_min = neighbour_flag >> 4 & 1;
  bool n_d_plus = neighbour_flag >> 5 & 1;

  double current = pressure[i];
  double previous = previous_pressure[i];
  double stencil_sum = 0.0;

  double beta_1_factor = 1.0;
  double beta_2_factor = 1.0;

  if (neighbour_count < 6) {
    beta_1_factor = 1.0 / (1.0 + lambda * beta);
    beta_2_factor = 1.0 - lambda * beta;
  }

  if (n_w_plus)
    stencil_sum += pressure[i + w_stride];

  if (n_w_min)
    stencil_sum += pressure[i - w_stride];

  if (n_h_plus)
    stencil_sum += pressure[i + h_stride];

  if (n_h_min)
    stencil_sum += pressure[i - h_stride];

  if (n_d_plus)
    stencil_sum += pressure[i + d_stride];

  if (n_d_min)
    stencil_sum += pressure[i - d_stride];

  double next_value =
      beta_1_factor * (neighbour_factor * current + lambda2 * stencil_sum -
                       beta_2_factor * previous);
  if (is_source) {
    next_value += signal;
  }

  pressure_next[i] = next_value;
}

__kernel void analysis_step(__global double *pressure, __global double *rms,
                            __global double *analysis, __global char *geometry,
                            uint size_w, uint size_h, uint size_d, double dt,
                            uint iteration) {
  size_t i = get_global_id(0);
  size_t w = (i / (size_h * size_d)) % size_w;
  size_t h = (i / (size_d)) % size_h;
  size_t d = i % size_d;

  size_t size = size_d * size_h * size_w;

  if (i >= size || i < 0) {
    return;
  }

  char geometry_type = geometry[i];

  if (geometry_type > 0) {
    return;
  }

  double current_pressure = pressure[i];
  double rms_sum = rms[i] + current_pressure * current_pressure;
  rms[i] = rms_sum;
  double iteration_factor = 1.0 / ((double)(iteration));
  // double rms_value = sqrt(iteration_factor * rms_sum);
  analysis[i] = iteration_factor * rms_sum;
  // analysis[i] = rms_value;
  // analysis[i] = 20.0 * log10(rms_value);
  // analysis[i] = 20.0 * log10(current_pressure);

  // double current_pressure = pressure[i];
  // analysis[i] += current_pressure * dt;

  // double current_max = analysis[i];
  // analysis[i] = max(fabs(current_pressure), current_max);
}
