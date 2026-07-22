// CUDA path for the ActionShift per-cell evidence scorer.
//
// Transfer-inclusive by construction: the host entry point uploads every input,
// launches one thread per output cell, and downloads the result. Reported latency
// is therefore end-to-end (H2D + kernel + D2H), matching ActionABI's honest
// transfer-inclusive benchmarking discipline. The per-cell arithmetic is
// bit-for-bit the same formula as the CPU kernel (float32), using the FIXED
// single-step-delayed lag alignment.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

template <typename T>
struct DeviceBuffer {
  T* data{nullptr};
  explicit DeviceBuffer(std::size_t count) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)),
               "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template <typename T>
void upload(T* device, const T* host, std::size_t count) {
  check_cuda(cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice),
             "cudaMemcpy H2D");
}

__global__ void score_cells_kernel(
    const float* history, const float* observed, const float* alpha,
    const float* sigma, const float* coeff, const std::int32_t* mode_target,
    const std::int32_t* mode_lag, std::size_t num_modes, std::size_t batch,
    std::size_t channels, std::size_t num_signs, std::size_t num_scales,
    float* out) {
  const std::size_t sk = num_signs * num_scales;
  const std::size_t per_j = sk;
  const std::size_t per_i = channels * per_j;
  const std::size_t per_b = channels * per_i;
  const std::size_t per_m = batch * per_b;
  const std::size_t total = num_modes * per_m;
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  std::size_t rem = idx;
  const std::size_t m = rem / per_m;
  rem %= per_m;
  const std::size_t b = rem / per_b;
  rem %= per_b;
  const std::size_t i = rem / per_i;
  rem %= per_i;
  const std::size_t j = rem / per_j;
  rem %= per_j;
  const std::size_t c = rem;  // flattened (s, k)

  const std::size_t lag = static_cast<std::size_t>(mode_lag[m]);
  const bool absolute = mode_target[m] == 1;
  const std::size_t lag_index = (lag * batch + b) * channels + j;
  float base = history[lag_index];
  if (absolute) {
    base -= history[((lag + 1) * batch + b) * channels + j];
  }
  const float obs = observed[b * channels + i];
  const float residual = obs - alpha[i] * base * coeff[c];
  const float standardized = residual / sigma[i];
  out[idx] = -0.5f * standardized * standardized;
}

}  // namespace

void score_cells_cuda_host(const float* history, std::size_t depth,
                           const float* observed, std::size_t batch,
                           std::size_t channels, const float* alpha,
                           const float* sigma, const float* signs,
                           std::size_t num_signs, const float* scales,
                           std::size_t num_scales, const std::int32_t* mode_target,
                           const std::int32_t* mode_lag, std::size_t num_modes,
                           float* out) {
  const std::size_t sk = num_signs * num_scales;
  std::vector<float> coeff(sk);
  for (std::size_t s = 0; s < num_signs; ++s) {
    for (std::size_t k = 0; k < num_scales; ++k) {
      coeff[s * num_scales + k] = signs[s] * scales[k];
    }
  }
  const std::size_t history_count = depth * batch * channels;
  const std::size_t out_count = num_modes * batch * channels * channels * sk;

  DeviceBuffer<float> d_history(history_count);
  DeviceBuffer<float> d_observed(batch * channels);
  DeviceBuffer<float> d_alpha(channels);
  DeviceBuffer<float> d_sigma(channels);
  DeviceBuffer<float> d_coeff(sk);
  DeviceBuffer<std::int32_t> d_target(num_modes);
  DeviceBuffer<std::int32_t> d_lag(num_modes);
  DeviceBuffer<float> d_out(out_count);

  upload(d_history.data, history, history_count);
  upload(d_observed.data, observed, batch * channels);
  upload(d_alpha.data, alpha, channels);
  upload(d_sigma.data, sigma, channels);
  upload(d_coeff.data, coeff.data(), sk);
  upload(d_target.data, mode_target, num_modes);
  upload(d_lag.data, mode_lag, num_modes);

  constexpr std::size_t threads = 256;
  const std::size_t blocks = (out_count + threads - 1) / threads;
  score_cells_kernel<<<static_cast<unsigned int>(blocks),
                       static_cast<unsigned int>(threads)>>>(
      d_history.data, d_observed.data, d_alpha.data, d_sigma.data, d_coeff.data,
      d_target.data, d_lag.data, num_modes, batch, channels, num_signs,
      num_scales, d_out.data);
  check_cuda(cudaGetLastError(), "score_cells_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "score_cells_kernel sync");
  check_cuda(cudaMemcpy(out, d_out.data, out_count * sizeof(float),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy D2H");
}
