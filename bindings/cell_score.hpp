// Per-cell and per-hypothesis evidence scoring for the ActionShift belief loop.
//
// This header is intentionally dependency-free so the CPU scoring path can be
// tested and embedded without pybind11 or CUDA.  Lag semantics follow the
// repository's fixed single-step-delayed convention:
//   delta:    base = history[lag]
//   absolute: base = history[lag] - history[lag + 1]

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace actionabi::cells {

template <typename Scalar>
void score_cells(const Scalar* history, [[maybe_unused]] std::size_t depth,
                 const Scalar* observed, std::size_t batch, std::size_t channels,
                 const Scalar* alpha, const Scalar* sigma, const Scalar* signs,
                 std::size_t num_signs, const Scalar* scales, std::size_t num_scales,
                 const std::int32_t* mode_target, const std::int32_t* mode_lag,
                 std::size_t num_modes, Scalar* out, unsigned int num_threads = 1) {
  const std::size_t sign_scale_count = num_signs * num_scales;
  const std::size_t per_input_channel = channels * sign_scale_count;
  const std::size_t per_batch = channels * per_input_channel;
  const std::size_t per_mode = batch * per_batch;

  std::vector<Scalar> coefficients(sign_scale_count);
  for (std::size_t sign = 0; sign < num_signs; ++sign) {
    for (std::size_t scale = 0; scale < num_scales; ++scale) {
      coefficients[sign * num_scales + scale] = signs[sign] * scales[scale];
    }
  }

  auto score_batch_range = [&](std::size_t begin, std::size_t end) {
    std::vector<Scalar> base(channels);
    for (std::size_t mode = 0; mode < num_modes; ++mode) {
      const std::size_t lag = static_cast<std::size_t>(mode_lag[mode]);
      const bool absolute = mode_target[mode] == 1;
      const Scalar* lag_slice = history + lag * batch * channels;
      const Scalar* previous_slice = history + (lag + 1) * batch * channels;
      Scalar* mode_output = out + mode * per_mode;

      for (std::size_t batch_index = begin; batch_index < end; ++batch_index) {
        const Scalar* lag_row = lag_slice + batch_index * channels;
        const Scalar* previous_row = previous_slice + batch_index * channels;
        for (std::size_t input = 0; input < channels; ++input) {
          base[input] = absolute ? lag_row[input] - previous_row[input]
                                 : lag_row[input];
        }

        const Scalar* observation = observed + batch_index * channels;
        Scalar* batch_output = mode_output + batch_index * per_batch;
        for (std::size_t output = 0; output < channels; ++output) {
          const Scalar observation_value = observation[output];
          const Scalar alpha_value = alpha[output];
          const Scalar inverse_sigma = Scalar(1) / sigma[output];
          Scalar* output_row = batch_output + output * per_input_channel;
          for (std::size_t input = 0; input < channels; ++input) {
            const Scalar drive = alpha_value * base[input];
            Scalar* cell = output_row + input * sign_scale_count;
            for (std::size_t coefficient = 0; coefficient < sign_scale_count;
                 ++coefficient) {
              const Scalar residual = observation_value - drive * coefficients[coefficient];
              const Scalar standardized = residual * inverse_sigma;
              cell[coefficient] = Scalar(-0.5) * standardized * standardized;
            }
          }
        }
      }
    }
  };

  if (num_threads <= 1 || batch <= 1) {
    score_batch_range(0, batch);
    return;
  }

  const unsigned int workers = static_cast<unsigned int>(
      std::min<std::size_t>(static_cast<std::size_t>(num_threads), batch));
  std::vector<std::thread> pool;
  pool.reserve(workers);
  const std::size_t chunk = (batch + workers - 1) / workers;
  for (unsigned int worker = 0; worker < workers; ++worker) {
    const std::size_t begin = static_cast<std::size_t>(worker) * chunk;
    const std::size_t end = std::min(begin + chunk, batch);
    if (begin >= end) {
      break;
    }
    pool.emplace_back(score_batch_range, begin, end);
  }
  for (auto& worker : pool) {
    worker.join();
  }
}

template <typename Scalar>
void score_hypotheses(const Scalar* predicted, std::size_t hypotheses,
                      std::size_t batch, std::size_t channels,
                      const Scalar* observed, const Scalar* alpha,
                      const Scalar* sigma, Scalar* out,
                      unsigned int num_threads = 1) {
  auto score_batch_range = [&](std::size_t begin, std::size_t end) {
    for (std::size_t hypothesis = 0; hypothesis < hypotheses; ++hypothesis) {
      for (std::size_t batch_index = begin; batch_index < end; ++batch_index) {
        const Scalar* prediction =
            predicted + (hypothesis * batch + batch_index) * channels;
        const Scalar* observation = observed + batch_index * channels;
        Scalar score = Scalar(0);
        for (std::size_t channel = 0; channel < channels; ++channel) {
          const Scalar residual =
              observation[channel] - alpha[channel] * prediction[channel];
          const Scalar standardized = residual / sigma[channel];
          score += Scalar(-0.5) * standardized * standardized;
        }
        out[batch_index * hypotheses + hypothesis] = score;
      }
    }
  };

  if (num_threads <= 1 || batch <= 1) {
    score_batch_range(0, batch);
    return;
  }

  const unsigned int workers = static_cast<unsigned int>(
      std::min<std::size_t>(static_cast<std::size_t>(num_threads), batch));
  std::vector<std::thread> pool;
  pool.reserve(workers);
  const std::size_t chunk = (batch + workers - 1) / workers;
  for (unsigned int worker = 0; worker < workers; ++worker) {
    const std::size_t begin = static_cast<std::size_t>(worker) * chunk;
    const std::size_t end = std::min(begin + chunk, batch);
    if (begin >= end) {
      break;
    }
    pool.emplace_back(score_batch_range, begin, end);
  }
  for (auto& worker : pool) {
    worker.join();
  }
}

}  // namespace actionabi::cells
