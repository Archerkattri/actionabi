// Per-cell and per-hypothesis evidence scoring for the ActionShift belief loop.
//
// This is the C++ scoring core repurposed as an *identification backend* for
// ActionShift's factorized-grammar belief. It computes exactly the per-cell
// Gaussian log-evidence that ``adaptation/factorized_grammar.py`` computes in
// torch, and the pooled per-hypothesis log-likelihood that
// ``adaptation/response.py`` computes -- but as a single fused pass with no
// intermediate tensors.
//
// Lag semantics are the FIXED single-step-delayed convention established in
// ``reports/scorer_fixes.md`` (defect 1): a command that arrives ``lag`` steps
// late explains the one-step transition at the delayed index, i.e. the observed
// response at step ``t`` is driven by the raw action at step ``t - lag``. In the
// per-step belief formulation the drive value for raw channel ``j`` is
//
//   delta:    base_j = history[lag][j]                      (raw_{t-lag})
//   absolute: base_j = history[lag][j] - history[lag+1][j]  (raw_{t-lag}
//                                                            - raw_{t-lag-1})
//
// which is the same alignment the CPU/CUDA trajectory scorers use after the fix
// (delayed_prev/delayed_next at row+lag -> row+lag+1). The per-cell predicted
// semantic value fed to channel ``i`` by raw channel ``j`` under sign ``s`` and
// scale ``k`` is ``alpha[i] * base_j * s * k`` and the contribution is the
// Gaussian log-kernel ``-0.5 * ((observed_i - predicted) / sigma_i)^2``.

#pragma once

#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace actionabi::cells {

// Factorized per-cell Gaussian evidence.
//
// Layout (all row-major, contiguous):
//   history:      (depth, batch, channels)
//   observed:     (batch, channels)
//   alpha, sigma: (channels)
//   signs:        (num_signs)
//   scales:       (num_scales)
//   mode_target:  (num_modes)  0 = delta, 1 = absolute
//   mode_lag:     (num_modes)
//   out:          (num_modes, batch, channels[i], channels[j], num_signs,
//                  num_scales)
//
// out[m,b,i,j,s,k] = -0.5 * ((observed[b,i]
//                    - alpha[i] * base_m[b,j] * signs[s] * scales[k]) / sigma[i])^2
template <typename Scalar>
void score_cells(const Scalar* history, [[maybe_unused]] std::size_t depth,
                 const Scalar* observed, std::size_t batch, std::size_t channels,
                 const Scalar* alpha, const Scalar* sigma, const Scalar* signs,
                 std::size_t num_signs, const Scalar* scales, std::size_t num_scales,
                 const std::int32_t* mode_target, const std::int32_t* mode_lag,
                 std::size_t num_modes, Scalar* out, unsigned int num_threads = 1) {
  const std::size_t sk = num_signs * num_scales;
  const std::size_t per_i = channels * sk;         // (j, s, k)
  const std::size_t per_b = channels * per_i;      // (i, j, s, k)
  const std::size_t per_m = batch * per_b;         // (b, i, j, s, k)

  // Precompute the coefficient grid c[s,k] = signs[s] * scales[k].
  std::vector<Scalar> coeff(sk);
  for (std::size_t s = 0; s < num_signs; ++s) {
    for (std::size_t k = 0; k < num_scales; ++k) {
      coeff[s * num_scales + k] = signs[s] * scales[k];
    }
  }

  auto score_batch_range = [&](std::size_t b_begin, std::size_t b_end) {
    std::vector<Scalar> base(channels);
    for (std::size_t m = 0; m < num_modes; ++m) {
      const std::size_t lag = static_cast<std::size_t>(mode_lag[m]);
      const bool absolute = mode_target[m] == 1;
      const Scalar* lag_slice = history + lag * batch * channels;
      const Scalar* lag_prev = history + (lag + 1) * batch * channels;
      Scalar* out_m = out + m * per_m;
      for (std::size_t b = b_begin; b < b_end; ++b) {
        // base drive value per raw channel j.
        const Scalar* lag_b = lag_slice + b * channels;
        const Scalar* prev_b = lag_prev + b * channels;
        for (std::size_t j = 0; j < channels; ++j) {
          base[j] = absolute ? (lag_b[j] - prev_b[j]) : lag_b[j];
        }
        const Scalar* obs_b = observed + b * channels;
        Scalar* out_mb = out_m + b * per_b;
        for (std::size_t i = 0; i < channels; ++i) {
          const Scalar obs_i = obs_b[i];
          const Scalar alpha_i = alpha[i];
          const Scalar inv_sigma_i = Scalar(1) / sigma[i];
          Scalar* out_mbi = out_mb + i * per_i;
          for (std::size_t j = 0; j < channels; ++j) {
            const Scalar drive = alpha_i * base[j];
            Scalar* out_mbij = out_mbi + j * sk;
            for (std::size_t c = 0; c < sk; ++c) {
              const Scalar residual = obs_i - drive * coeff[c];
              const Scalar standardized = residual * inv_sigma_i;
              out_mbij[c] = Scalar(-0.5) * standardized * standardized;
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
  const unsigned int workers =
      static_cast<unsigned int>(std::min<std::size_t>(num_threads, batch));
  std::vector<std::thread> pool;
  pool.reserve(workers);
  const std::size_t chunk = (batch + workers - 1) / workers;
  for (unsigned int w = 0; w < workers; ++w) {
    const std::size_t b_begin = static_cast<std::size_t>(w) * chunk;
    const std::size_t b_end = std::min(b_begin + chunk, batch);
    if (b_begin >= b_end) {
      break;
    }
    pool.emplace_back(score_batch_range, b_begin, b_end);
  }
  for (auto& thread : pool) {
    thread.join();
  }
}

// Pooled per-hypothesis Gaussian log-likelihood (matches
// ``ResponseModel.log_likelihood``).
//
// Layout:
//   predicted:    (hypotheses, batch, channels)
//   observed:     (batch, channels)
//   alpha, sigma: (channels)
//   out:          (batch, hypotheses)
//
// out[b,h] = sum_i -0.5 * ((observed[b,i] - alpha[i] * predicted[h,b,i]) / sigma[i])^2
template <typename Scalar>
void score_hypotheses(const Scalar* predicted, std::size_t hypotheses,
                      std::size_t batch, std::size_t channels,
                      const Scalar* observed, const Scalar* alpha,
                      const Scalar* sigma, Scalar* out,
                      unsigned int num_threads = 1) {
  auto score_batch_range = [&](std::size_t b_begin, std::size_t b_end) {
    for (std::size_t h = 0; h < hypotheses; ++h) {
      for (std::size_t b = b_begin; b < b_end; ++b) {
        const Scalar* pred = predicted + (h * batch + b) * channels;
        const Scalar* obs = observed + b * channels;
        Scalar accumulator = Scalar(0);
        for (std::size_t i = 0; i < channels; ++i) {
          const Scalar residual = obs[i] - alpha[i] * pred[i];
          const Scalar standardized = residual / sigma[i];
          accumulator += Scalar(-0.5) * standardized * standardized;
        }
        out[b * hypotheses + h] = accumulator;
      }
    }
  };

  if (num_threads <= 1 || batch <= 1) {
    score_batch_range(0, batch);
    return;
  }
  const unsigned int workers =
      static_cast<unsigned int>(std::min<std::size_t>(num_threads, batch));
  std::vector<std::thread> pool;
  pool.reserve(workers);
  const std::size_t chunk = (batch + workers - 1) / workers;
  for (unsigned int w = 0; w < workers; ++w) {
    const std::size_t b_begin = static_cast<std::size_t>(w) * chunk;
    const std::size_t b_end = std::min(b_begin + chunk, batch);
    if (b_begin >= b_end) {
      break;
    }
    pool.emplace_back(score_batch_range, b_begin, b_end);
  }
  for (auto& thread : pool) {
    thread.join();
  }
}

}  // namespace actionabi::cells
