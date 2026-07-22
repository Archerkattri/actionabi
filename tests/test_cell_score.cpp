// Unit tests for the ActionShift-facing evidence-scoring core (bindings/).
//
// These guard the fused per-cell and per-hypothesis kernels that back the
// pybind11 identification backend: the Gaussian log-kernel value, the FIXED
// single-step-delayed lag alignment (delta vs absolute base), and the
// equivalence of the multi-threaded path with the serial path.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

#include "../bindings/cell_score.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

double reference_cell(double obs, double alpha, double base, double sign, double scale,
                      double sigma) {
  const double residual = obs - alpha * base * sign * scale;
  const double standardized = residual / sigma;
  return -0.5 * standardized * standardized;
}

}  // namespace

TEST_CASE("score_cells computes the Gaussian log-kernel per cell", "[cells]") {
  // depth=3, batch=1, channels=2, signs={-1,1}, scales={0.5,2.0}, one delta mode
  // at lag 0. history[0] = current raw, history[1] = one step ago.
  const std::size_t depth = 3, batch = 1, channels = 2, num_signs = 2, num_scales = 2;
  std::vector<double> history = {
      // depth 0 (current): b0 -> [0.3, -0.7]
      0.3, -0.7,
      // depth 1: [0.1, 0.4]
      0.1, 0.4,
      // depth 2: [0.0, 0.0]
      0.0, 0.0,
  };
  std::vector<double> observed = {0.25, 0.10};
  std::vector<double> alpha = {1.0, 0.8};
  std::vector<double> sigma = {0.5, 0.2};
  std::vector<double> signs = {-1.0, 1.0};
  std::vector<double> scales = {0.5, 2.0};
  std::vector<std::int32_t> mode_target = {0};  // delta
  std::vector<std::int32_t> mode_lag = {0};

  std::vector<double> out(1 * batch * channels * channels * num_signs * num_scales);
  actionabi::cells::score_cells<double>(
      history.data(), depth, observed.data(), batch, channels, alpha.data(),
      sigma.data(), signs.data(), num_signs, scales.data(), num_scales,
      mode_target.data(), mode_lag.data(), mode_target.size(), out.data());

  // delta base at lag 0 == history[0] (current raw).
  const std::vector<double> base = {0.3, -0.7};
  const std::size_t sk = num_signs * num_scales;
  for (std::size_t i = 0; i < channels; ++i) {
    for (std::size_t j = 0; j < channels; ++j) {
      for (std::size_t s = 0; s < num_signs; ++s) {
        for (std::size_t k = 0; k < num_scales; ++k) {
          const std::size_t idx =
              ((i * channels + j) * num_signs + s) * num_scales + k;
          const double expected = reference_cell(observed[i], alpha[i], base[j],
                                                  signs[s], scales[k], sigma[i]);
          CHECK_THAT(out[idx], WithinRel(expected, 1e-12) || WithinAbs(expected, 1e-14));
        }
      }
    }
  }
  (void)sk;
}

TEST_CASE("absolute mode uses the fixed single-step-delayed delta base", "[cells][lag]") {
  // A lagged absolute contract's drive is raw_{t-lag} - raw_{t-lag-1}. With
  // lag=1 that is history[1] - history[2].
  const std::size_t depth = 4, batch = 1, channels = 2, num_signs = 1, num_scales = 1;
  std::vector<double> history = {
      0.9, 0.9,   // depth 0 (current)
      0.4, -0.2,  // depth 1 (raw_{t-1})
      0.1, 0.5,   // depth 2 (raw_{t-2})
      0.0, 0.0,   // depth 3
  };
  std::vector<double> observed = {0.30, -0.70};
  std::vector<double> alpha = {1.0, 1.0};
  std::vector<double> sigma = {1.0, 1.0};
  std::vector<double> signs = {1.0};
  std::vector<double> scales = {1.0};
  std::vector<std::int32_t> mode_target = {1};  // absolute
  std::vector<std::int32_t> mode_lag = {1};

  std::vector<double> out(channels * channels);
  actionabi::cells::score_cells<double>(
      history.data(), depth, observed.data(), batch, channels, alpha.data(),
      sigma.data(), signs.data(), num_signs, scales.data(), num_scales,
      mode_target.data(), mode_lag.data(), mode_target.size(), out.data());

  // base_j = history[1][j] - history[2][j] = {0.3, -0.7}
  const std::vector<double> base = {0.4 - 0.1, -0.2 - 0.5};
  for (std::size_t i = 0; i < channels; ++i) {
    for (std::size_t j = 0; j < channels; ++j) {
      const double expected = reference_cell(observed[i], 1.0, base[j], 1.0, 1.0, 1.0);
      CHECK_THAT(out[i * channels + j], WithinAbs(expected, 1e-14));
    }
  }
  // Cell (i=j) with base == observed should be exactly zero loss on this trace.
  CHECK_THAT(out[0 * channels + 0], WithinAbs(0.0, 1e-14));
  CHECK_THAT(out[1 * channels + 1], WithinAbs(0.0, 1e-14));
}

TEST_CASE("score_hypotheses matches the pooled Gaussian log-likelihood", "[pool]") {
  const std::size_t hypotheses = 2, batch = 2, channels = 3;
  std::vector<double> predicted = {
      // hypothesis 0: b0, b1
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
      // hypothesis 1: b0, b1
      -0.1, 0.0, 0.2, 0.3, -0.4, 0.1,
  };
  std::vector<double> observed = {0.15, 0.25, 0.20, 0.35, 0.45, 0.55};
  std::vector<double> alpha = {1.0, 0.9, 1.1};
  std::vector<double> sigma = {0.3, 0.4, 0.5};
  std::vector<double> out(batch * hypotheses);
  actionabi::cells::score_hypotheses<double>(predicted.data(), hypotheses, batch,
                                             channels, observed.data(), alpha.data(),
                                             sigma.data(), out.data());
  for (std::size_t h = 0; h < hypotheses; ++h) {
    for (std::size_t b = 0; b < batch; ++b) {
      double expected = 0.0;
      for (std::size_t i = 0; i < channels; ++i) {
        const double pred = predicted[(h * batch + b) * channels + i];
        const double residual = observed[b * channels + i] - alpha[i] * pred;
        const double standardized = residual / sigma[i];
        expected += -0.5 * standardized * standardized;
      }
      CHECK_THAT(out[b * hypotheses + h], WithinRel(expected, 1e-12));
    }
  }
}

TEST_CASE("multi-threaded scoring equals the serial path", "[cells][threads]") {
  const std::size_t depth = 3, batch = 37, channels = 6, num_signs = 2, num_scales = 7;
  std::vector<double> history(depth * batch * channels);
  std::vector<double> observed(batch * channels);
  for (std::size_t n = 0; n < history.size(); ++n) {
    history[n] = std::sin(0.1 * static_cast<double>(n));
  }
  for (std::size_t n = 0; n < observed.size(); ++n) {
    observed[n] = std::cos(0.07 * static_cast<double>(n));
  }
  std::vector<double> alpha(channels, 1.0), sigma(channels, 0.5);
  std::vector<double> signs = {-1.0, 1.0};
  std::vector<double> scales = {0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0};
  std::vector<std::int32_t> mode_target = {0, 1};
  std::vector<std::int32_t> mode_lag = {0, 1};
  const std::size_t size =
      mode_target.size() * batch * channels * channels * num_signs * num_scales;
  std::vector<double> serial(size), parallel(size);
  actionabi::cells::score_cells<double>(
      history.data(), depth, observed.data(), batch, channels, alpha.data(),
      sigma.data(), signs.data(), num_signs, scales.data(), num_scales,
      mode_target.data(), mode_lag.data(), mode_target.size(), serial.data(), 1);
  actionabi::cells::score_cells<double>(
      history.data(), depth, observed.data(), batch, channels, alpha.data(),
      sigma.data(), signs.data(), num_signs, scales.data(), num_scales,
      mode_target.data(), mode_lag.data(), mode_target.size(), parallel.data(), 4);
  for (std::size_t n = 0; n < size; ++n) {
    REQUIRE(serial[n] == parallel[n]);
  }
}
