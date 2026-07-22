// pybind11 module exposing ActionABI's evidence-scoring core to ActionShift.
//
// Two entry points, each in float32 and float64 variants (the caller dispatches
// on the tensor dtype so parity is exact per precision):
//
//   score_cells_f32 / score_cells_f64
//       Factorized per-cell Gaussian evidence for the full-grammar belief.
//   score_hypotheses_f32 / score_hypotheses_f64
//       Pooled per-hypothesis log-likelihood for the nine-contract pool belief.
//
// Optional CUDA variants (compiled when ACTIONABI_CELLS_CUDA is defined) take the
// same host numpy arrays and run the scoring on the GPU, transfer-inclusive, so
// the reported latency is end-to-end and honest.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "cell_score.hpp"

namespace py = pybind11;

#ifdef ACTIONABI_CELLS_CUDA
// Defined in cell_score_cuda.cu (external linkage, file scope).
void score_cells_cuda_host(const float* history, std::size_t depth,
                           const float* observed, std::size_t batch,
                           std::size_t channels, const float* alpha,
                           const float* sigma, const float* signs,
                           std::size_t num_signs, const float* scales,
                           std::size_t num_scales, const std::int32_t* mode_target,
                           const std::int32_t* mode_lag, std::size_t num_modes,
                           float* out);
#endif

namespace {

template <typename Scalar>
using Array = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;

template <typename Scalar>
py::array_t<Scalar> score_cells_impl(Array<Scalar> history, Array<Scalar> observed,
                                     Array<Scalar> alpha, Array<Scalar> sigma,
                                     Array<Scalar> signs, Array<Scalar> scales,
                                     IntArray mode_target, IntArray mode_lag,
                                     unsigned int num_threads) {
  if (history.ndim() != 3) {
    throw std::invalid_argument("history must be (depth, batch, channels)");
  }
  if (observed.ndim() != 2) {
    throw std::invalid_argument("observed must be (batch, channels)");
  }
  const std::size_t depth = static_cast<std::size_t>(history.shape(0));
  const std::size_t batch = static_cast<std::size_t>(history.shape(1));
  const std::size_t channels = static_cast<std::size_t>(history.shape(2));
  if (static_cast<std::size_t>(observed.shape(0)) != batch ||
      static_cast<std::size_t>(observed.shape(1)) != channels) {
    throw std::invalid_argument("observed shape must match history batch/channels");
  }
  if (static_cast<std::size_t>(alpha.size()) != channels ||
      static_cast<std::size_t>(sigma.size()) != channels) {
    throw std::invalid_argument("alpha and sigma must have one value per channel");
  }
  const std::size_t num_signs = static_cast<std::size_t>(signs.size());
  const std::size_t num_scales = static_cast<std::size_t>(scales.size());
  const std::size_t num_modes = static_cast<std::size_t>(mode_target.size());
  if (static_cast<std::size_t>(mode_lag.size()) != num_modes) {
    throw std::invalid_argument("mode_target and mode_lag must have equal length");
  }
  for (std::size_t m = 0; m < num_modes; ++m) {
    const std::size_t lag = static_cast<std::size_t>(mode_lag.at(m));
    if (mode_lag.at(m) < 0 || lag + 1 >= depth) {
      throw std::invalid_argument("history depth too small for requested lag");
    }
  }

  py::array_t<Scalar> out({num_modes, batch, channels, channels, num_signs, num_scales});
  {
    py::gil_scoped_release release;
    actionabi::cells::score_cells<Scalar>(
        history.data(), depth, observed.data(), batch, channels, alpha.data(),
        sigma.data(), signs.data(), num_signs, scales.data(), num_scales,
        mode_target.data(), mode_lag.data(), num_modes, out.mutable_data(),
        num_threads);
  }
  return out;
}

template <typename Scalar>
py::array_t<Scalar> score_hypotheses_impl(Array<Scalar> predicted,
                                          Array<Scalar> observed, Array<Scalar> alpha,
                                          Array<Scalar> sigma,
                                          unsigned int num_threads) {
  if (predicted.ndim() != 3) {
    throw std::invalid_argument("predicted must be (hypotheses, batch, channels)");
  }
  if (observed.ndim() != 2) {
    throw std::invalid_argument("observed must be (batch, channels)");
  }
  const std::size_t hypotheses = static_cast<std::size_t>(predicted.shape(0));
  const std::size_t batch = static_cast<std::size_t>(predicted.shape(1));
  const std::size_t channels = static_cast<std::size_t>(predicted.shape(2));
  if (static_cast<std::size_t>(observed.shape(0)) != batch ||
      static_cast<std::size_t>(observed.shape(1)) != channels) {
    throw std::invalid_argument("observed shape must match predicted batch/channels");
  }
  if (static_cast<std::size_t>(alpha.size()) != channels ||
      static_cast<std::size_t>(sigma.size()) != channels) {
    throw std::invalid_argument("alpha and sigma must have one value per channel");
  }
  py::array_t<Scalar> out({batch, hypotheses});
  {
    py::gil_scoped_release release;
    actionabi::cells::score_hypotheses<Scalar>(
        predicted.data(), hypotheses, batch, channels, observed.data(),
        alpha.data(), sigma.data(), out.mutable_data(), num_threads);
  }
  return out;
}

#ifdef ACTIONABI_CELLS_CUDA
py::array_t<float> score_cells_cuda_impl(Array<float> history, Array<float> observed,
                                         Array<float> alpha, Array<float> sigma,
                                         Array<float> signs, Array<float> scales,
                                         IntArray mode_target, IntArray mode_lag) {
  const std::size_t depth = static_cast<std::size_t>(history.shape(0));
  const std::size_t batch = static_cast<std::size_t>(history.shape(1));
  const std::size_t channels = static_cast<std::size_t>(history.shape(2));
  const std::size_t num_signs = static_cast<std::size_t>(signs.size());
  const std::size_t num_scales = static_cast<std::size_t>(scales.size());
  const std::size_t num_modes = static_cast<std::size_t>(mode_target.size());
  py::array_t<float> out({num_modes, batch, channels, channels, num_signs, num_scales});
  {
    py::gil_scoped_release release;
    score_cells_cuda_host(history.data(), depth, observed.data(), batch, channels,
                          alpha.data(), sigma.data(), signs.data(), num_signs,
                          scales.data(), num_scales, mode_target.data(),
                          mode_lag.data(), num_modes, out.mutable_data());
  }
  return out;
}
#endif

}  // namespace

PYBIND11_MODULE(actionabi_cells, module) {
  module.doc() =
      "ActionABI evidence-scoring core exposed for ActionShift belief updates "
      "(fixed single-step-delayed lag semantics).";

  module.def("score_cells_f32", &score_cells_impl<float>, py::arg("history"),
             py::arg("observed"), py::arg("alpha"), py::arg("sigma"),
             py::arg("signs"), py::arg("scales"), py::arg("mode_target"),
             py::arg("mode_lag"), py::arg("num_threads") = 1U);
  module.def("score_cells_f64", &score_cells_impl<double>, py::arg("history"),
             py::arg("observed"), py::arg("alpha"), py::arg("sigma"),
             py::arg("signs"), py::arg("scales"), py::arg("mode_target"),
             py::arg("mode_lag"), py::arg("num_threads") = 1U);
  module.def("score_hypotheses_f32", &score_hypotheses_impl<float>,
             py::arg("predicted"), py::arg("observed"), py::arg("alpha"),
             py::arg("sigma"), py::arg("num_threads") = 1U);
  module.def("score_hypotheses_f64", &score_hypotheses_impl<double>,
             py::arg("predicted"), py::arg("observed"), py::arg("alpha"),
             py::arg("sigma"), py::arg("num_threads") = 1U);

#ifdef ACTIONABI_CELLS_CUDA
  module.attr("has_cuda") = true;
  module.def("score_cells_cuda", &score_cells_cuda_impl, py::arg("history"),
             py::arg("observed"), py::arg("alpha"), py::arg("sigma"),
             py::arg("signs"), py::arg("scales"), py::arg("mode_target"),
             py::arg("mode_lag"));
#else
  module.attr("has_cuda") = false;
#endif
}
