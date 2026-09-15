// Python bindings for the CPU evidence-scoring backend used by ActionShift.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "cell_score.hpp"

namespace py = pybind11;

template <typename Scalar>
using Array = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;

#ifdef ACTIONABI_CELLS_CUDA
void score_cells_cuda_host(const float* history, std::size_t depth,
                           const float* observed, std::size_t batch,
                           std::size_t channels, const float* alpha,
                           const float* sigma, const float* signs,
                           std::size_t num_signs, const float* scales,
                           std::size_t num_scales, const std::int32_t* mode_target,
                           const std::int32_t* mode_lag, std::size_t num_modes,
                           float* out);
#endif

template <typename Scalar>
py::array_t<Scalar> score_cells_impl(
    Array<Scalar> history, Array<Scalar> observed, Array<Scalar> alpha,
    Array<Scalar> sigma, Array<Scalar> signs, Array<Scalar> scales,
    IntArray mode_target, IntArray mode_lag, unsigned int num_threads) {
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
  const std::size_t num_modes = static_cast<std::size_t>(mode_target.size());
  if (static_cast<std::size_t>(mode_lag.size()) != num_modes) {
    throw std::invalid_argument("mode_target and mode_lag must have equal length");
  }
  for (std::size_t mode = 0; mode < num_modes; ++mode) {
    if (mode_lag.at(mode) < 0 ||
        static_cast<std::size_t>(mode_lag.at(mode)) + 1 >= depth) {
      throw std::invalid_argument("history depth too small for requested lag");
    }
    if (mode_target.at(mode) != 0 && mode_target.at(mode) != 1) {
      throw std::invalid_argument("mode_target must be 0 (delta) or 1 (absolute)");
    }
  }

  py::array_t<Scalar> output({num_modes, batch, channels, channels,
                              static_cast<std::size_t>(signs.size()),
                              static_cast<std::size_t>(scales.size())});
  {
    py::gil_scoped_release release;
    actionabi::cells::score_cells(
        history.data(), depth, observed.data(), batch, channels, alpha.data(),
        sigma.data(), signs.data(), static_cast<std::size_t>(signs.size()),
        scales.data(), static_cast<std::size_t>(scales.size()), mode_target.data(),
        mode_lag.data(), num_modes, output.mutable_data(), num_threads);
  }
  return output;
}

template <typename Scalar>
py::array_t<Scalar> score_hypotheses_impl(
    Array<Scalar> predicted, Array<Scalar> observed, Array<Scalar> alpha,
    Array<Scalar> sigma, unsigned int num_threads) {
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
  py::array_t<Scalar> output({batch, hypotheses});
  {
    py::gil_scoped_release release;
    actionabi::cells::score_hypotheses(
        predicted.data(), hypotheses, batch, channels, observed.data(), alpha.data(),
        sigma.data(), output.mutable_data(), num_threads);
  }
  return output;
}

#ifdef ACTIONABI_CELLS_CUDA
py::array_t<float> score_cells_cuda_impl(
    Array<float> history, Array<float> observed, Array<float> alpha,
    Array<float> sigma, Array<float> signs, Array<float> scales,
    IntArray mode_target, IntArray mode_lag) {
  if (history.ndim() != 3 || observed.ndim() != 2) {
    throw std::invalid_argument("invalid history/observed rank");
  }
  const std::size_t depth = static_cast<std::size_t>(history.shape(0));
  const std::size_t batch = static_cast<std::size_t>(history.shape(1));
  const std::size_t channels = static_cast<std::size_t>(history.shape(2));
  const std::size_t num_modes = static_cast<std::size_t>(mode_target.size());
  if (static_cast<std::size_t>(observed.shape(0)) != batch ||
      static_cast<std::size_t>(observed.shape(1)) != channels ||
      static_cast<std::size_t>(alpha.size()) != channels ||
      static_cast<std::size_t>(sigma.size()) != channels ||
      static_cast<std::size_t>(mode_lag.size()) != num_modes) {
    throw std::invalid_argument("CUDA scorer input shapes do not agree");
  }
  for (std::size_t mode = 0; mode < num_modes; ++mode) {
    if (mode_lag.at(mode) < 0 ||
        static_cast<std::size_t>(mode_lag.at(mode)) + 1 >= depth) {
      throw std::invalid_argument("history depth too small for requested lag");
    }
  }
  py::array_t<float> output({num_modes, batch, channels, channels,
                             static_cast<std::size_t>(signs.size()),
                             static_cast<std::size_t>(scales.size())});
  {
    py::gil_scoped_release release;
    score_cells_cuda_host(
        history.data(), depth, observed.data(), batch, channels, alpha.data(),
        sigma.data(), signs.data(), static_cast<std::size_t>(signs.size()),
        scales.data(), static_cast<std::size_t>(scales.size()), mode_target.data(),
        mode_lag.data(), num_modes, output.mutable_data());
  }
  return output;
}
#endif

PYBIND11_MODULE(actionabi_cells, module) {
  module.doc() = "ActionABI CPU evidence-scoring core for ActionShift.";
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
  module.def("score_cells_cuda", &score_cells_cuda_impl,
             py::arg("history"), py::arg("observed"), py::arg("alpha"),
             py::arg("sigma"), py::arg("signs"), py::arg("scales"),
             py::arg("mode_target"), py::arg("mode_lag"));
#else
  module.attr("has_cuda") = false;
#endif
}
