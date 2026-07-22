#include "actionabi/score.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace actionabi {
namespace {

double huber(double residual, double delta) {
  const double magnitude = std::abs(residual);
  if (magnitude <= delta) {
    return 0.5 * residual * residual;
  }
  return delta * (magnitude - 0.5 * delta);
}

std::vector<double> decode(std::span<const double> raw,
                           const ActionContract& contract) {
  std::vector<double> semantic(contract.permutation.size());
  for (std::size_t component = 0; component < semantic.size(); ++component) {
    semantic[component] =
        raw[contract.permutation[component]] * contract.sign[component] *
        contract.scale[component];
  }
  if (contract.gripper_inverted) {
    semantic.back() *= -1.0;
  }
  return semantic;
}

struct LossAccumulator {
  double train_sum{0.0};
  double heldout_sum{0.0};
  std::uint64_t train_count{0};
  std::uint64_t heldout_count{0};
  std::vector<double> heldout_dimension_sum;
  std::vector<std::uint64_t> heldout_dimension_count;
};

}  // namespace

ScoreResult score_cpu(const TrajectoryDataset& dataset,
                      const ActionContract& contract,
                      const ScoreOptions& options) {
  validate(contract);
  const auto dimension = contract.permutation.size();
  if (dimension != dataset.state_dimension() ||
      dimension != dataset.action_dimension()) {
    throw std::invalid_argument(
        "contract dimension must match state and action dimensions");
  }
  if (!std::isfinite(options.huber_delta) || options.huber_delta <= 0.0) {
    throw std::invalid_argument("huber_delta must be finite and positive");
  }
  if (!std::isfinite(options.train_episode_fraction) ||
      options.train_episode_fraction <= 0.0 ||
      options.train_episode_fraction >= 1.0) {
    throw std::invalid_argument("train_episode_fraction must be between zero and one");
  }
  if (dataset.episodes().size() < 2) {
    throw std::invalid_argument("held-out scoring requires at least two episodes");
  }
  const auto train_episodes = std::clamp<std::size_t>(
      static_cast<std::size_t>(std::floor(
          options.train_episode_fraction * dataset.episodes().size())),
      1, dataset.episodes().size() - 1);
  LossAccumulator accumulator;
  accumulator.heldout_dimension_sum.assign(dimension, 0.0);
  accumulator.heldout_dimension_count.assign(dimension, 0);
  const std::size_t offset = static_cast<std::size_t>(contract.lag_steps) + 1;

  for (std::size_t episode_index = 0;
       episode_index < dataset.episodes().size(); ++episode_index) {
    const auto& episode = dataset.episodes()[episode_index];
    if (episode.end - episode.begin <= offset) {
      continue;
    }
    const bool training = episode_index < train_episodes;
    for (std::size_t row = episode.begin; row + offset < episode.end; ++row) {
      const auto command = decode(dataset.action(row), contract);
      // Physical response model: a lagged command produces a SINGLE-STEP delayed
      // response, ``executed_t = decode(raw_{t-lag})``. The observable that a
      // command at ``row`` explains is therefore the one-step transition at the
      // lagged position ``row + lag`` -> ``row + lag + 1`` (== row+offset-1 ->
      // row+offset), NOT the multi-step span ``row -> row+offset``. For a delta
      // (and velocity) target we take the consecutive one-step delta at the
      // delayed index; for an absolute target the observable is the delayed
      // absolute state ``row+offset`` (unchanged). This matches the Python
      // reference scorer (experiments/run_falsification.py::score_hypothesis) for
      // every lag, and reduces to the old behaviour at lag == 0 (offset == 1).
      const auto delayed_prev = dataset.state(row + offset - 1);
      const auto delayed_next = dataset.state(row + offset);
      const auto elapsed_ns =
          dataset.timestamps_ns()[row + offset] -
          dataset.timestamps_ns()[row + offset - 1];
      if (elapsed_ns <= 0) {
        throw std::invalid_argument("trajectory contains nonpositive elapsed time");
      }
      const double elapsed_seconds = static_cast<double>(elapsed_ns) * 1e-9;
      for (std::size_t component = 0; component < dimension; ++component) {
        double observable = 0.0;
        switch (contract.target) {
          case TargetKind::AbsolutePosition:
            observable = delayed_next[component];
            break;
          case TargetKind::DeltaPosition:
            observable = delayed_next[component] - delayed_prev[component];
            break;
          case TargetKind::Velocity:
            observable = (delayed_next[component] - delayed_prev[component]) /
                         elapsed_seconds;
            break;
        }
        const double loss = huber(command[component] - observable,
                                  options.huber_delta);
        if (training) {
          accumulator.train_sum += loss;
          ++accumulator.train_count;
        } else {
          accumulator.heldout_sum += loss;
          ++accumulator.heldout_count;
          accumulator.heldout_dimension_sum[component] += loss;
          ++accumulator.heldout_dimension_count[component];
        }
      }
    }
  }
  if (accumulator.train_count == 0 || accumulator.heldout_count == 0) {
    throw std::invalid_argument(
        "contract lag leaves no train or held-out residuals");
  }
  std::vector<double> per_dimension(dimension);
  for (std::size_t component = 0; component < dimension; ++component) {
    per_dimension[component] =
        accumulator.heldout_dimension_sum[component] /
        static_cast<double>(accumulator.heldout_dimension_count[component]);
  }
  return {
      .train_loss = accumulator.train_sum /
                    static_cast<double>(accumulator.train_count),
      .heldout_loss = accumulator.heldout_sum /
                      static_cast<double>(accumulator.heldout_count),
      .residual_count = accumulator.train_count + accumulator.heldout_count,
      .per_dimension_loss = std::move(per_dimension),
  };
}

}  // namespace actionabi
