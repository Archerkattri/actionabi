#pragma once

#include <cstdint>
#include <vector>

#include "actionabi/contract.hpp"
#include "actionabi/trajectory.hpp"

namespace actionabi {

struct ScoreOptions {
  double huber_delta{1.0};
  double train_episode_fraction{0.7};
};

struct ScoreResult {
  double train_loss;
  double heldout_loss;
  std::uint64_t residual_count;
  std::vector<double> per_dimension_loss;
};

ScoreResult score_cpu(const TrajectoryDataset& dataset,
                      const ActionContract& contract,
                      const ScoreOptions& options);

}  // namespace actionabi
