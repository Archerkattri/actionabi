#pragma once

#include <vector>

#include "actionabi/contract.hpp"
#include "actionabi/score.hpp"
#include "actionabi/trajectory.hpp"

namespace actionabi {

struct CudaTiming {
  double kernel_milliseconds{};
};

std::vector<ScoreResult> score_cuda_batch(
    const TrajectoryDataset& dataset,
    const std::vector<ActionContract>& contracts,
    const ScoreOptions& options,
    CudaTiming* timing = nullptr);

}  // namespace actionabi
