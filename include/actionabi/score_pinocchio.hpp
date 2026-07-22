#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "actionabi/contract.hpp"
#include "actionabi/score.hpp"
#include "actionabi/trajectory.hpp"

namespace actionabi {

struct KinematicScoreResult {
  bool supported{};
  std::optional<ScoreResult> score;
  std::string warning;
};

KinematicScoreResult score_cartesian_with_pinocchio(
    const TrajectoryDataset& dataset,
    const ActionContract& contract,
    const std::filesystem::path& urdf,
    const std::string& end_effector_frame,
    const ScoreOptions& options = {});

}  // namespace actionabi
