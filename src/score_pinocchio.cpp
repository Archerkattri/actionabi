#include "actionabi/score_pinocchio.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

namespace actionabi {
namespace {

double huber(double residual, double delta) {
  const double magnitude = std::abs(residual);
  return magnitude <= delta ? 0.5 * residual * residual
                            : delta * (magnitude - 0.5 * delta);
}

KinematicScoreResult unsupported(std::string warning) {
  return {.supported = false, .score = std::nullopt, .warning = std::move(warning)};
}

Eigen::Vector3d frame_translation(const pinocchio::Model& model,
                                  pinocchio::Data& data,
                                  pinocchio::FrameIndex frame,
                                  std::span<const double> state) {
  Eigen::VectorXd configuration(model.nq);
  for (Eigen::Index index = 0; index < configuration.size(); ++index) {
    configuration[index] = state[static_cast<std::size_t>(index)];
  }
  pinocchio::forwardKinematics(model, data, configuration);
  pinocchio::updateFramePlacements(model, data);
  return data.oMf[frame].translation();
}

Eigen::Matrix3d frame_rotation(const pinocchio::Model& model,
                               pinocchio::Data& data,
                               pinocchio::FrameIndex frame,
                               std::span<const double> state) {
  Eigen::VectorXd configuration(model.nq);
  for (Eigen::Index index = 0; index < configuration.size(); ++index) {
    configuration[index] = state[static_cast<std::size_t>(index)];
  }
  pinocchio::forwardKinematics(model, data, configuration);
  pinocchio::updateFramePlacements(model, data);
  return data.oMf[frame].rotation();
}

}  // namespace

KinematicScoreResult score_cartesian_with_pinocchio(
    const TrajectoryDataset& dataset,
    const ActionContract& contract,
    const std::filesystem::path& urdf,
    const std::string& end_effector_frame,
    const ScoreOptions& options) {
  try {
    validate(contract);
  } catch (const std::exception& error) {
    return unsupported(std::string("invalid contract: ") + error.what());
  }
  if (contract.space != ActionSpace::Cartesian) {
    return unsupported("kinematic evidence only scores Cartesian hypotheses");
  }
  const auto dimension = contract.permutation.size();
  if ((dimension != 2 && dimension != 3) ||
      dataset.action_dimension() != dimension) {
    return unsupported("translation evidence requires a 2D or 3D Cartesian action");
  }
  if (contract.gripper_inverted) {
    return unsupported("gripper evidence is not represented by the kinematic model");
  }
  if (!std::isfinite(options.huber_delta) || options.huber_delta <= 0.0 ||
      !std::isfinite(options.train_episode_fraction) ||
      options.train_episode_fraction <= 0.0 ||
      options.train_episode_fraction >= 1.0 || dataset.episodes().size() < 2) {
    return unsupported("invalid scoring options or insufficient episodes");
  }

  pinocchio::Model model;
  try {
    pinocchio::urdf::buildModel(urdf.string(), model);
  } catch (const std::exception& error) {
    return unsupported(std::string("could not load URDF: ") + error.what());
  }
  if (model.nq != static_cast<Eigen::Index>(dataset.state_dimension())) {
    return unsupported("trajectory state does not contain every model configuration variable");
  }
  if (!model.existFrame(end_effector_frame)) {
    return unsupported("end-effector frame is missing from the URDF");
  }
  const auto frame = model.getFrameId(end_effector_frame);
  pinocchio::Data data(model);
  const auto train_episodes = std::clamp<std::size_t>(
      static_cast<std::size_t>(std::floor(
          options.train_episode_fraction * dataset.episodes().size())),
      1, dataset.episodes().size() - 1);
  const auto offset = static_cast<std::size_t>(contract.lag_steps) + 1;
  double train_sum = 0.0;
  double heldout_sum = 0.0;
  std::uint64_t train_count = 0;
  std::uint64_t heldout_count = 0;
  std::vector<double> dimension_sum(dimension, 0.0);
  std::vector<std::uint64_t> dimension_count(dimension, 0);

  for (std::size_t episode_index = 0;
       episode_index < dataset.episodes().size(); ++episode_index) {
    const auto& episode = dataset.episodes()[episode_index];
    for (std::size_t row = episode.begin; row + offset < episode.end; ++row) {
      const auto current = frame_translation(model, data, frame, dataset.state(row));
      const auto future =
          frame_translation(model, data, frame, dataset.state(row + offset));
      Eigen::Vector3d command = Eigen::Vector3d::Zero();
      const auto raw = dataset.action(row);
      for (std::size_t component = 0; component < dimension; ++component) {
        command[static_cast<Eigen::Index>(component)] =
            raw[contract.permutation[component]] * contract.sign[component] *
            contract.scale[component];
      }
      if (contract.frame == ReferenceFrame::Tool) {
        command = frame_rotation(model, data, frame, dataset.state(row)) * command;
      }
      const auto elapsed_ns =
          dataset.timestamps_ns()[row + offset] - dataset.timestamps_ns()[row];
      if (elapsed_ns <= 0) {
        return unsupported("trajectory contains a nonpositive elapsed time");
      }
      const double elapsed_seconds = static_cast<double>(elapsed_ns) * 1e-9;
      for (std::size_t component = 0; component < dimension; ++component) {
        double observable = future[static_cast<Eigen::Index>(component)];
        if (contract.target == TargetKind::DeltaPosition) {
          observable -= current[static_cast<Eigen::Index>(component)];
        } else if (contract.target == TargetKind::Velocity) {
          observable = (future[static_cast<Eigen::Index>(component)] -
                        current[static_cast<Eigen::Index>(component)]) /
                       elapsed_seconds;
        }
        const auto loss = huber(
            command[static_cast<Eigen::Index>(component)] - observable,
            options.huber_delta);
        if (episode_index < train_episodes) {
          train_sum += loss;
          ++train_count;
        } else {
          heldout_sum += loss;
          ++heldout_count;
          dimension_sum[component] += loss;
          ++dimension_count[component];
        }
      }
    }
  }
  if (train_count == 0 || heldout_count == 0) {
    return unsupported("contract lag leaves no train or held-out residuals");
  }
  std::vector<double> per_dimension(dimension);
  for (std::size_t component = 0; component < dimension; ++component) {
    per_dimension[component] =
        dimension_sum[component] / static_cast<double>(dimension_count[component]);
  }
  return {
      .supported = true,
      .score = ScoreResult{.train_loss = train_sum / static_cast<double>(train_count),
                           .heldout_loss = heldout_sum /
                                           static_cast<double>(heldout_count),
                           .residual_count = train_count + heldout_count,
                           .per_dimension_loss = std::move(per_dimension)},
      .warning = "",
  };
}

}  // namespace actionabi
