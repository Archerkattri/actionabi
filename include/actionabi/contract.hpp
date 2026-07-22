#pragma once

#include <cstdint>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace actionabi {

enum class TargetKind { AbsolutePosition, DeltaPosition, Velocity };
enum class ActionSpace { Joint, Cartesian };
enum class ReferenceFrame { Unspecified, World, Base, Tool };

struct ActionContract {
  TargetKind target;
  ActionSpace space;
  ReferenceFrame frame;
  std::vector<std::uint32_t> permutation;
  std::vector<std::int8_t> sign;
  std::vector<double> scale;
  std::int32_t lag_steps;
  bool gripper_inverted;

  bool operator==(const ActionContract&) const = default;
};

void validate(const ActionContract& contract);
nlohmann::json to_json(const ActionContract& contract);
ActionContract contract_from_json(const nlohmann::json& encoded);

}  // namespace actionabi
