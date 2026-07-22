#include "actionabi/contract.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace actionabi {
namespace {

std::string_view target_name(TargetKind target) {
  switch (target) {
    case TargetKind::AbsolutePosition:
      return "absolute_position";
    case TargetKind::DeltaPosition:
      return "delta_position";
    case TargetKind::Velocity:
      return "velocity";
  }
  throw std::invalid_argument("invalid target enum");
}

std::string_view space_name(ActionSpace space) {
  switch (space) {
    case ActionSpace::Joint:
      return "joint";
    case ActionSpace::Cartesian:
      return "cartesian";
  }
  throw std::invalid_argument("invalid action-space enum");
}

std::string_view frame_name(ReferenceFrame frame) {
  switch (frame) {
    case ReferenceFrame::Unspecified:
      return "unspecified";
    case ReferenceFrame::World:
      return "world";
    case ReferenceFrame::Base:
      return "base";
    case ReferenceFrame::Tool:
      return "tool";
  }
  throw std::invalid_argument("invalid reference-frame enum");
}

TargetKind parse_target(const std::string& value) {
  if (value == "absolute_position") {
    return TargetKind::AbsolutePosition;
  }
  if (value == "delta_position") {
    return TargetKind::DeltaPosition;
  }
  if (value == "velocity") {
    return TargetKind::Velocity;
  }
  throw std::invalid_argument("unknown target: " + value);
}

ActionSpace parse_space(const std::string& value) {
  if (value == "joint") {
    return ActionSpace::Joint;
  }
  if (value == "cartesian") {
    return ActionSpace::Cartesian;
  }
  throw std::invalid_argument("unknown action space: " + value);
}

ReferenceFrame parse_frame(const std::string& value) {
  if (value == "unspecified") {
    return ReferenceFrame::Unspecified;
  }
  if (value == "world") {
    return ReferenceFrame::World;
  }
  if (value == "base") {
    return ReferenceFrame::Base;
  }
  if (value == "tool") {
    return ReferenceFrame::Tool;
  }
  throw std::invalid_argument("unknown reference frame: " + value);
}

}  // namespace

void validate(const ActionContract& contract) {
  const auto dimension = contract.permutation.size();
  if (dimension == 0) {
    throw std::invalid_argument("contract must contain at least one component");
  }
  if (contract.sign.size() != dimension || contract.scale.size() != dimension) {
    throw std::invalid_argument("contract component dimensions must match");
  }
  auto permutation = contract.permutation;
  std::sort(permutation.begin(), permutation.end());
  for (std::size_t index = 0; index < dimension; ++index) {
    if (permutation[index] != index) {
      throw std::invalid_argument("permutation must be a bijection");
    }
  }
  if (std::any_of(contract.sign.begin(), contract.sign.end(),
                  [](std::int8_t value) { return value != -1 && value != 1; })) {
    throw std::invalid_argument("sign values must be -1 or 1");
  }
  if (std::any_of(contract.scale.begin(), contract.scale.end(),
                  [](double value) { return !std::isfinite(value) || value <= 0.0; })) {
    throw std::invalid_argument("scale values must be finite and positive");
  }
  if (contract.lag_steps < 0) {
    throw std::invalid_argument("lag_steps must be nonnegative");
  }
  if (contract.space == ActionSpace::Joint &&
      contract.frame != ReferenceFrame::Unspecified) {
    throw std::invalid_argument(
        "joint actions require an unspecified reference frame");
  }
  if (contract.space == ActionSpace::Cartesian &&
      contract.frame == ReferenceFrame::Unspecified) {
    throw std::invalid_argument(
        "cartesian actions require a world, base, or tool reference frame");
  }
}

nlohmann::json to_json(const ActionContract& contract) {
  validate(contract);
  std::vector<int> signs;
  signs.reserve(contract.sign.size());
  for (const auto value : contract.sign) {
    signs.push_back(value);
  }
  return {
      {"schema_version", "1.0"},
      {"target", target_name(contract.target)},
      {"space", space_name(contract.space)},
      {"frame", frame_name(contract.frame)},
      {"permutation", contract.permutation},
      {"sign", signs},
      {"scale", contract.scale},
      {"lag_steps", contract.lag_steps},
      {"gripper_inverted", contract.gripper_inverted},
  };
}

ActionContract contract_from_json(const nlohmann::json& encoded) {
  if (!encoded.is_object()) {
    throw std::invalid_argument("contract JSON must be an object");
  }
  const std::set<std::string> allowed = {
      "schema_version", "target", "space", "frame", "permutation",
      "sign",           "scale",  "lag_steps", "gripper_inverted",
  };
  for (const auto& [key, value] : encoded.items()) {
    static_cast<void>(value);
    if (!allowed.contains(key)) {
      throw std::invalid_argument("unknown contract field: " + key);
    }
  }
  if (encoded.at("schema_version").get<std::string>() != "1.0") {
    throw std::invalid_argument("unsupported contract schema_version");
  }
  std::vector<std::int8_t> signs;
  for (const int value : encoded.at("sign").get<std::vector<int>>()) {
    signs.push_back(static_cast<std::int8_t>(value));
  }
  ActionContract contract{
      .target = parse_target(encoded.at("target").get<std::string>()),
      .space = parse_space(encoded.at("space").get<std::string>()),
      .frame = parse_frame(encoded.at("frame").get<std::string>()),
      .permutation =
          encoded.at("permutation").get<std::vector<std::uint32_t>>(),
      .sign = std::move(signs),
      .scale = encoded.at("scale").get<std::vector<double>>(),
      .lag_steps = encoded.at("lag_steps").get<std::int32_t>(),
      .gripper_inverted = encoded.at("gripper_inverted").get<bool>(),
  };
  validate(contract);
  return contract;
}

}  // namespace actionabi
