#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <nlohmann/json.hpp>

#include "actionabi/contract.hpp"

namespace {

actionabi::ActionContract valid_contract() {
  return {
      .target = actionabi::TargetKind::DeltaPosition,
      .space = actionabi::ActionSpace::Cartesian,
      .frame = actionabi::ReferenceFrame::Tool,
      .permutation = {1, 0},
      .sign = {1, -1},
      .scale = {0.01, 2.0},
      .lag_steps = 2,
      .gripper_inverted = true,
  };
}

}  // namespace

TEST_CASE("contract JSON round trip uses stable enum strings") {
  const auto contract = valid_contract();

  const nlohmann::json encoded = actionabi::to_json(contract);
  const auto decoded = actionabi::contract_from_json(encoded);

  CHECK(decoded == contract);
  CHECK(encoded.at("target") == "delta_position");
  CHECK(encoded.at("space") == "cartesian");
  CHECK(encoded.at("frame") == "tool");
  CHECK(encoded.at("schema_version") == "1.0");
}

TEST_CASE("contract rejects a non-bijective permutation") {
  auto contract = valid_contract();
  contract.permutation = {0, 0};
  CHECK_THROWS_WITH(actionabi::validate(contract),
                    "permutation must be a bijection");
}

TEST_CASE("contract rejects invalid sign") {
  auto contract = valid_contract();
  contract.sign = {1, 0};
  CHECK_THROWS_WITH(actionabi::validate(contract),
                    "sign values must be -1 or 1");
}

TEST_CASE("contract rejects nonpositive or nonfinite scale") {
  auto contract = valid_contract();
  contract.scale = {1.0, 0.0};
  CHECK_THROWS_WITH(actionabi::validate(contract),
                    "scale values must be finite and positive");
}

TEST_CASE("contract rejects mismatched component dimensions") {
  auto contract = valid_contract();
  contract.sign = {1};
  CHECK_THROWS_WITH(actionabi::validate(contract),
                    "contract component dimensions must match");
}

TEST_CASE("joint actions reject Cartesian reference frames") {
  auto contract = valid_contract();
  contract.space = actionabi::ActionSpace::Joint;
  CHECK_THROWS_WITH(actionabi::validate(contract),
                    "joint actions require an unspecified reference frame");
}

TEST_CASE("JSON rejects unknown fields") {
  auto encoded = actionabi::to_json(valid_contract());
  encoded["oracle_contract_id"] = 7;
  CHECK_THROWS_WITH(actionabi::contract_from_json(encoded),
                    "unknown contract field: oracle_contract_id");
}
