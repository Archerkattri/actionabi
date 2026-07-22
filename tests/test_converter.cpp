#include <map>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <nlohmann/json.hpp>

#include "actionabi/contract.hpp"
#include "actionabi/converter.hpp"
#include "actionabi/evidence.hpp"
#include "actionabi/identifiability.hpp"

namespace {

actionabi::ActionContract contract() {
  return {
      .target = actionabi::TargetKind::DeltaPosition,
      .space = actionabi::ActionSpace::Joint,
      .frame = actionabi::ReferenceFrame::Unspecified,
      .permutation = {1, 0},
      .sign = {1, -1},
      .scale = {0.5, 2.0},
      .lag_steps = 1,
      .gripper_inverted = true,
  };
}

actionabi::EquivalenceReport identified_report() {
  const auto scored = actionabi::ScoredHypothesis{
      contract(), {0.0, 0.0, 10, {0.0, 0.0}}};
  return actionabi::analyze_identifiability({scored}, 0.0, 0.95, {});
}

}  // namespace

TEST_CASE("converter applies permutation sign scale and gripper inversion") {
  const auto spec = actionabi::make_converter(
      identified_report(), std::string(64, 'a'));

  const auto converted = actionabi::convert_instantaneous(spec, {2.0, 3.0});

  REQUIRE(converted.size() == 2);
  CHECK(converted[0] == Catch::Approx(1.5));
  CHECK(converted[1] == Catch::Approx(4.0));
}

TEST_CASE("converter serialization is bound to evidence hash") {
  const auto spec = actionabi::make_converter(
      identified_report(), std::string(64, 'a'));
  const auto encoded = actionabi::converter_to_json(spec);

  const auto loaded = actionabi::converter_from_json(
      encoded, std::string(64, 'a'));

  CHECK(loaded.contract == spec.contract);
  CHECK_THROWS_WITH(
      actionabi::converter_from_json(encoded, std::string(64, 'b')),
      "converter evidence hash does not match the loaded report");
}

TEST_CASE("ambiguous contract fails closed") {
  auto alternate = contract();
  alternate.target = actionabi::TargetKind::AbsolutePosition;
  const std::vector<actionabi::ScoredHypothesis> tied = {
      {contract(), {0.0, 0.0, 10, {0.0, 0.0}}},
      {alternate, {0.0, 0.0, 10, {0.0, 0.0}}},
  };
  const auto report = actionabi::analyze_identifiability(tied, 0.0, 0.95, {});

  CHECK_THROWS_WITH(
      actionabi::make_converter(report, std::string(64, 'a')),
      "cannot emit converter: field target is ambiguous");
}

TEST_CASE("unsupported field fails closed") {
  const auto scored = actionabi::ScoredHypothesis{
      contract(), {0.0, 0.0, 10, {0.0, 0.0}}};
  const auto report = actionabi::analyze_identifiability(
      {scored}, 0.0, 0.95, {"frame"});

  CHECK_THROWS_WITH(
      actionabi::make_converter(report, std::string(64, 'a')),
      "cannot emit converter: field frame is unsupported");
}
