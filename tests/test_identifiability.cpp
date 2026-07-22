#include <map>
#include <set>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "actionabi/contract.hpp"
#include "actionabi/evidence.hpp"
#include "actionabi/identifiability.hpp"

namespace {

actionabi::ActionContract contract(actionabi::TargetKind target, int lag = 0) {
  return {
      .target = target,
      .space = actionabi::ActionSpace::Joint,
      .frame = actionabi::ReferenceFrame::Unspecified,
      .permutation = {0},
      .sign = {1},
      .scale = {1.0},
      .lag_steps = lag,
      .gripper_inverted = false,
  };
}

actionabi::ScoredHypothesis scored(actionabi::ActionContract value, double loss) {
  return {std::move(value), {loss, loss, 10, {loss}}};
}

}  // namespace

TEST_CASE("equal-score hypotheses remain an ambiguous equivalence set") {
  const std::vector<actionabi::ScoredHypothesis> ranked = {
      scored(contract(actionabi::TargetKind::AbsolutePosition), 0.1),
      scored(contract(actionabi::TargetKind::DeltaPosition), 0.1),
  };

  const auto report = actionabi::analyze_identifiability(
      ranked, 0.0, 0.95, {});

  CHECK(report.retained.size() == 2);
  CHECK(report.fields.at("target") == actionabi::FieldStatus::Ambiguous);
  CHECK(report.fields.at("lag_steps") == actionabi::FieldStatus::Identified);
}

TEST_CASE("threshold retains near-best hypotheses without lexicographic tie breaking") {
  const std::vector<actionabi::ScoredHypothesis> ranked = {
      scored(contract(actionabi::TargetKind::DeltaPosition, 0), 0.1),
      scored(contract(actionabi::TargetKind::DeltaPosition, 1), 0.1009),
      scored(contract(actionabi::TargetKind::Velocity, 2), 0.2),
  };

  const auto report = actionabi::analyze_identifiability(
      ranked, 0.001, 0.95, {});

  CHECK(report.retained.size() == 2);
  CHECK(report.fields.at("target") == actionabi::FieldStatus::Identified);
  CHECK(report.fields.at("lag_steps") == actionabi::FieldStatus::Ambiguous);
}

TEST_CASE("caller can mark a field unsupported despite agreement") {
  const std::vector<actionabi::ScoredHypothesis> ranked = {
      scored(contract(actionabi::TargetKind::DeltaPosition), 0.1),
  };

  const auto report = actionabi::analyze_identifiability(
      ranked, 0.0, 0.95, {"space", "frame"});

  CHECK(report.fields.at("space") == actionabi::FieldStatus::Unsupported);
  CHECK(report.fields.at("frame") == actionabi::FieldStatus::Unsupported);
}

TEST_CASE("split-conformal threshold uses finite-sample upper quantile") {
  const std::vector<double> calibration_scores = {0.01, 0.02, 0.03, 0.04,
                                                   0.05, 0.06, 0.07, 0.08,
                                                   0.09, 0.10};

  const double threshold =
      actionabi::split_conformal_threshold(calibration_scores, 0.2);

  CHECK(threshold == Catch::Approx(0.09));
}

TEST_CASE("Wilson interval contains observed coverage") {
  const auto interval = actionabi::wilson_interval(95, 100);

  CHECK(interval.lower < 0.95);
  CHECK(interval.upper > 0.95);
  CHECK(interval.lower > 0.85);
  CHECK(interval.upper <= 1.0);
}
