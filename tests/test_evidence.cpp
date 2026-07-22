#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "actionabi/contract.hpp"
#include "actionabi/evidence.hpp"
#include "actionabi/score.hpp"

namespace {

actionabi::ActionContract contract(double scale) {
  return {
      .target = actionabi::TargetKind::DeltaPosition,
      .space = actionabi::ActionSpace::Joint,
      .frame = actionabi::ReferenceFrame::Unspecified,
      .permutation = {0},
      .sign = {1},
      .scale = {scale},
      .lag_steps = 0,
      .gripper_inverted = false,
  };
}

}  // namespace

TEST_CASE("evidence report contains required provenance and split fields") {
  actionabi::EvidenceReport report{
      .schema_version = "1.0",
      .dataset_sha256 = std::string(64, 'a'),
      .source_filename = "source.parquet",
      .train_episode_ids = {0, 1},
      .heldout_episode_ids = {2},
      .score_options = {.huber_delta = 0.5, .train_episode_fraction = 0.7},
      .ranked = {
          {contract(2.0), {0.2, 0.3, 10, {0.3}}},
          {contract(1.0), {0.0, 0.1, 10, {0.1}}},
      },
      .counterexamples = {{2, 4, 0, 0.7}},
      .warnings = {"one stationary component"},
      .backend = "cpu",
      .build_revision = "abc1234",
  };

  const auto encoded = actionabi::evidence_to_json(report);

  CHECK(encoded.at("dataset").at("sha256") == std::string(64, 'a'));
  CHECK(encoded.at("split").at("train_episode_ids") ==
        nlohmann::json::array({0, 1}));
  CHECK(encoded.at("split").at("heldout_episode_ids") ==
        nlohmann::json::array({2}));
  CHECK(encoded.at("backend") == "cpu");
  CHECK(encoded.at("ranked_hypotheses").size() == 2);
  CHECK(encoded.at("counterexamples").size() == 1);
}

TEST_CASE("evidence serialization sorts hypotheses by held-out loss") {
  actionabi::EvidenceReport report{
      .schema_version = "1.0",
      .dataset_sha256 = std::string(64, 'a'),
      .source_filename = "source.parquet",
      .train_episode_ids = {0},
      .heldout_episode_ids = {1},
      .score_options = {},
      .ranked = {
          {contract(2.0), {0.2, 0.3, 10, {0.3}}},
          {contract(1.0), {0.0, 0.1, 10, {0.1}}},
      },
      .counterexamples = {},
      .warnings = {},
      .backend = "cpu",
      .build_revision = "abc1234",
  };

  const auto first = actionabi::evidence_to_json(report).dump(2);
  const auto second = actionabi::evidence_to_json(report).dump(2);
  const auto parsed = nlohmann::json::parse(first);

  CHECK(first == second);
  CHECK(parsed.at("ranked_hypotheses")[0].at("score").at("heldout_loss") ==
        0.1);
  CHECK(parsed.at("ranked_hypotheses")[0].at("contract").at("scale")[0] ==
        1.0);
}
