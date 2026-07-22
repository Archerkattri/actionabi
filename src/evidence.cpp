#include "actionabi/evidence.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace actionabi {
namespace {

nlohmann::json score_to_json(const ScoreResult& score) {
  return {
      {"train_loss", score.train_loss},
      {"heldout_loss", score.heldout_loss},
      {"residual_count", score.residual_count},
      {"per_dimension_loss", score.per_dimension_loss},
  };
}

}  // namespace

nlohmann::json evidence_to_json(const EvidenceReport& report) {
  if (report.schema_version != "1.0") {
    throw std::invalid_argument("unsupported evidence schema_version");
  }
  auto ranked = report.ranked;
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const ScoredHypothesis& left,
                      const ScoredHypothesis& right) {
                     return left.score.heldout_loss < right.score.heldout_loss;
                   });
  nlohmann::json hypotheses = nlohmann::json::array();
  for (const auto& hypothesis : ranked) {
    hypotheses.push_back({
        {"contract", to_json(hypothesis.contract)},
        {"score", score_to_json(hypothesis.score)},
    });
  }
  nlohmann::json counterexamples = nlohmann::json::array();
  for (const auto& counterexample : report.counterexamples) {
    counterexamples.push_back({
        {"episode_id", counterexample.episode_id},
        {"row", counterexample.row},
        {"component", counterexample.component},
        {"absolute_residual", counterexample.absolute_residual},
    });
  }
  return {
      {"schema_version", report.schema_version},
      {"dataset",
       {{"source_filename", report.source_filename},
        {"sha256", report.dataset_sha256}}},
      {"split",
       {{"train_episode_ids", report.train_episode_ids},
        {"heldout_episode_ids", report.heldout_episode_ids}}},
      {"score_options",
       {{"huber_delta", report.score_options.huber_delta},
        {"train_episode_fraction",
         report.score_options.train_episode_fraction}}},
      {"ranked_hypotheses", std::move(hypotheses)},
      {"counterexamples", std::move(counterexamples)},
      {"warnings", report.warnings},
      {"backend", report.backend},
      {"build_revision", report.build_revision},
  };
}

}  // namespace actionabi
