#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "actionabi/contract.hpp"
#include "actionabi/score.hpp"

namespace actionabi {

struct ScoredHypothesis {
  ActionContract contract;
  ScoreResult score;
};

struct Counterexample {
  std::int64_t episode_id;
  std::size_t row;
  std::size_t component;
  double absolute_residual;
};

struct EvidenceReport {
  std::string schema_version;
  std::string dataset_sha256;
  std::string source_filename;
  std::vector<std::int64_t> train_episode_ids;
  std::vector<std::int64_t> heldout_episode_ids;
  ScoreOptions score_options;
  std::vector<ScoredHypothesis> ranked;
  std::vector<Counterexample> counterexamples;
  std::vector<std::string> warnings;
  std::string backend;
  std::string build_revision;
};

nlohmann::json evidence_to_json(const EvidenceReport& report);

}  // namespace actionabi
