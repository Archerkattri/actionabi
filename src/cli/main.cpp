#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "actionabi/contract.hpp"
#include "actionabi/evidence.hpp"
#include "actionabi/score.hpp"
#include "actionabi/trajectory.hpp"

#ifndef ACTIONABI_VERSION
#define ACTIONABI_VERSION "unknown"
#endif

#ifndef ACTIONABI_GIT_REVISION
#define ACTIONABI_GIT_REVISION "unknown"
#endif

int main(int argc, char** argv) {
  CLI::App app{"Infer and audit undocumented robot action tensors"};
  const std::string version =
      std::string{"ActionABI "} + ACTIONABI_VERSION + " (git " +
      ACTIONABI_GIT_REVISION + ")";
  app.set_version_flag("--version", version);
  auto* infer = app.add_subcommand("infer", "Score supplied contract hypotheses");
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  std::vector<std::filesystem::path> contract_paths;
  infer->add_option("--input", input_path, "canonical trajectory JSONL")
      ->required()
      ->check(CLI::ExistingFile);
  infer->add_option("--contract", contract_paths, "ActionSpec JSON; repeatable")
      ->required()
      ->check(CLI::ExistingFile);
  infer->add_option("--output", output_path, "evidence report JSON")
      ->required();
  CLI11_PARSE(app, argc, argv);
  if (*infer) {
    const auto dataset = actionabi::load_jsonl(input_path);
    const actionabi::ScoreOptions options;
    std::vector<actionabi::ScoredHypothesis> ranked;
    ranked.reserve(contract_paths.size());
    for (const auto& contract_path : contract_paths) {
      std::ifstream contract_input(contract_path);
      if (!contract_input) {
        throw std::runtime_error("could not open contract: " +
                                 contract_path.string());
      }
      nlohmann::json encoded;
      contract_input >> encoded;
      auto contract = actionabi::contract_from_json(encoded);
      ranked.push_back(
          {contract, actionabi::score_cpu(dataset, contract, options)});
    }
    const auto train_count = std::clamp<std::size_t>(
        static_cast<std::size_t>(
            options.train_episode_fraction * dataset.episodes().size()),
        1, dataset.episodes().size() - 1);
    std::vector<std::int64_t> train_episode_ids;
    std::vector<std::int64_t> heldout_episode_ids;
    for (std::size_t index = 0; index < dataset.episodes().size(); ++index) {
      auto& destination =
          index < train_count ? train_episode_ids : heldout_episode_ids;
      destination.push_back(dataset.episodes()[index].episode_id);
    }
    const actionabi::EvidenceReport report{
        .schema_version = "1.0",
        .dataset_sha256 = dataset.provenance().source_sha256,
        .source_filename = dataset.provenance().source_filename,
        .train_episode_ids = std::move(train_episode_ids),
        .heldout_episode_ids = std::move(heldout_episode_ids),
        .score_options = options,
        .ranked = std::move(ranked),
        .counterexamples = {},
        .warnings = {},
        .backend = "cpu",
        .build_revision = ACTIONABI_GIT_REVISION,
    };
    std::ofstream output(output_path);
    if (!output) {
      throw std::runtime_error("could not open output: " + output_path.string());
    }
    output << actionabi::evidence_to_json(report).dump(2) << '\n';
  }
  return 0;
}
