#include "actionabi/trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace actionabi {
namespace {

std::string required_string(const nlohmann::json& object,
                            const std::string& field) {
  if (!object.contains(field)) {
    throw std::invalid_argument("metadata is missing provenance field: " + field);
  }
  return object.at(field).get<std::string>();
}

SourceProvenance parse_metadata(const nlohmann::json& metadata) {
  if (metadata.value("record_type", "") != "metadata") {
    throw std::invalid_argument("first JSONL record must be metadata");
  }
  if (metadata.value("schema_version", "") != "1.0") {
    throw std::invalid_argument("unsupported trajectory schema_version");
  }
  SourceProvenance provenance{
      .source_filename = required_string(metadata, "source_filename"),
      .source_sha256 = required_string(metadata, "source_sha256"),
      .extraction_date = required_string(metadata, "extraction_date"),
      .state_columns =
          metadata.at("state_columns").get<std::vector<std::string>>(),
      .state_units = metadata.at("state_units").get<std::vector<std::string>>(),
  };
  if (provenance.state_columns.empty()) {
    throw std::invalid_argument("metadata state_columns must not be empty");
  }
  if (provenance.state_columns.size() != provenance.state_units.size()) {
    throw std::invalid_argument(
        "metadata state_columns and state_units must have equal length");
  }
  return provenance;
}

[[noreturn]] void row_error(std::int64_t episode_id, std::size_t row,
                            const std::string& message) {
  throw std::invalid_argument("episode " + std::to_string(episode_id) +
                              " row " + std::to_string(row) + ": " + message);
}

void require_finite(const std::vector<double>& values, std::int64_t episode_id,
                    std::size_t row, const std::string& field) {
  if (std::any_of(values.begin(), values.end(),
                  [](double value) { return !std::isfinite(value); })) {
    row_error(episode_id, row, field + " contains a nonfinite value");
  }
}

}  // namespace

std::span<const double> TrajectoryDataset::state(std::size_t row) const {
  if (row >= row_count()) {
    throw std::out_of_range("state row is out of range");
  }
  return {states_.data() + row * state_dimension_, state_dimension_};
}

std::span<const double> TrajectoryDataset::action(std::size_t row) const {
  if (row >= row_count()) {
    throw std::out_of_range("action row is out of range");
  }
  return {actions_.data() + row * action_dimension_, action_dimension_};
}

TrajectoryDataset load_jsonl(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::invalid_argument("could not open trajectory: " + path.string());
  }
  std::string line;
  if (!std::getline(input, line)) {
    throw std::invalid_argument("trajectory JSONL is empty");
  }
  TrajectoryDataset dataset;
  dataset.provenance_ = parse_metadata(nlohmann::json::parse(line));
  dataset.state_dimension_ = dataset.provenance_.state_columns.size();

  std::optional<std::int64_t> current_episode;
  std::optional<std::int64_t> previous_timestamp;
  std::set<std::int64_t> completed_episodes;
  std::size_t row = 0;
  while (std::getline(input, line)) {
    if (line.empty()) {
      throw std::invalid_argument("blank JSONL record at sample row " +
                                  std::to_string(row));
    }
    const auto record = nlohmann::json::parse(line);
    if (record.value("record_type", "") != "sample") {
      throw std::invalid_argument("non-sample JSONL record at sample row " +
                                  std::to_string(row));
    }
    const auto episode_id = record.at("episode_id").get<std::int64_t>();
    const auto timestamp = record.at("t_ns").get<std::int64_t>();
    const auto state = record.at("state").get<std::vector<double>>();
    const auto action = record.at("action").get<std::vector<double>>();

    if (!current_episode || episode_id != *current_episode) {
      if (completed_episodes.contains(episode_id)) {
        row_error(episode_id, row, "episode rows must be contiguous");
      }
      if (current_episode) {
        dataset.episodes_.back().end = row;
        completed_episodes.insert(*current_episode);
      }
      current_episode = episode_id;
      previous_timestamp.reset();
      dataset.episodes_.push_back({episode_id, row, row});
    }
    if (previous_timestamp && timestamp <= *previous_timestamp) {
      row_error(episode_id, row, "timestamps must be strictly increasing");
    }
    if (state.size() != dataset.state_dimension_) {
      row_error(episode_id, row,
                "state dimension changed from " +
                    std::to_string(dataset.state_dimension_) + " to " +
                    std::to_string(state.size()));
    }
    if (row == 0) {
      if (action.empty()) {
        row_error(episode_id, row, "action must not be empty");
      }
      dataset.action_dimension_ = action.size();
    } else if (action.size() != dataset.action_dimension_) {
      row_error(episode_id, row,
                "action dimension changed from " +
                    std::to_string(dataset.action_dimension_) + " to " +
                    std::to_string(action.size()));
    }
    require_finite(state, episode_id, row, "state");
    require_finite(action, episode_id, row, "action");
    dataset.states_.insert(dataset.states_.end(), state.begin(), state.end());
    dataset.actions_.insert(dataset.actions_.end(), action.begin(), action.end());
    dataset.timestamps_ns_.push_back(timestamp);
    previous_timestamp = timestamp;
    ++row;
  }
  if (row == 0) {
    throw std::invalid_argument("trajectory contains no sample records");
  }
  dataset.episodes_.back().end = row;
  return dataset;
}

}  // namespace actionabi
