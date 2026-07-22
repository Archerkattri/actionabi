#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace actionabi {

struct SourceProvenance {
  std::string source_filename;
  std::string source_sha256;
  std::string extraction_date;
  std::vector<std::string> state_columns;
  std::vector<std::string> state_units;
};

struct EpisodeSpan {
  std::int64_t episode_id;
  std::size_t begin;
  std::size_t end;

  bool operator==(const EpisodeSpan&) const = default;
};

class TrajectoryDataset {
 public:
  std::size_t row_count() const noexcept { return timestamps_ns_.size(); }
  std::size_t state_dimension() const noexcept { return state_dimension_; }
  std::size_t action_dimension() const noexcept { return action_dimension_; }
  const SourceProvenance& provenance() const noexcept { return provenance_; }
  const std::vector<EpisodeSpan>& episodes() const noexcept { return episodes_; }
  const std::vector<std::int64_t>& timestamps_ns() const noexcept {
    return timestamps_ns_;
  }
  const std::vector<double>& states_flat() const noexcept { return states_; }
  const std::vector<double>& actions_flat() const noexcept { return actions_; }
  std::span<const double> state(std::size_t row) const;
  std::span<const double> action(std::size_t row) const;

 private:
  friend TrajectoryDataset load_jsonl(const std::filesystem::path& path);

  SourceProvenance provenance_;
  std::size_t state_dimension_{0};
  std::size_t action_dimension_{0};
  std::vector<double> states_;
  std::vector<double> actions_;
  std::vector<std::int64_t> timestamps_ns_;
  std::vector<EpisodeSpan> episodes_;
};

TrajectoryDataset load_jsonl(const std::filesystem::path& path);

}  // namespace actionabi
