#pragma once

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

namespace actionabi {

struct ProbeCandidate {
  std::vector<double> action;
  std::vector<std::vector<double>> predicted_observations;
  bool safe;
};

struct HypothesisPair {
  std::size_t first;
  std::size_t second;

  bool operator==(const HypothesisPair&) const = default;
  bool operator<(const HypothesisPair& other) const {
    return std::tie(first, second) < std::tie(other.first, other.second);
  }
};

struct ProbeSelection {
  std::vector<std::size_t> selected_candidate_indices;
  std::vector<HypothesisPair> inseparable_pairs;
  bool exact;
};

ProbeSelection select_separating_probes(
    const std::vector<ProbeCandidate>& candidates, double observation_tolerance);

std::vector<std::vector<double>> bounded_axis_pulses(
    const std::vector<double>& lower, const std::vector<double>& upper,
    const std::vector<double>& amplitude);

}  // namespace actionabi
