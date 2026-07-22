#include "actionabi/probe.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace actionabi {
namespace {

using PairSet = std::set<HypothesisPair>;

bool separated(const std::vector<double>& left, const std::vector<double>& right,
               double tolerance) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("predicted observation dimensions must match");
  }
  for (std::size_t component = 0; component < left.size(); ++component) {
    if (std::abs(left[component] - right[component]) > tolerance) {
      return true;
    }
  }
  return false;
}

PairSet candidate_cover(const ProbeCandidate& candidate, double tolerance) {
  PairSet cover;
  if (!candidate.safe) {
    return cover;
  }
  for (std::size_t first = 0; first < candidate.predicted_observations.size();
       ++first) {
    for (std::size_t second = first + 1;
         second < candidate.predicted_observations.size(); ++second) {
      if (separated(candidate.predicted_observations[first],
                    candidate.predicted_observations[second], tolerance)) {
        cover.insert({first, second});
      }
    }
  }
  return cover;
}

bool covers_all(const PairSet& covered, const PairSet& universe) {
  return std::includes(covered.begin(), covered.end(), universe.begin(),
                       universe.end());
}

std::vector<std::size_t> greedy_cover(const std::vector<PairSet>& covers,
                                      const PairSet& universe) {
  PairSet covered;
  std::vector<std::size_t> selected;
  while (!covers_all(covered, universe)) {
    std::size_t best_index = covers.size();
    std::size_t best_gain = 0;
    for (std::size_t index = 0; index < covers.size(); ++index) {
      std::size_t gain = 0;
      for (const auto& pair : covers[index]) {
        gain += !covered.contains(pair);
      }
      if (gain > best_gain) {
        best_gain = gain;
        best_index = index;
      }
    }
    if (best_gain == 0) {
      break;
    }
    selected.push_back(best_index);
    covered.insert(covers[best_index].begin(), covers[best_index].end());
  }
  return selected;
}

std::vector<std::size_t> exact_cover(const std::vector<PairSet>& covers,
                                     const PairSet& universe) {
  auto best = greedy_cover(covers, universe);
  std::vector<std::size_t> selected;
  PairSet covered;
  std::function<void()> search = [&]() {
    if (covers_all(covered, universe)) {
      auto candidate = selected;
      std::sort(candidate.begin(), candidate.end());
      if (best.empty() || candidate.size() < best.size() ||
          (candidate.size() == best.size() && candidate < best)) {
        best = std::move(candidate);
      }
      return;
    }
    if (!best.empty() && selected.size() >= best.size()) {
      return;
    }
    HypothesisPair target{};
    std::size_t fewest_choices = std::numeric_limits<std::size_t>::max();
    for (const auto& pair : universe) {
      if (covered.contains(pair)) {
        continue;
      }
      std::size_t choices = 0;
      for (const auto& cover : covers) {
        choices += cover.contains(pair);
      }
      if (choices < fewest_choices) {
        target = pair;
        fewest_choices = choices;
      }
    }
    for (std::size_t index = 0; index < covers.size(); ++index) {
      if (!covers[index].contains(target) ||
          std::find(selected.begin(), selected.end(), index) != selected.end()) {
        continue;
      }
      const auto previous = covered;
      selected.push_back(index);
      covered.insert(covers[index].begin(), covers[index].end());
      search();
      selected.pop_back();
      covered = previous;
    }
  };
  search();
  return best;
}

}  // namespace

ProbeSelection select_separating_probes(
    const std::vector<ProbeCandidate>& candidates,
    double observation_tolerance) {
  if (candidates.empty()) {
    throw std::invalid_argument("at least one probe candidate is required");
  }
  if (!std::isfinite(observation_tolerance) || observation_tolerance < 0.0) {
    throw std::invalid_argument(
        "observation_tolerance must be finite and nonnegative");
  }
  const auto hypothesis_count = candidates.front().predicted_observations.size();
  if (hypothesis_count < 2) {
    throw std::invalid_argument("at least two hypotheses are required");
  }
  for (const auto& candidate : candidates) {
    if (candidate.predicted_observations.size() != hypothesis_count) {
      throw std::invalid_argument(
          "all candidates must predict the same hypothesis set");
    }
  }
  PairSet all_pairs;
  for (std::size_t first = 0; first < hypothesis_count; ++first) {
    for (std::size_t second = first + 1; second < hypothesis_count; ++second) {
      all_pairs.insert({first, second});
    }
  }
  std::vector<PairSet> covers;
  covers.reserve(candidates.size());
  PairSet separable;
  for (const auto& candidate : candidates) {
    covers.push_back(candidate_cover(candidate, observation_tolerance));
    separable.insert(covers.back().begin(), covers.back().end());
  }
  std::vector<HypothesisPair> inseparable;
  std::set_difference(all_pairs.begin(), all_pairs.end(), separable.begin(),
                      separable.end(), std::back_inserter(inseparable));
  const bool exact = hypothesis_count <= 20;
  auto selected =
      exact ? exact_cover(covers, separable) : greedy_cover(covers, separable);
  return {std::move(selected), std::move(inseparable), exact};
}

std::vector<std::vector<double>> bounded_axis_pulses(
    const std::vector<double>& lower, const std::vector<double>& upper,
    const std::vector<double>& amplitude) {
  if (lower.size() != upper.size() || lower.size() != amplitude.size() ||
      lower.empty()) {
    throw std::invalid_argument("pulse bound dimensions must match and be nonempty");
  }
  std::vector<std::vector<double>> pulses;
  for (std::size_t component = 0; component < lower.size(); ++component) {
    if (!std::isfinite(lower[component]) || !std::isfinite(upper[component]) ||
        !std::isfinite(amplitude[component]) || amplitude[component] <= 0.0 ||
        lower[component] > 0.0 || upper[component] < 0.0) {
      throw std::invalid_argument("pulse bounds and amplitudes are invalid");
    }
    if (amplitude[component] <= upper[component]) {
      std::vector<double> pulse(lower.size(), 0.0);
      pulse[component] = amplitude[component];
      pulses.push_back(std::move(pulse));
    }
    if (-amplitude[component] >= lower[component]) {
      std::vector<double> pulse(lower.size(), 0.0);
      pulse[component] = -amplitude[component];
      pulses.push_back(std::move(pulse));
    }
  }
  return pulses;
}

}  // namespace actionabi
