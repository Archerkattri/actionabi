#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "actionabi/probe.hpp"

TEST_CASE("exact selector finds the known one-probe minimum") {
  const std::vector<actionabi::ProbeCandidate> candidates = {
      {{0.1}, {{0.0}, {0.0}, {1.0}}, true},
      {{0.2}, {{0.0}, {1.0}, {2.0}}, true},
      {{0.3}, {{0.0}, {0.0}, {0.0}}, true},
  };

  const auto result = actionabi::select_separating_probes(candidates, 1e-9);

  CHECK(result.selected_candidate_indices == std::vector<std::size_t>{1});
  CHECK(result.inseparable_pairs.empty());
  CHECK(result.exact);
}

TEST_CASE("selector reports hypothesis pairs no safe probe can separate") {
  const std::vector<actionabi::ProbeCandidate> candidates = {
      {{0.1}, {{0.0}, {0.0}, {1.0}}, true},
      {{0.2}, {{0.0}, {1.0}, {2.0}}, false},
  };

  const auto result = actionabi::select_separating_probes(candidates, 1e-9);

  CHECK(result.inseparable_pairs ==
        std::vector<actionabi::HypothesisPair>{{0, 1}});
}

TEST_CASE("exact selector uses two probes when no single probe covers all pairs") {
  const std::vector<actionabi::ProbeCandidate> candidates = {
      {{0.1}, {{0.0}, {1.0}, {0.0}, {1.0}}, true},
      {{0.2}, {{0.0}, {0.0}, {1.0}, {1.0}}, true},
  };

  const auto result = actionabi::select_separating_probes(candidates, 1e-9);

  CHECK(result.selected_candidate_indices ==
        std::vector<std::size_t>{0, 1});
  CHECK(result.inseparable_pairs.empty());
}

TEST_CASE("bounded axis pulses never exceed component safety limits") {
  const auto pulses = actionabi::bounded_axis_pulses(
      {-0.1, -0.2}, {0.1, 0.3}, {0.05, 0.25});

  CHECK(pulses == std::vector<std::vector<double>>{
                       {0.05, 0.0}, {-0.05, 0.0}, {0.0, 0.25}});
}
