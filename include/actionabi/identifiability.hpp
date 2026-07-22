#pragma once

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "actionabi/evidence.hpp"

namespace actionabi {

enum class FieldStatus { Identified, Ambiguous, Unsupported };

struct EquivalenceReport {
  std::vector<ScoredHypothesis> retained;
  std::map<std::string, FieldStatus> fields;
  double calibrated_coverage;
};

struct CoverageInterval {
  double lower;
  double upper;
};

EquivalenceReport analyze_identifiability(
    const std::vector<ScoredHypothesis>& ranked, double score_threshold,
    double calibrated_coverage, const std::set<std::string>& unsupported_fields);

double split_conformal_threshold(const std::vector<double>& calibration_scores,
                                 double alpha);
CoverageInterval wilson_interval(std::size_t successes, std::size_t trials,
                                 double z = 1.959963984540054);

}  // namespace actionabi
