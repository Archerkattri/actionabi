#include "actionabi/identifiability.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace actionabi {
namespace {

template <typename Projection>
FieldStatus field_status(const std::vector<ScoredHypothesis>& retained,
                         const std::string& field,
                         const std::set<std::string>& unsupported,
                         Projection projection) {
  if (unsupported.contains(field)) {
    return FieldStatus::Unsupported;
  }
  const auto first = projection(retained.front().contract);
  const bool identical = std::all_of(
      retained.begin() + 1, retained.end(), [&](const ScoredHypothesis& item) {
        return projection(item.contract) == first;
      });
  return identical ? FieldStatus::Identified : FieldStatus::Ambiguous;
}

}  // namespace

EquivalenceReport analyze_identifiability(
    const std::vector<ScoredHypothesis>& ranked, double score_threshold,
    double calibrated_coverage,
    const std::set<std::string>& unsupported_fields) {
  if (ranked.empty()) {
    throw std::invalid_argument("at least one scored hypothesis is required");
  }
  if (!std::isfinite(score_threshold) || score_threshold < 0.0) {
    throw std::invalid_argument("score_threshold must be finite and nonnegative");
  }
  if (!std::isfinite(calibrated_coverage) || calibrated_coverage < 0.0 ||
      calibrated_coverage > 1.0) {
    throw std::invalid_argument("calibrated_coverage must be between zero and one");
  }
  const double best = std::min_element(
                          ranked.begin(), ranked.end(),
                          [](const ScoredHypothesis& left,
                             const ScoredHypothesis& right) {
                            return left.score.heldout_loss <
                                   right.score.heldout_loss;
                          })
                          ->score.heldout_loss;
  std::vector<ScoredHypothesis> retained;
  std::copy_if(ranked.begin(), ranked.end(), std::back_inserter(retained),
               [&](const ScoredHypothesis& hypothesis) {
                 return hypothesis.score.heldout_loss <= best + score_threshold;
               });
  std::stable_sort(retained.begin(), retained.end(),
                   [](const ScoredHypothesis& left,
                      const ScoredHypothesis& right) {
                     return left.score.heldout_loss < right.score.heldout_loss;
                   });

  std::map<std::string, FieldStatus> fields;
  fields["target"] = field_status(retained, "target", unsupported_fields,
                                   [](const ActionContract& value) {
                                     return value.target;
                                   });
  fields["space"] = field_status(retained, "space", unsupported_fields,
                                  [](const ActionContract& value) {
                                    return value.space;
                                  });
  fields["frame"] = field_status(retained, "frame", unsupported_fields,
                                  [](const ActionContract& value) {
                                    return value.frame;
                                  });
  fields["permutation"] = field_status(
      retained, "permutation", unsupported_fields,
      [](const ActionContract& value) -> const auto& { return value.permutation; });
  fields["sign"] = field_status(
      retained, "sign", unsupported_fields,
      [](const ActionContract& value) -> const auto& { return value.sign; });
  fields["scale"] = field_status(
      retained, "scale", unsupported_fields,
      [](const ActionContract& value) -> const auto& { return value.scale; });
  fields["lag_steps"] = field_status(retained, "lag_steps", unsupported_fields,
                                      [](const ActionContract& value) {
                                        return value.lag_steps;
                                      });
  fields["gripper_inverted"] = field_status(
      retained, "gripper_inverted", unsupported_fields,
      [](const ActionContract& value) { return value.gripper_inverted; });
  return {std::move(retained), std::move(fields), calibrated_coverage};
}

double split_conformal_threshold(const std::vector<double>& calibration_scores,
                                 double alpha) {
  if (calibration_scores.empty()) {
    throw std::invalid_argument("calibration_scores must not be empty");
  }
  if (!std::isfinite(alpha) || alpha <= 0.0 || alpha >= 1.0) {
    throw std::invalid_argument("alpha must be between zero and one");
  }
  auto sorted = calibration_scores;
  if (std::any_of(sorted.begin(), sorted.end(),
                  [](double value) { return !std::isfinite(value) || value < 0.0; })) {
    throw std::invalid_argument(
        "calibration scores must be finite and nonnegative");
  }
  std::sort(sorted.begin(), sorted.end());
  const auto rank = static_cast<std::size_t>(std::ceil(
      (static_cast<double>(sorted.size()) + 1.0) * (1.0 - alpha)));
  const auto index = std::min(std::max<std::size_t>(rank, 1), sorted.size()) - 1;
  return sorted[index];
}

CoverageInterval wilson_interval(std::size_t successes, std::size_t trials,
                                 double z) {
  if (trials == 0 || successes > trials) {
    throw std::invalid_argument("coverage counts are invalid");
  }
  if (!std::isfinite(z) || z <= 0.0) {
    throw std::invalid_argument("z must be finite and positive");
  }
  const double n = static_cast<double>(trials);
  const double proportion = static_cast<double>(successes) / n;
  const double z_squared = z * z;
  const double denominator = 1.0 + z_squared / n;
  const double center = (proportion + z_squared / (2.0 * n)) / denominator;
  const double radius =
      z * std::sqrt((proportion * (1.0 - proportion) / n) +
                    z_squared / (4.0 * n * n)) /
      denominator;
  return {std::max(0.0, center - radius), std::min(1.0, center + radius)};
}

}  // namespace actionabi
