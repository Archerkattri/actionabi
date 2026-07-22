#include "actionabi/converter.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace actionabi {
namespace {

std::string status_name(FieldStatus status) {
  switch (status) {
    case FieldStatus::Identified:
      return "identified";
    case FieldStatus::Ambiguous:
      return "ambiguous";
    case FieldStatus::Unsupported:
      return "unsupported";
  }
  throw std::invalid_argument("invalid field status");
}

void validate_sha256(const std::string& hash) {
  if (hash.size() != 64 ||
      !std::all_of(hash.begin(), hash.end(), [](unsigned char character) {
        return std::isxdigit(character) != 0;
      })) {
    throw std::invalid_argument("evidence_report_sha256 must be 64 hexadecimal characters");
  }
}

}  // namespace

ConverterSpec make_converter(const EquivalenceReport& report,
                             const std::string& evidence_report_sha256) {
  if (report.retained.empty()) {
    throw std::invalid_argument("cannot emit converter from an empty equivalence set");
  }
  validate_sha256(evidence_report_sha256);
  const std::vector<std::string> required_fields = {
      "target", "space", "frame", "permutation", "sign", "scale",
      "lag_steps", "gripper_inverted",
  };
  for (const auto& field : required_fields) {
    const auto status = report.fields.at(field);
    if (status != FieldStatus::Identified) {
      throw std::invalid_argument("cannot emit converter: field " + field +
                                  " is " + status_name(status));
    }
  }
  return {"1.0", evidence_report_sha256, report.retained.front().contract};
}

std::vector<double> convert_instantaneous(
    const ConverterSpec& spec, const std::vector<double>& raw_action) {
  validate(spec.contract);
  if (raw_action.size() != spec.contract.permutation.size()) {
    throw std::invalid_argument("raw action dimension does not match converter");
  }
  std::vector<double> converted(raw_action.size());
  for (std::size_t component = 0; component < converted.size(); ++component) {
    converted[component] =
        raw_action[spec.contract.permutation[component]] *
        spec.contract.sign[component] * spec.contract.scale[component];
  }
  if (spec.contract.gripper_inverted) {
    converted.back() *= -1.0;
  }
  return converted;
}

nlohmann::json converter_to_json(const ConverterSpec& spec) {
  validate_sha256(spec.evidence_report_sha256);
  if (spec.schema_version != "1.0") {
    throw std::invalid_argument("unsupported converter schema_version");
  }
  return {
      {"schema_version", spec.schema_version},
      {"evidence_report_sha256", spec.evidence_report_sha256},
      {"contract", to_json(spec.contract)},
  };
}

ConverterSpec converter_from_json(
    const nlohmann::json& encoded,
    const std::string& expected_evidence_sha256) {
  validate_sha256(expected_evidence_sha256);
  const auto hash = encoded.at("evidence_report_sha256").get<std::string>();
  validate_sha256(hash);
  if (hash != expected_evidence_sha256) {
    throw std::invalid_argument(
        "converter evidence hash does not match the loaded report");
  }
  ConverterSpec spec{
      .schema_version = encoded.at("schema_version").get<std::string>(),
      .evidence_report_sha256 = hash,
      .contract = contract_from_json(encoded.at("contract")),
  };
  static_cast<void>(converter_to_json(spec));
  return spec;
}

}  // namespace actionabi
