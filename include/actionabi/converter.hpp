#pragma once

#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "actionabi/contract.hpp"
#include "actionabi/identifiability.hpp"

namespace actionabi {

struct ConverterSpec {
  std::string schema_version;
  std::string evidence_report_sha256;
  ActionContract contract;
};

ConverterSpec make_converter(const EquivalenceReport& report,
                             const std::string& evidence_report_sha256);
std::vector<double> convert_instantaneous(const ConverterSpec& spec,
                                          const std::vector<double>& raw_action);
nlohmann::json converter_to_json(const ConverterSpec& spec);
ConverterSpec converter_from_json(const nlohmann::json& encoded,
                                  const std::string& expected_evidence_sha256);

}  // namespace actionabi
