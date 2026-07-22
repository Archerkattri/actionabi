#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "actionabi/contract.hpp"
#include "actionabi/score_pinocchio.hpp"
#include "actionabi/trajectory.hpp"

namespace {

std::vector<double> tool_xy(double q0, double q1) {
  return {std::cos(q0) + std::cos(q0 + q1),
          std::sin(q0) + std::sin(q0 + q1)};
}

std::filesystem::path write_fixture(bool wrong_state_dimension = false) {
  const auto path = std::filesystem::temp_directory_path() /
                    (wrong_state_dimension ? "actionabi-pin-bad.jsonl"
                                           : "actionabi-pin-good.jsonl");
  std::ofstream output(path);
  output << nlohmann::json{{"record_type", "metadata"},
                           {"schema_version", "1.0"},
                           {"source_filename", "generated"},
                           {"source_sha256", std::string(64, 'a')},
                           {"extraction_date", "2026-07-18"},
                           {"state_columns", wrong_state_dimension
                                                 ? std::vector<std::string>{"q0"}
                                                 : std::vector<std::string>{"q0", "q1"}},
                           {"state_units", wrong_state_dimension
                                               ? std::vector<std::string>{"rad"}
                                               : std::vector<std::string>{"rad", "rad"}}}
         << '\n';
  for (int episode = 0; episode < 4; ++episode) {
    for (int row = 0; row < 4; ++row) {
      const double q0 = 0.05 * static_cast<double>(episode + row);
      const double q1 = -0.03 * static_cast<double>(episode + row);
      const auto current = tool_xy(q0, q1);
      const auto future = tool_xy(q0 + 0.05, q1 - 0.03);
      output << nlohmann::json{
                    {"record_type", "sample"},
                    {"episode_id", episode},
                    {"t_ns", static_cast<std::int64_t>(row) * 20'000'000},
                    {"state", wrong_state_dimension ? std::vector<double>{q0}
                                                     : std::vector<double>{q0, q1}},
                    {"action", row == 3
                                   ? std::vector<double>{0.0, 0.0}
                                   : std::vector<double>{future[0] - current[0],
                                                         future[1] - current[1]}},
                }
             << '\n';
    }
  }
  return path;
}

actionabi::ActionContract cartesian_delta() {
  return {.target = actionabi::TargetKind::DeltaPosition,
          .space = actionabi::ActionSpace::Cartesian,
          .frame = actionabi::ReferenceFrame::World,
          .permutation = {0, 1},
          .sign = {1, 1},
          .scale = {1.0, 1.0},
          .lag_steps = 0,
          .gripper_inverted = false};
}

}  // namespace

TEST_CASE("Pinocchio scores known Cartesian displacement") {
  const auto path = write_fixture();
  const auto dataset = actionabi::load_jsonl(path);
  const auto result = actionabi::score_cartesian_with_pinocchio(
      dataset, cartesian_delta(), ACTIONABI_TWO_LINK_URDF, "tool");
  std::filesystem::remove(path);
  REQUIRE(result.supported);
  REQUIRE(result.score.has_value());
  CHECK(result.score->heldout_loss < 1e-12);
}

TEST_CASE("Pinocchio fails closed on missing kinematic evidence") {
  const auto path = write_fixture(true);
  const auto dataset = actionabi::load_jsonl(path);
  const auto wrong_dimension = actionabi::score_cartesian_with_pinocchio(
      dataset, cartesian_delta(), ACTIONABI_TWO_LINK_URDF, "tool");
  const auto missing_frame = actionabi::score_cartesian_with_pinocchio(
      dataset, cartesian_delta(), ACTIONABI_TWO_LINK_URDF, "missing");
  std::filesystem::remove(path);
  CHECK_FALSE(wrong_dimension.supported);
  CHECK_FALSE(wrong_dimension.score.has_value());
  CHECK_FALSE(missing_frame.supported);
  CHECK_FALSE(missing_frame.score.has_value());
}
