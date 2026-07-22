#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <nlohmann/json.hpp>

#include "actionabi/contract.hpp"
#include "actionabi/score.hpp"
#include "actionabi/trajectory.hpp"

namespace {

class ScoringFixture {
 public:
  ScoringFixture(actionabi::TargetKind target, std::size_t episodes,
                 bool corrupt_holdout = false, double outlier = 0.0,
                 bool invert_gripper = false) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("actionabi-score-" + std::to_string(stamp) + ".jsonl");
    std::ofstream output(path_);
    output << nlohmann::json{
                  {"record_type", "metadata"},
                  {"schema_version", "1.0"},
                  {"source_filename", "synthetic.json"},
                  {"source_sha256", std::string(64, 'b')},
                  {"extraction_date", "2026-07-18"},
                  {"state_columns", {"q0", "q1"}},
                  {"state_units", {"rad", "rad"}},
              }
           << '\n';
    for (std::size_t episode = 0; episode < episodes; ++episode) {
      std::vector<double> state{0.1 * static_cast<double>(episode),
                                -0.05 * static_cast<double>(episode)};
      for (std::size_t step = 0; step < 6; ++step) {
        const std::vector<double> command{
            0.03 * static_cast<double>(step + 1),
            -0.02 * static_cast<double>(step + 1)};
        std::vector<double> raw{
            (invert_gripper ? -command[1] : command[1]) / 2.0,
            command[0] / 0.5};
        output << nlohmann::json{
                      {"record_type", "sample"},
                      {"episode_id", episode},
                      {"t_ns", static_cast<std::int64_t>(step * 100'000'000)},
                      {"state", state},
                      {"action", raw},
                  }
               << '\n';
        if (step == 5) {
          continue;
        }
        if (target == actionabi::TargetKind::AbsolutePosition) {
          state = command;
        } else if (target == actionabi::TargetKind::DeltaPosition) {
          state[0] += command[0];
          state[1] += command[1];
        } else {
          state[0] += command[0] * 0.1;
          state[1] += command[1] * 0.1;
        }
        if (corrupt_holdout && episode + 1 == episodes && step == 2) {
          state[0] += outlier == 0.0 ? 1.0 : outlier;
        }
      }
    }
  }

  ~ScoringFixture() { std::filesystem::remove(path_); }
  actionabi::TrajectoryDataset load() const { return actionabi::load_jsonl(path_); }

 private:
  std::filesystem::path path_;
};

actionabi::ActionContract contract(actionabi::TargetKind target) {
  return {
      .target = target,
      .space = actionabi::ActionSpace::Joint,
      .frame = actionabi::ReferenceFrame::Unspecified,
      .permutation = {1, 0},
      .sign = {1, 1},
      .scale = {0.5, 2.0},
      .lag_steps = 0,
      .gripper_inverted = false,
  };
}

}  // namespace

TEST_CASE("CPU scorer recovers absolute delta and velocity targets") {
  for (const auto target : {actionabi::TargetKind::AbsolutePosition,
                            actionabi::TargetKind::DeltaPosition,
                            actionabi::TargetKind::Velocity}) {
    const auto dataset = ScoringFixture(target, 4).load();
    const auto result = actionabi::score_cpu(dataset, contract(target), {});
    CAPTURE(static_cast<int>(target));
    CHECK(result.train_loss == Catch::Approx(0.0).margin(1e-14));
    CHECK(result.heldout_loss == Catch::Approx(0.0).margin(1e-14));
    CHECK(result.residual_count == 40);
    REQUIRE(result.per_dimension_loss.size() == 2);
  }
}

TEST_CASE("held-out corruption cannot leak into training loss") {
  const auto dataset =
      ScoringFixture(actionabi::TargetKind::DeltaPosition, 4, true).load();

  const auto result = actionabi::score_cpu(
      dataset, contract(actionabi::TargetKind::DeltaPosition), {});

  CHECK(result.train_loss == Catch::Approx(0.0).margin(1e-14));
  CHECK(result.heldout_loss > 0.0);
}

TEST_CASE("Huber loss limits a held-out outlier") {
  const auto dataset = ScoringFixture(actionabi::TargetKind::DeltaPosition, 4,
                                      true, 1000.0)
                           .load();
  actionabi::ScoreOptions options;
  options.huber_delta = 1.0;

  const auto result = actionabi::score_cpu(
      dataset, contract(actionabi::TargetKind::DeltaPosition), options);

  CHECK(result.heldout_loss > 40.0);
  CHECK(result.heldout_loss < 60.0);
}

TEST_CASE("lag truncation changes residual count without crossing episodes") {
  const auto dataset =
      ScoringFixture(actionabi::TargetKind::DeltaPosition, 4).load();
  auto delayed = contract(actionabi::TargetKind::DeltaPosition);
  delayed.lag_steps = 2;

  const auto result = actionabi::score_cpu(dataset, delayed, {});

  CHECK(result.residual_count == 24);
}

TEST_CASE("gripper inversion applies to the final semantic component") {
  const auto dataset = ScoringFixture(actionabi::TargetKind::DeltaPosition, 4,
                                      false, 0.0, true)
                           .load();
  auto inverted = contract(actionabi::TargetKind::DeltaPosition);
  inverted.gripper_inverted = true;

  const auto result = actionabi::score_cpu(dataset, inverted, {});

  CHECK(result.heldout_loss == Catch::Approx(0.0).margin(1e-14));
}

namespace {

// Writes a delta-target trace generated under the SINGLE-STEP-DELAYED physical
// response model: a command at action row ``r`` produces the one-step state
// transition at the delayed index (r+lag -> r+lag+1), i.e.
// state[r+lag+1] = state[r+lag] + decode(action[r]).  This is the model the
// bridge exposed, and the model the fixed scorer must score against. Consecutive
// commands are made to differ so that the wrong lag hypotheses are strictly
// worse than the true one.
class LaggedDeltaFixture {
 public:
  LaggedDeltaFixture(int lag, std::size_t episodes = 4, std::size_t rows = 9) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("actionabi-lag-" + std::to_string(stamp) + ".jsonl");
    const std::size_t offset = static_cast<std::size_t>(lag) + 1;
    std::ofstream output(path_);
    output << nlohmann::json{
                  {"record_type", "metadata"},
                  {"schema_version", "1.0"},
                  {"source_filename", "synthetic-lag.json"},
                  {"source_sha256", std::string(64, 'c')},
                  {"extraction_date", "2026-07-21"},
                  {"state_columns", {"q0", "q1"}},
                  {"state_units", {"rad", "rad"}},
              }
           << '\n';
    for (std::size_t episode = 0; episode < episodes; ++episode) {
      // Deterministic, row-varying raw actions.
      std::vector<std::vector<double>> raw(rows, std::vector<double>(2));
      for (std::size_t r = 0; r < rows; ++r) {
        raw[r][0] = 0.03 * static_cast<double>(r + 1) + 0.10 * static_cast<double>(episode);
        raw[r][1] = -0.02 * static_cast<double>(r + 1) - 0.05 * static_cast<double>(episode);
      }
      // True contract: perm {1,0}, sign {1,1}, scale {0.5, 2.0}.
      auto decode_true = [](const std::vector<double>& a) {
        return std::vector<double>{a[1] * 0.5, a[0] * 2.0};
      };
      // Build states so the true delta contract at this lag has zero residual.
      std::vector<std::vector<double>> state(rows, std::vector<double>(2, 0.0));
      state[0] = {0.11 + 0.01 * static_cast<double>(episode),
                  -0.07 - 0.02 * static_cast<double>(episode)};
      for (std::size_t i = 1; i < offset && i < rows; ++i) {
        state[i] = {state[i - 1][0] + 0.013, state[i - 1][1] - 0.017};
      }
      for (std::size_t r = 0; r + offset < rows; ++r) {
        const auto delta = decode_true(raw[r]);
        state[r + offset] = {state[r + offset - 1][0] + delta[0],
                             state[r + offset - 1][1] + delta[1]};
      }
      for (std::size_t r = 0; r < rows; ++r) {
        output << nlohmann::json{
                      {"record_type", "sample"},
                      {"episode_id", episode},
                      {"t_ns", static_cast<std::int64_t>(r * 100'000'000)},
                      {"state", state[r]},
                      {"action", raw[r]},
                  }
               << '\n';
      }
    }
  }
  ~LaggedDeltaFixture() { std::filesystem::remove(path_); }
  actionabi::TrajectoryDataset load() const { return actionabi::load_jsonl(path_); }

 private:
  std::filesystem::path path_;
};

actionabi::ActionContract lagged_contract(int lag) {
  return {
      .target = actionabi::TargetKind::DeltaPosition,
      .space = actionabi::ActionSpace::Joint,
      .frame = actionabi::ReferenceFrame::Unspecified,
      .permutation = {1, 0},
      .sign = {1, 1},
      .scale = {0.5, 2.0},
      .lag_steps = lag,
      .gripper_inverted = false,
  };
}

}  // namespace

TEST_CASE("lagged delta observable is single-step delayed, not a multi-step span") {
  // Regression test for the lag-observable scorer bug: the C++ delta observable
  // used to span lag+1 steps (state[row+lag+1] - state[row]) whereas the physical
  // response is a single-step delay (state[row+lag+1] - state[row+lag]). Under
  // the OLD span semantics the TRUE lagged contract carried a nonzero residual
  // (an extra decode term) and lost to wrong-lag hypotheses; both CHECKs below
  // FAIL on the old scorer and PASS with the fix.
  for (const int lag : {1, 2}) {
    const auto dataset = LaggedDeltaFixture(lag).load();
    const auto truth = actionabi::score_cpu(dataset, lagged_contract(lag), {});
    CAPTURE(lag);
    // (1) The true contract now fits the delayed one-step response exactly.
    CHECK(truth.heldout_loss == Catch::Approx(0.0).margin(1e-12));
    CHECK(truth.train_loss == Catch::Approx(0.0).margin(1e-12));
    // (2) The true lag strictly wins over every wrong lag (the bug barely
    //     recovered lag; here the correct lag must be uniquely lossless).
    for (const int other : {0, 1, 2}) {
      if (other == lag) {
        continue;
      }
      const auto rival = actionabi::score_cpu(dataset, lagged_contract(other), {});
      CAPTURE(other);
      CHECK(rival.heldout_loss > truth.heldout_loss + 1e-9);
    }
  }
}

TEST_CASE("scorer rejects dimension mismatch") {
  const auto dataset =
      ScoringFixture(actionabi::TargetKind::DeltaPosition, 4).load();
  auto wrong = contract(actionabi::TargetKind::DeltaPosition);
  wrong.permutation = {0};
  wrong.sign = {1};
  wrong.scale = {1.0};

  CHECK_THROWS_WITH(actionabi::score_cpu(dataset, wrong, {}),
                    "contract dimension must match state and action dimensions");
}
