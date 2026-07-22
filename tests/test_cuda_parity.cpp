#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "actionabi/contract.hpp"
#include "actionabi/score.hpp"
#include "actionabi/score_cuda.hpp"
#include "actionabi/trajectory.hpp"

namespace {

class CudaFixture {
 public:
  CudaFixture() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("actionabi-cuda-" + std::to_string(stamp) + ".jsonl");
    std::ofstream output(path_);
    output << nlohmann::json{
                  {"record_type", "metadata"},
                  {"schema_version", "1.0"},
                  {"source_filename", "synthetic.json"},
                  {"source_sha256", std::string(64, 'c')},
                  {"extraction_date", "2026-07-18"},
                  {"state_columns", {"q0", "q1"}},
                  {"state_units", {"rad", "rad"}},
              }
           << '\n';
    for (std::int64_t episode = 0; episode < 6; ++episode) {
      std::vector<double> state{0.0, 0.0};
      for (std::int64_t step = 0; step < 20; ++step) {
        const std::vector<double> action{
            0.01 * static_cast<double>(step + 1),
            -0.02 * static_cast<double>(step + 1)};
        output << nlohmann::json{
                      {"record_type", "sample"},
                      {"episode_id", episode},
                      {"t_ns", step * 20'000'000},
                      {"state", state},
                      {"action", action},
                  }
               << '\n';
        state[0] += action[0];
        state[1] += action[1];
      }
    }
  }
  ~CudaFixture() { std::filesystem::remove(path_); }
  actionabi::TrajectoryDataset load() const { return actionabi::load_jsonl(path_); }

 private:
  std::filesystem::path path_;
};

actionabi::ActionContract contract(actionabi::TargetKind target, int lag,
                                   double scale) {
  return {
      .target = target,
      .space = actionabi::ActionSpace::Joint,
      .frame = actionabi::ReferenceFrame::Unspecified,
      .permutation = {0, 1},
      .sign = {1, 1},
      .scale = {scale, scale},
      .lag_steps = lag,
      .gripper_inverted = false,
  };
}

}  // namespace

TEST_CASE("CUDA batched scorer matches CPU reference") {
  const auto dataset = CudaFixture().load();
  const std::vector<actionabi::ActionContract> contracts = {
      contract(actionabi::TargetKind::DeltaPosition, 0, 1.0),
      contract(actionabi::TargetKind::DeltaPosition, 1, 1.0),
      contract(actionabi::TargetKind::Velocity, 0, 1.0),
      contract(actionabi::TargetKind::DeltaPosition, 0, 2.0),
  };

  const auto gpu = actionabi::score_cuda_batch(dataset, contracts, {});

  REQUIRE(gpu.size() == contracts.size());
  for (std::size_t index = 0; index < contracts.size(); ++index) {
    const auto cpu = actionabi::score_cpu(dataset, contracts[index], {});
    CAPTURE(index);
    CHECK(gpu[index].train_loss == Catch::Approx(cpu.train_loss).epsilon(1e-10));
    CHECK(gpu[index].heldout_loss ==
          Catch::Approx(cpu.heldout_loss).epsilon(1e-10));
    CHECK(gpu[index].residual_count == cpu.residual_count);
    REQUIRE(gpu[index].per_dimension_loss.size() ==
            cpu.per_dimension_loss.size());
    for (std::size_t component = 0;
         component < cpu.per_dimension_loss.size(); ++component) {
      CHECK(gpu[index].per_dimension_loss[component] ==
            Catch::Approx(cpu.per_dimension_loss[component]).epsilon(1e-10));
    }
  }
}
