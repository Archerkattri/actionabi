#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "actionabi/contract.hpp"
#include "actionabi/score.hpp"
#include "actionabi/score_cuda.hpp"
#include "actionabi/trajectory.hpp"

namespace {

class BenchmarkDataset {
 public:
  BenchmarkDataset(std::size_t rows_per_episode, std::size_t dimension) {
    path_ = std::filesystem::temp_directory_path() /
            ("actionabi-benchmark-" + std::to_string(rows_per_episode) + ".jsonl");
    std::ofstream output(path_);
    std::vector<std::string> state_columns;
    std::vector<std::string> state_units;
    for (std::size_t component = 0; component < dimension; ++component) {
      state_columns.push_back("q" + std::to_string(component));
      state_units.push_back("rad");
    }
    output << nlohmann::json{
                  {"record_type", "metadata"},
                  {"schema_version", "1.0"},
                  {"source_filename", "benchmark.json"},
                  {"source_sha256", std::string(64, 'd')},
                  {"extraction_date", "2026-07-18"},
                  {"state_columns", state_columns},
                  {"state_units", state_units},
              }
           << '\n';
    for (std::int64_t episode = 0; episode < 4; ++episode) {
      std::vector<double> state(dimension, 0.0);
      for (std::size_t row = 0; row < rows_per_episode; ++row) {
        std::vector<double> action(dimension);
        for (std::size_t component = 0; component < dimension; ++component) {
          const double sign = component % 2 == 0 ? 1.0 : -1.0;
          action[component] = sign * 0.001 * static_cast<double>(component + 1) *
                              static_cast<double>(row + 1);
        }
        output << nlohmann::json{
                      {"record_type", "sample"},
                      {"episode_id", episode},
                      {"t_ns", static_cast<std::int64_t>(row * 20'000'000)},
                      {"state", state},
                      {"action", action},
                  }
               << '\n';
        for (std::size_t component = 0; component < dimension; ++component) {
          state[component] += action[component];
        }
      }
    }
  }
  ~BenchmarkDataset() { std::filesystem::remove(path_); }
  actionabi::TrajectoryDataset load() const { return actionabi::load_jsonl(path_); }

 private:
  std::filesystem::path path_;
};

std::vector<actionabi::ActionContract> make_contracts(std::size_t count,
                                                       std::size_t dimension) {
  std::vector<actionabi::ActionContract> contracts;
  contracts.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const double scale = 0.5 + static_cast<double>(index % 17) / 16.0;
    std::vector<std::uint32_t> permutation(dimension);
    std::vector<std::int8_t> sign(dimension, std::int8_t{1});
    std::vector<double> scales(dimension, scale);
    for (std::size_t component = 0; component < dimension; ++component) {
      permutation[component] = static_cast<std::uint32_t>(component);
    }
    if (dimension > 1 && index % 2 != 0) {
      sign.back() = std::int8_t{-1};
    }
    contracts.push_back({
        .target = index % 3 == 0 ? actionabi::TargetKind::DeltaPosition
                                 : actionabi::TargetKind::Velocity,
        .space = actionabi::ActionSpace::Joint,
        .frame = actionabi::ReferenceFrame::Unspecified,
        .permutation = std::move(permutation),
        .sign = std::move(sign),
        .scale = std::move(scales),
        .lag_steps = static_cast<std::int32_t>(index % 3),
        .gripper_inverted = false,
    });
  }
  return contracts;
}

std::vector<actionabi::ScoreResult> score_multicore(
    const actionabi::TrajectoryDataset& dataset,
    const std::vector<actionabi::ActionContract>& contracts,
    unsigned int thread_count) {
  std::vector<actionabi::ScoreResult> results(contracts.size());
  std::atomic<std::size_t> next{0};
  std::vector<std::thread> workers;
  workers.reserve(thread_count);
  for (unsigned int worker = 0; worker < thread_count; ++worker) {
    workers.emplace_back([&]() {
      while (true) {
        const auto index = next.fetch_add(1);
        if (index >= contracts.size()) {
          break;
        }
        results[index] = actionabi::score_cpu(dataset, contracts[index], {});
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  return results;
}

std::vector<actionabi::ScoreResult> score_single_thread(
    const actionabi::TrajectoryDataset& dataset,
    const std::vector<actionabi::ActionContract>& contracts) {
  std::vector<actionabi::ScoreResult> results;
  results.reserve(contracts.size());
  for (const auto& contract : contracts) {
    results.push_back(actionabi::score_cpu(dataset, contract, {}));
  }
  return results;
}

double time_milliseconds(const std::function<void()>& operation) {
  const auto start = std::chrono::steady_clock::now();
  operation();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

nlohmann::json distribution(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto at = [&](double quantile) {
    const auto index = static_cast<std::size_t>(
        quantile * static_cast<double>(values.size() - 1));
    return values[index];
  };
  return {{"median", at(0.5)}, {"p10", at(0.1)}, {"p90", at(0.9)}};
}

std::string cpu_model() {
  std::ifstream input("/proc/cpuinfo");
  std::string line;
  while (std::getline(input, line)) {
    constexpr const char* prefix = "model name";
    if (line.starts_with(prefix)) {
      const auto separator = line.find(':');
      return separator == std::string::npos ? line : line.substr(separator + 2);
    }
  }
  return "unknown";
}

std::size_t estimated_peak_device_bytes(
    const actionabi::TrajectoryDataset& dataset, std::size_t hypotheses,
    std::size_t dimension) {
  const auto rows = dataset.row_count();
  const auto dataset_bytes = dataset.states_flat().size() * sizeof(double) +
                             dataset.actions_flat().size() * sizeof(double) +
                             rows * sizeof(std::int64_t) +
                             2 * rows * sizeof(std::size_t);
  const auto contract_bytes =
      hypotheses * (sizeof(int) + sizeof(std::int32_t) + sizeof(std::uint8_t)) +
      hypotheses * dimension *
          (sizeof(std::uint32_t) + sizeof(std::int8_t) + sizeof(double));
  const auto accumulator_bytes =
      hypotheses * (2 * sizeof(double) + 2 * sizeof(unsigned long long)) +
      hypotheses * dimension *
          (sizeof(double) + sizeof(unsigned long long));
  return dataset_bytes + contract_bytes + accumulator_bytes;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"Transfer-inclusive ActionABI backend benchmark"};
  std::filesystem::path output_path;
  std::size_t hypotheses = 128;
  std::size_t total_evaluations = 10'000'000;
  std::size_t warmups = 5;
  std::size_t measurements = 30;
  std::size_t dimension = 2;
  std::size_t rows_per_episode = 0;
  unsigned int cpu_threads = std::max(1U, std::thread::hardware_concurrency());
  app.add_option("--output", output_path)->required();
  app.add_option("--hypotheses", hypotheses);
  app.add_option("--total-evaluations", total_evaluations);
  app.add_option("--warmups", warmups);
  app.add_option("--measurements", measurements);
  app.add_option("--dimension", dimension);
  app.add_option("--rows-per-episode", rows_per_episode);
  app.add_option("--cpu-threads", cpu_threads);
  CLI11_PARSE(app, argc, argv);
  if (hypotheses == 0 || measurements == 0 || dimension == 0 || cpu_threads == 0) {
    throw std::invalid_argument(
        "hypotheses, measurements, dimension, and cpu threads must be positive");
  }
  if (rows_per_episode == 0) {
    rows_per_episode = std::max<std::size_t>(
        4, total_evaluations / (hypotheses * dimension * 4) + 2);
  }
  const auto dataset = BenchmarkDataset(rows_per_episode, dimension).load();
  const auto contracts = make_contracts(hypotheses, dimension);
  std::vector<actionabi::ScoreResult> sink;
  actionabi::CudaTiming cuda_timing;
  for (std::size_t iteration = 0; iteration < warmups; ++iteration) {
    sink = score_single_thread(dataset, contracts);
    sink = score_multicore(dataset, contracts, cpu_threads);
    sink = actionabi::score_cuda_batch(dataset, contracts, {}, &cuda_timing);
  }
  std::vector<double> single_thread_times;
  std::vector<double> cpu_times;
  std::vector<double> cuda_times;
  std::vector<double> cuda_kernel_times;
  for (std::size_t iteration = 0; iteration < measurements; ++iteration) {
    single_thread_times.push_back(time_milliseconds(
        [&]() { sink = score_single_thread(dataset, contracts); }));
    cpu_times.push_back(time_milliseconds(
        [&]() { sink = score_multicore(dataset, contracts, cpu_threads); }));
    cuda_times.push_back(time_milliseconds(
        [&]() {
          sink = actionabi::score_cuda_batch(dataset, contracts, {},
                                             &cuda_timing);
        }));
    cuda_kernel_times.push_back(cuda_timing.kernel_milliseconds);
  }
  const auto single_thread_distribution = distribution(single_thread_times);
  const auto cpu_distribution = distribution(cpu_times);
  const auto cuda_distribution = distribution(cuda_times);
  const auto cuda_kernel_distribution = distribution(cuda_kernel_times);
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
    throw std::runtime_error("could not query CUDA device properties");
  }
  nlohmann::json report{
      {"schema_version", "1.0"},
      {"gpu", properties.name},
      {"cpu", cpu_model()},
      {"compiler", ACTIONABI_COMPILER},
      {"build_type", ACTIONABI_BUILD_TYPE},
      {"cpu_threads", cpu_threads},
      {"workloads",
       nlohmann::json::array({
           {
               {"hypotheses", hypotheses},
               {"dimension", dimension},
               {"rows_per_episode", rows_per_episode},
               {"cpu_threads", cpu_threads},
               {"rows", dataset.row_count()},
               {"residual_evaluations",
                hypotheses * dataset.row_count() * dimension},
               {"warmups", warmups},
               {"measurements", measurements},
               {"single_thread_cpu_ms", single_thread_distribution},
               {"multicore_cpu_ms", cpu_distribution},
               {"cuda_transfer_inclusive_ms", cuda_distribution},
               {"cuda_kernel_only_ms", cuda_kernel_distribution},
               {"single_thread_cpu_samples_ms", single_thread_times},
               {"multicore_cpu_samples_ms", cpu_times},
               {"cuda_transfer_samples_ms", cuda_times},
               {"cuda_kernel_samples_ms", cuda_kernel_times},
               {"estimated_peak_device_bytes",
                estimated_peak_device_bytes(dataset, hypotheses, dimension)},
               {"speedup_over_multicore_cpu",
                cpu_distribution.at("median").get<double>() /
                    cuda_distribution.at("median").get<double>()},
           },
       })},
  };
  std::ofstream output(output_path);
  output << report.dump(2) << '\n';
  return sink.empty() ? 1 : 0;
}
