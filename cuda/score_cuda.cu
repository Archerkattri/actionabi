#include "actionabi/score_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace actionabi {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
               "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t bytes() const { return count_ * sizeof(T); }

 private:
  T* data_{nullptr};
  std::size_t count_;
};

class CudaEvent {
 public:
  CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~CudaEvent() { cudaEventDestroy(event_); }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_{};
};

template <typename T>
void upload(DeviceBuffer<T>& destination, const std::vector<T>& source) {
  check_cuda(cudaMemcpy(destination.data(), source.data(), destination.bytes(),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy host to device");
}

__device__ double huber_device(double residual, double delta) {
  const double magnitude = fabs(residual);
  return magnitude <= delta ? 0.5 * residual * residual
                            : delta * (magnitude - 0.5 * delta);
}

__global__ void score_kernel(
    const double* states, const double* actions, const std::int64_t* timestamps,
    const std::size_t* episode_end, const std::size_t* episode_order,
    std::size_t row_count, std::size_t dimension, std::size_t train_episodes,
    const int* targets, const std::uint32_t* permutations,
    const std::int8_t* signs, const double* scales, const std::int32_t* lags,
    const std::uint8_t* gripper_inverted, std::size_t hypothesis_count,
    double huber_delta, double* train_sums, double* heldout_sums,
    unsigned long long* train_counts, unsigned long long* heldout_counts,
    double* heldout_dimension_sums,
    unsigned long long* heldout_dimension_counts) {
  const std::size_t linear =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = hypothesis_count * row_count * dimension;
  if (linear >= total) {
    return;
  }
  const std::size_t component = linear % dimension;
  const std::size_t row_and_hypothesis = linear / dimension;
  const std::size_t row = row_and_hypothesis % row_count;
  const std::size_t hypothesis = row_and_hypothesis / row_count;
  const std::size_t offset = static_cast<std::size_t>(lags[hypothesis]) + 1;
  if (row + offset >= episode_end[row]) {
    return;
  }
  const std::size_t contract_index = hypothesis * dimension + component;
  const std::size_t raw_component = permutations[contract_index];
  double command = actions[row * dimension + raw_component] *
                   static_cast<double>(signs[contract_index]) *
                   scales[contract_index];
  if (gripper_inverted[hypothesis] != 0 && component + 1 == dimension) {
    command *= -1.0;
  }
  // Single-step delayed observable (see src/score_cpu.cpp): a lagged command
  // explains the one-step transition at the delayed index row+offset-1 ->
  // row+offset, not the multi-step span row -> row+offset. Reduces to the old
  // behaviour at lag == 0 (offset == 1); kept bit-for-bit identical to the CPU
  // scorer for the parity test.
  const double delayed_prev = states[(row + offset - 1) * dimension + component];
  const double delayed_next = states[(row + offset) * dimension + component];
  double observable = delayed_next;
  if (targets[hypothesis] == 1) {
    observable = delayed_next - delayed_prev;
  } else if (targets[hypothesis] == 2) {
    const auto elapsed_ns = timestamps[row + offset] - timestamps[row + offset - 1];
    observable = (delayed_next - delayed_prev) /
                 (static_cast<double>(elapsed_ns) * 1e-9);
  }
  const double loss = huber_device(command - observable, huber_delta);
  if (episode_order[row] < train_episodes) {
    atomicAdd(&train_sums[hypothesis], loss);
    atomicAdd(&train_counts[hypothesis], 1ULL);
  } else {
    atomicAdd(&heldout_sums[hypothesis], loss);
    atomicAdd(&heldout_counts[hypothesis], 1ULL);
    atomicAdd(&heldout_dimension_sums[contract_index], loss);
    atomicAdd(&heldout_dimension_counts[contract_index], 1ULL);
  }
}

int target_code(TargetKind target) {
  switch (target) {
    case TargetKind::AbsolutePosition:
      return 0;
    case TargetKind::DeltaPosition:
      return 1;
    case TargetKind::Velocity:
      return 2;
  }
  throw std::invalid_argument("invalid target enum");
}

}  // namespace

std::vector<ScoreResult> score_cuda_batch(
    const TrajectoryDataset& dataset,
    const std::vector<ActionContract>& contracts,
    const ScoreOptions& options,
    CudaTiming* timing) {
  if (contracts.empty()) {
    throw std::invalid_argument("CUDA scoring requires at least one contract");
  }
  if (!std::isfinite(options.huber_delta) || options.huber_delta <= 0.0) {
    throw std::invalid_argument("huber_delta must be finite and positive");
  }
  if (!std::isfinite(options.train_episode_fraction) ||
      options.train_episode_fraction <= 0.0 ||
      options.train_episode_fraction >= 1.0) {
    throw std::invalid_argument(
        "train_episode_fraction must be between zero and one");
  }
  if (dataset.episodes().size() < 2) {
    throw std::invalid_argument("held-out scoring requires at least two episodes");
  }
  const std::size_t dimension = contracts.front().permutation.size();
  for (const auto& contract : contracts) {
    validate(contract);
    if (contract.permutation.size() != dimension ||
        dimension != dataset.state_dimension() ||
        dimension != dataset.action_dimension()) {
      throw std::invalid_argument(
          "all CUDA contracts must match state and action dimensions");
    }
  }
  const std::size_t hypothesis_count = contracts.size();
  const std::size_t row_count = dataset.row_count();
  const std::size_t train_episodes = std::clamp<std::size_t>(
      static_cast<std::size_t>(std::floor(
          options.train_episode_fraction * dataset.episodes().size())),
      1, dataset.episodes().size() - 1);

  std::vector<std::size_t> episode_end(row_count);
  std::vector<std::size_t> episode_order(row_count);
  for (std::size_t index = 0; index < dataset.episodes().size(); ++index) {
    const auto& episode = dataset.episodes()[index];
    for (std::size_t row = episode.begin; row < episode.end; ++row) {
      episode_end[row] = episode.end;
      episode_order[row] = index;
    }
  }
  std::vector<int> targets;
  std::vector<std::uint32_t> permutations;
  std::vector<std::int8_t> signs;
  std::vector<double> scales;
  std::vector<std::int32_t> lags;
  std::vector<std::uint8_t> gripper_inverted;
  targets.reserve(hypothesis_count);
  permutations.reserve(hypothesis_count * dimension);
  signs.reserve(hypothesis_count * dimension);
  scales.reserve(hypothesis_count * dimension);
  lags.reserve(hypothesis_count);
  gripper_inverted.reserve(hypothesis_count);
  for (const auto& contract : contracts) {
    targets.push_back(target_code(contract.target));
    permutations.insert(permutations.end(), contract.permutation.begin(),
                        contract.permutation.end());
    signs.insert(signs.end(), contract.sign.begin(), contract.sign.end());
    scales.insert(scales.end(), contract.scale.begin(), contract.scale.end());
    lags.push_back(contract.lag_steps);
    gripper_inverted.push_back(contract.gripper_inverted ? 1 : 0);
  }

  DeviceBuffer<double> d_states(dataset.states_flat().size());
  DeviceBuffer<double> d_actions(dataset.actions_flat().size());
  DeviceBuffer<std::int64_t> d_timestamps(dataset.timestamps_ns().size());
  DeviceBuffer<std::size_t> d_episode_end(row_count);
  DeviceBuffer<std::size_t> d_episode_order(row_count);
  DeviceBuffer<int> d_targets(targets.size());
  DeviceBuffer<std::uint32_t> d_permutations(permutations.size());
  DeviceBuffer<std::int8_t> d_signs(signs.size());
  DeviceBuffer<double> d_scales(scales.size());
  DeviceBuffer<std::int32_t> d_lags(lags.size());
  DeviceBuffer<std::uint8_t> d_gripper(gripper_inverted.size());
  DeviceBuffer<double> d_train_sums(hypothesis_count);
  DeviceBuffer<double> d_heldout_sums(hypothesis_count);
  DeviceBuffer<unsigned long long> d_train_counts(hypothesis_count);
  DeviceBuffer<unsigned long long> d_heldout_counts(hypothesis_count);
  DeviceBuffer<double> d_dimension_sums(hypothesis_count * dimension);
  DeviceBuffer<unsigned long long> d_dimension_counts(hypothesis_count * dimension);

  upload(d_states, dataset.states_flat());
  upload(d_actions, dataset.actions_flat());
  upload(d_timestamps, dataset.timestamps_ns());
  upload(d_episode_end, episode_end);
  upload(d_episode_order, episode_order);
  upload(d_targets, targets);
  upload(d_permutations, permutations);
  upload(d_signs, signs);
  upload(d_scales, scales);
  upload(d_lags, lags);
  upload(d_gripper, gripper_inverted);
  check_cuda(cudaMemset(d_train_sums.data(), 0, d_train_sums.bytes()),
             "cudaMemset train sums");
  check_cuda(cudaMemset(d_heldout_sums.data(), 0, d_heldout_sums.bytes()),
             "cudaMemset heldout sums");
  check_cuda(cudaMemset(d_train_counts.data(), 0, d_train_counts.bytes()),
             "cudaMemset train counts");
  check_cuda(cudaMemset(d_heldout_counts.data(), 0, d_heldout_counts.bytes()),
             "cudaMemset heldout counts");
  check_cuda(cudaMemset(d_dimension_sums.data(), 0, d_dimension_sums.bytes()),
             "cudaMemset dimension sums");
  check_cuda(cudaMemset(d_dimension_counts.data(), 0, d_dimension_counts.bytes()),
             "cudaMemset dimension counts");

  constexpr std::size_t threads = 256;
  const std::size_t total = hypothesis_count * row_count * dimension;
  const auto blocks = static_cast<unsigned int>((total + threads - 1) / threads);
  CudaEvent kernel_start;
  CudaEvent kernel_end;
  check_cuda(cudaEventRecord(kernel_start.get()), "cudaEventRecord kernel start");
  score_kernel<<<blocks, threads>>>(
      d_states.data(), d_actions.data(), d_timestamps.data(),
      d_episode_end.data(), d_episode_order.data(), row_count, dimension,
      train_episodes, d_targets.data(), d_permutations.data(), d_signs.data(),
      d_scales.data(), d_lags.data(), d_gripper.data(), hypothesis_count,
      options.huber_delta, d_train_sums.data(), d_heldout_sums.data(),
      d_train_counts.data(), d_heldout_counts.data(), d_dimension_sums.data(),
      d_dimension_counts.data());
  check_cuda(cudaGetLastError(), "score kernel launch");
  check_cuda(cudaEventRecord(kernel_end.get()), "cudaEventRecord kernel end");
  check_cuda(cudaEventSynchronize(kernel_end.get()), "score kernel execution");
  if (timing != nullptr) {
    float elapsed = 0.0F;
    check_cuda(cudaEventElapsedTime(&elapsed, kernel_start.get(), kernel_end.get()),
               "cudaEventElapsedTime");
    timing->kernel_milliseconds = static_cast<double>(elapsed);
  }

  std::vector<double> train_sums(hypothesis_count);
  std::vector<double> heldout_sums(hypothesis_count);
  std::vector<unsigned long long> train_counts(hypothesis_count);
  std::vector<unsigned long long> heldout_counts(hypothesis_count);
  std::vector<double> dimension_sums(hypothesis_count * dimension);
  std::vector<unsigned long long> dimension_counts(hypothesis_count * dimension);
  check_cuda(cudaMemcpy(train_sums.data(), d_train_sums.data(),
                        d_train_sums.bytes(), cudaMemcpyDeviceToHost),
             "cudaMemcpy train sums");
  check_cuda(cudaMemcpy(heldout_sums.data(), d_heldout_sums.data(),
                        d_heldout_sums.bytes(), cudaMemcpyDeviceToHost),
             "cudaMemcpy heldout sums");
  check_cuda(cudaMemcpy(train_counts.data(), d_train_counts.data(),
                        d_train_counts.bytes(), cudaMemcpyDeviceToHost),
             "cudaMemcpy train counts");
  check_cuda(cudaMemcpy(heldout_counts.data(), d_heldout_counts.data(),
                        d_heldout_counts.bytes(), cudaMemcpyDeviceToHost),
             "cudaMemcpy heldout counts");
  check_cuda(cudaMemcpy(dimension_sums.data(), d_dimension_sums.data(),
                        d_dimension_sums.bytes(), cudaMemcpyDeviceToHost),
             "cudaMemcpy dimension sums");
  check_cuda(cudaMemcpy(dimension_counts.data(), d_dimension_counts.data(),
                        d_dimension_counts.bytes(), cudaMemcpyDeviceToHost),
             "cudaMemcpy dimension counts");

  std::vector<ScoreResult> results;
  results.reserve(hypothesis_count);
  for (std::size_t hypothesis = 0; hypothesis < hypothesis_count; ++hypothesis) {
    if (train_counts[hypothesis] == 0 || heldout_counts[hypothesis] == 0) {
      throw std::invalid_argument(
          "contract lag leaves no train or held-out residuals");
    }
    std::vector<double> per_dimension(dimension);
    for (std::size_t component = 0; component < dimension; ++component) {
      const auto index = hypothesis * dimension + component;
      per_dimension[component] =
          dimension_sums[index] / static_cast<double>(dimension_counts[index]);
    }
    results.push_back({
        train_sums[hypothesis] / static_cast<double>(train_counts[hypothesis]),
        heldout_sums[hypothesis] /
            static_cast<double>(heldout_counts[hypothesis]),
        train_counts[hypothesis] + heldout_counts[hypothesis],
        std::move(per_dimension),
    });
  }
  return results;
}

}  // namespace actionabi
