#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "actionabi/trajectory.hpp"

namespace {

class TemporaryJsonl {
 public:
  explicit TemporaryJsonl(const std::string& contents) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("actionabi-trajectory-" + std::to_string(stamp) + ".jsonl");
    std::ofstream output(path_);
    output << contents;
  }

  ~TemporaryJsonl() { std::filesystem::remove(path_); }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

const std::string metadata =
    R"({"record_type":"metadata","schema_version":"1.0","source_filename":"source.parquet","source_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","extraction_date":"2026-07-18","state_columns":["q0","q1"],"state_units":["rad","rad"]})";

}  // namespace

TEST_CASE("trajectory loader preserves provenance and contiguous episode spans") {
  TemporaryJsonl input(
      metadata + "\n" +
      R"({"record_type":"sample","episode_id":2,"t_ns":0,"state":[0.0,1.0],"action":[0.1,0.2]})" +
      "\n" +
      R"({"record_type":"sample","episode_id":2,"t_ns":10,"state":[0.1,1.1],"action":[0.2,0.3]})" +
      "\n" +
      R"({"record_type":"sample","episode_id":5,"t_ns":0,"state":[2.0,3.0],"action":[0.4,0.5]})" +
      "\n");

  const auto dataset = actionabi::load_jsonl(input.path());

  CHECK(dataset.row_count() == 3);
  CHECK(dataset.state_dimension() == 2);
  CHECK(dataset.action_dimension() == 2);
  CHECK(dataset.episodes().size() == 2);
  CHECK(dataset.episodes()[0] == actionabi::EpisodeSpan{2, 0, 2});
  CHECK(dataset.episodes()[1] == actionabi::EpisodeSpan{5, 2, 3});
  CHECK(dataset.provenance().source_filename == "source.parquet");
  CHECK(dataset.state(1)[0] == 0.1);
  CHECK(dataset.action(2)[1] == 0.5);
}

TEST_CASE("trajectory rejects nonmonotonic timestamps with episode and row") {
  TemporaryJsonl input(
      metadata + "\n" +
      R"({"record_type":"sample","episode_id":2,"t_ns":10,"state":[0,1],"action":[0,1]})" +
      "\n" +
      R"({"record_type":"sample","episode_id":2,"t_ns":10,"state":[0,1],"action":[0,1]})" +
      "\n");

  CHECK_THROWS_WITH(actionabi::load_jsonl(input.path()),
                    "episode 2 row 1: timestamps must be strictly increasing");
}

TEST_CASE("trajectory rejects changing action dimension") {
  TemporaryJsonl input(
      metadata + "\n" +
      R"({"record_type":"sample","episode_id":2,"t_ns":0,"state":[0,1],"action":[0,1]})" +
      "\n" +
      R"({"record_type":"sample","episode_id":2,"t_ns":1,"state":[0,1],"action":[0]})" +
      "\n");

  CHECK_THROWS_WITH(actionabi::load_jsonl(input.path()),
                    "episode 2 row 1: action dimension changed from 2 to 1");
}

TEST_CASE("trajectory rejects missing provenance") {
  TemporaryJsonl input(
      std::string{
          R"({"record_type":"metadata","schema_version":"1.0","state_columns":["q0"],"state_units":["rad"]})"} +
      "\n" +
      R"({"record_type":"sample","episode_id":0,"t_ns":0,"state":[0],"action":[0]})" +
      "\n");

  CHECK_THROWS_WITH(actionabi::load_jsonl(input.path()),
                    "metadata is missing provenance field: source_filename");
}

TEST_CASE("trajectory rejects noncontiguous repeated episodes") {
  TemporaryJsonl input(
      metadata + "\n" +
      R"({"record_type":"sample","episode_id":2,"t_ns":0,"state":[0,1],"action":[0,1]})" +
      "\n" +
      R"({"record_type":"sample","episode_id":5,"t_ns":0,"state":[0,1],"action":[0,1]})" +
      "\n" +
      R"({"record_type":"sample","episode_id":2,"t_ns":1,"state":[0,1],"action":[0,1]})" +
      "\n");

  CHECK_THROWS_WITH(actionabi::load_jsonl(input.path()),
                    "episode 2 row 2: episode rows must be contiguous");
}
