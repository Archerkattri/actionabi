cmake_minimum_required(VERSION 3.26)

project(ActionABI VERSION 0.1.0 LANGUAGES CXX)

include(cmake/ActionABIOptions.cmake)
include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

FetchContent_Declare(
  cli11
  URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.5.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
  catch2
  URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.8.1.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(cli11 catch2 nlohmann_json)

execute_process(
  COMMAND git rev-parse --short=12 HEAD
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE ACTIONABI_GIT_REVISION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT ACTIONABI_GIT_REVISION)
  set(ACTIONABI_GIT_REVISION "unknown")
endif()

add_library(actionabi_core src/contract.cpp src/trajectory.cpp src/score_cpu.cpp
             src/evidence.cpp src/identifiability.cpp src/probe.cpp
             src/converter.cpp)
target_include_directories(actionabi_core PUBLIC include)
target_link_libraries(actionabi_core PUBLIC nlohmann_json::nlohmann_json)

if(ACTIONABI_ENABLE_PINOCCHIO)
  find_package(pinocchio 3 REQUIRED)
  target_sources(actionabi_core PRIVATE src/score_pinocchio.cpp)
  target_link_libraries(actionabi_core PUBLIC pinocchio::pinocchio)
  target_compile_definitions(actionabi_core PUBLIC ACTIONABI_HAS_PINOCCHIO=1)
endif()

add_executable(actionabi src/cli/main.cpp)
target_link_libraries(actionabi PRIVATE actionabi_core CLI11::CLI11)
target_compile_definitions(
  actionabi PRIVATE
  ACTIONABI_VERSION="${PROJECT_VERSION}"
  ACTIONABI_GIT_REVISION="${ACTIONABI_GIT_REVISION}"
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(actionabi_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_compile_options(actionabi PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(ACTIONABI_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  add_library(actionabi_cuda cuda/score_cuda.cu)
  target_include_directories(actionabi_cuda PUBLIC include)
  target_link_libraries(actionabi_cuda PUBLIC actionabi_core)
  set_target_properties(actionabi_cuda PROPERTIES
                        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
endif()

if(ACTIONABI_BUILD_BENCHMARKS AND ACTIONABI_ENABLE_CUDA)
  add_executable(score_benchmark benchmarks/score_benchmark.cpp)
  target_link_libraries(score_benchmark PRIVATE actionabi_cuda CLI11::CLI11 CUDA::cudart)
  target_compile_definitions(
    score_benchmark PRIVATE
    ACTIONABI_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    ACTIONABI_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  )
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(score_benchmark PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endif()

if(ACTIONABI_BUILD_PYBIND)
  # The ActionShift-facing evidence-scoring backend. Built as a Python extension
  # named ``actionabi_cells``; pybind11 is discovered via its installed CMake
  # config (``pybind11.get_cmake_dir()``) or FetchContent as a fallback.
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    FetchContent_Declare(
      pybind11
      URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(pybind11)
  endif()
  pybind11_add_module(actionabi_cells bindings/module.cpp)
  target_include_directories(actionabi_cells PRIVATE bindings)
  if(ACTIONABI_CELLS_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 120)
    endif()
    target_sources(actionabi_cells PRIVATE bindings/cell_score_cuda.cu)
    target_link_libraries(actionabi_cells PRIVATE CUDA::cudart)
    target_compile_definitions(actionabi_cells PRIVATE ACTIONABI_CELLS_CUDA=1)
    set_target_properties(actionabi_cells PROPERTIES
                          CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  endif()
endif()

if(ACTIONABI_BUILD_TESTS)
  enable_testing()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(
    NAME cli_version
    COMMAND "${Python3_EXECUTABLE}" -m unittest tests.test_cli -v
  )
  set_tests_properties(
    cli_version PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  add_executable(test_contract tests/test_contract.cpp)
  target_link_libraries(test_contract PRIVATE actionabi_core Catch2::Catch2WithMain)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_contract PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME contract COMMAND test_contract)
  add_executable(test_trajectory tests/test_trajectory.cpp)
  target_link_libraries(test_trajectory PRIVATE actionabi_core Catch2::Catch2WithMain)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_trajectory PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME trajectory COMMAND test_trajectory)
  add_executable(test_score tests/test_score.cpp)
  target_link_libraries(test_score PRIVATE actionabi_core Catch2::Catch2WithMain)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_score PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME score COMMAND test_score)
  add_executable(test_cell_score tests/test_cell_score.cpp)
  target_link_libraries(test_cell_score PRIVATE Catch2::Catch2WithMain)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_cell_score PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME cell_score COMMAND test_cell_score)
  add_executable(test_evidence tests/test_evidence.cpp)
  target_link_libraries(test_evidence PRIVATE actionabi_core Catch2::Catch2WithMain)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_evidence PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME evidence COMMAND test_evidence)
  add_executable(test_identifiability tests/test_identifiability.cpp)
  target_link_libraries(test_identifiability PRIVATE actionabi_core Catch2::Catch2WithMain)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_identifiability PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME identifiability COMMAND test_identifiability)
  add_executable(test_probe tests/test_probe.cpp)
  target_link_libraries(test_probe PRIVATE actionabi_core Catch2::Catch2WithMain)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_probe PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME probe COMMAND test_probe)
  add_executable(test_converter tests/test_converter.cpp)
  target_link_libraries(test_converter PRIVATE actionabi_core Catch2::Catch2WithMain)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_converter PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME converter COMMAND test_converter)
  if(ACTIONABI_ENABLE_PINOCCHIO)
    add_executable(test_pinocchio tests/test_pinocchio.cpp)
    target_link_libraries(test_pinocchio PRIVATE actionabi_core Catch2::Catch2WithMain)
    target_compile_definitions(
      test_pinocchio PRIVATE
      ACTIONABI_TWO_LINK_URDF="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/two_link.urdf"
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(test_pinocchio PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME pinocchio COMMAND test_pinocchio)
  endif()
  if(ACTIONABI_ENABLE_CUDA)
    add_executable(test_cuda_parity tests/test_cuda_parity.cpp)
    target_link_libraries(test_cuda_parity PRIVATE actionabi_cuda Catch2::Catch2WithMain)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(test_cuda_parity PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME cuda_parity COMMAND test_cuda_parity)
    if(ACTIONABI_BUILD_BENCHMARKS)
      add_test(
        NAME benchmark_smoke
        COMMAND "${Python3_EXECUTABLE}" -m unittest tests.test_benchmark -v
      )
      set_tests_properties(
        benchmark_smoke PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      )
    endif()
  endif()
endif()
