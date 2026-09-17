if(NOT DEFINED ACTIONABI_BINARY_DIR)
  message(FATAL_ERROR "ACTIONABI_BINARY_DIR is required")
endif()

set(prefix "${ACTIONABI_BINARY_DIR}/install-smoke")
file(REMOVE_RECURSE "${prefix}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${ACTIONABI_BINARY_DIR}" --prefix "${prefix}"
  RESULT_VARIABLE install_result
  OUTPUT_VARIABLE install_output
  ERROR_VARIABLE install_error
)
if(NOT install_result EQUAL 0)
  message(FATAL_ERROR "install failed:\n${install_output}\n${install_error}")
endif()

set(executable "${prefix}/bin/actionabi${ACTIONABI_EXECUTABLE_SUFFIX}")
if(NOT EXISTS "${executable}")
  message(FATAL_ERROR "installed CLI is missing: ${executable}")
endif()
if(NOT EXISTS "${prefix}/include/actionabi/contract.hpp")
  message(FATAL_ERROR "installed public headers are missing")
endif()

execute_process(
  COMMAND "${executable}" --version
  RESULT_VARIABLE smoke_result
  OUTPUT_VARIABLE smoke_output
  ERROR_VARIABLE smoke_error
)
if(NOT smoke_result EQUAL 0)
  message(FATAL_ERROR "installed CLI failed:\n${smoke_output}\n${smoke_error}")
endif()
if(NOT smoke_output MATCHES "ActionABI 1.1.1")
  message(FATAL_ERROR "unexpected installed CLI version: ${smoke_output}")
endif()
