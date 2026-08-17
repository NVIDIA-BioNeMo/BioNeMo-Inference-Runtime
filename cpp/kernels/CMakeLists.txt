# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

nanobind_add_module(_cutedsl_kernels bindings.cpp cubin_runtime.cpp)
target_include_directories(_cutedsl_kernels PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(_cutedsl_kernels PRIVATE CUDA::cuda_driver)

if(NOT DEFINED BIOIR_CUBIN_MATERIALIZED_DIR OR
   BIOIR_CUBIN_MATERIALIZED_DIR STREQUAL "")
  message(
    FATAL_ERROR
      "BIOIR_CUBIN_MATERIALIZED_DIR is required. Build through setup.py or run "
      "cpp/cmake/materialize_cubin_payloads.py before configuring CMake.")
endif()
if(NOT IS_ABSOLUTE "${BIOIR_CUBIN_MATERIALIZED_DIR}")
  message(FATAL_ERROR "BIOIR_CUBIN_MATERIALIZED_DIR must be absolute")
endif()
if(NOT IS_DIRECTORY "${BIOIR_CUBIN_MATERIALIZED_DIR}")
  message(
    FATAL_ERROR
      "CUBIN materialization directory does not exist: ${BIOIR_CUBIN_MATERIALIZED_DIR}")
endif()

set(BIOIR_CUBIN_MATERIALIZATION_STAMP
    "${BIOIR_CUBIN_MATERIALIZED_DIR}/materialization.json")
if(NOT EXISTS "${BIOIR_CUBIN_MATERIALIZATION_STAMP}")
  message(
    FATAL_ERROR
      "Incomplete CUBIN materialization: missing ${BIOIR_CUBIN_MATERIALIZATION_STAMP}")
endif()

set(BIOIR_CUBIN_PAYLOAD_SOURCES)
foreach(BIOIR_CUBIN_SHARD 0 1 2 3 4 5 6 7 8 9 a b c d e f)
  set(BIOIR_CUBIN_PAYLOAD_SOURCE
      "${BIOIR_CUBIN_MATERIALIZED_DIR}/cubin_payloads_${BIOIR_CUBIN_SHARD}.S")
  if(NOT EXISTS "${BIOIR_CUBIN_PAYLOAD_SOURCE}")
    message(
      FATAL_ERROR
        "Incomplete CUBIN materialization: missing ${BIOIR_CUBIN_PAYLOAD_SOURCE}")
  endif()
  list(APPEND BIOIR_CUBIN_PAYLOAD_SOURCES "${BIOIR_CUBIN_PAYLOAD_SOURCE}")
endforeach()
target_sources(_cutedsl_kernels PRIVATE ${BIOIR_CUBIN_PAYLOAD_SOURCES})
target_include_directories(_cutedsl_kernels
                           PRIVATE "${BIOIR_CUBIN_MATERIALIZED_DIR}")

# Every family directory is registered identically: its handwritten launcher
# and bindings translation units, its generated registry pair, and its own
# directory on the include path. Families differ only in whether they keep a
# launcher.h, so enumerate the sources instead of repeating this per family.
# cutedsl_*.h siblings (shared launch/ABI headers) also match the glob, so key
# discovery on the launcher every family directory must contain.
file(GLOB BIOIR_CUBIN_FAMILY_LAUNCHERS
     "${CMAKE_CURRENT_LIST_DIR}/cutedsl_*/launcher.cpp")
list(SORT BIOIR_CUBIN_FAMILY_LAUNCHERS)
if(NOT BIOIR_CUBIN_FAMILY_LAUNCHERS)
  message(FATAL_ERROR "No CuTeDSL kernel families found in ${CMAKE_CURRENT_LIST_DIR}")
endif()

foreach(BIOIR_CUBIN_FAMILY_LAUNCHER ${BIOIR_CUBIN_FAMILY_LAUNCHERS})
  get_filename_component(BIOIR_CUBIN_FAMILY_ROOT "${BIOIR_CUBIN_FAMILY_LAUNCHER}" DIRECTORY)
  get_filename_component(BIOIR_CUBIN_FAMILY_DIR "${BIOIR_CUBIN_FAMILY_ROOT}" NAME)
  string(REGEX REPLACE "^cutedsl_" "" BIOIR_CUBIN_FAMILY "${BIOIR_CUBIN_FAMILY_DIR}")

  set(BIOIR_CUBIN_REGISTRY_HEADER
      "${BIOIR_CUBIN_MATERIALIZED_DIR}/${BIOIR_CUBIN_FAMILY}_registry.h")
  set(BIOIR_CUBIN_REGISTRY_SOURCE
      "${BIOIR_CUBIN_MATERIALIZED_DIR}/${BIOIR_CUBIN_FAMILY}_registry.cpp")
  foreach(BIOIR_CUBIN_FAMILY_FILE
          "${BIOIR_CUBIN_REGISTRY_HEADER}" "${BIOIR_CUBIN_REGISTRY_SOURCE}"
          "${BIOIR_CUBIN_FAMILY_ROOT}/bindings.cpp"
          "${BIOIR_CUBIN_FAMILY_ROOT}/launcher.cpp")
    if(NOT EXISTS "${BIOIR_CUBIN_FAMILY_FILE}")
      message(
        FATAL_ERROR
          "Incomplete ${BIOIR_CUBIN_FAMILY} CUBIN family: missing "
          "${BIOIR_CUBIN_FAMILY_FILE}")
    endif()
  endforeach()

  target_sources(
    _cutedsl_kernels
    PRIVATE "${BIOIR_CUBIN_FAMILY_ROOT}/bindings.cpp"
            "${BIOIR_CUBIN_FAMILY_ROOT}/launcher.cpp"
            "${BIOIR_CUBIN_REGISTRY_SOURCE}" "${BIOIR_CUBIN_REGISTRY_HEADER}")
  if(EXISTS "${BIOIR_CUBIN_FAMILY_ROOT}/launcher.h")
    target_sources(_cutedsl_kernels
                   PRIVATE "${BIOIR_CUBIN_FAMILY_ROOT}/launcher.h")
  endif()
  set_source_files_properties("${BIOIR_CUBIN_REGISTRY_HEADER}"
                              PROPERTIES HEADER_FILE_ONLY TRUE)
  target_include_directories(_cutedsl_kernels
                             PRIVATE "${BIOIR_CUBIN_FAMILY_ROOT}")
endforeach()

install(TARGETS _cutedsl_kernels LIBRARY DESTINATION bionemo_ir/libs)
