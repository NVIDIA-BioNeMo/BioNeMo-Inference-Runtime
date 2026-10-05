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

# CUTLASS headers come from the pinned 3rdparty/cutlass submodule;
# BIOIR_CUTLASS_INCLUDE_DIR or CUTLASS_PATH/CUTLASS_ROOT override them.
set(BIOIR_CUTLASS_INCLUDE_DIR
    ""
    CACHE PATH "Directory containing the CUTLASS cutlass/ and cute/ headers")

if(BIOIR_CUTLASS_INCLUDE_DIR STREQUAL "")
  foreach(BIOIR_CUTLASS_ROOT "$ENV{CUTLASS_PATH}" "$ENV{CUTLASS_ROOT}")
    if(EXISTS "${BIOIR_CUTLASS_ROOT}/include/cutlass/cutlass.h" AND
       EXISTS "${BIOIR_CUTLASS_ROOT}/include/cute/tensor.hpp")
      set(BIOIR_CUTLASS_INCLUDE_DIR "${BIOIR_CUTLASS_ROOT}/include")
      break()
    endif()
  endforeach()
endif()

set(BIOIR_CUTLASS_SUBMODULE_DIR "${BIOIR_REPO_ROOT}/3rdparty/cutlass")
if(BIOIR_CUTLASS_INCLUDE_DIR STREQUAL "" AND
   EXISTS "${BIOIR_CUTLASS_SUBMODULE_DIR}/include/cutlass/cutlass.h")
  set(BIOIR_CUTLASS_INCLUDE_DIR "${BIOIR_CUTLASS_SUBMODULE_DIR}/include")
endif()

if(NOT EXISTS "${BIOIR_CUTLASS_INCLUDE_DIR}/cutlass/cutlass.h" OR
   NOT EXISTS "${BIOIR_CUTLASS_INCLUDE_DIR}/cute/tensor.hpp")
  message(
    FATAL_ERROR
      "CUTLASS C++ headers were not found. Check out the pinned submodule with "
      "'git submodule update --init 3rdparty/cutlass', or set "
      "BIOIR_CUTLASS_INCLUDE_DIR/CUTLASS_PATH to a directory containing "
      "include/cutlass and include/cute.")
endif()

if(EXISTS "${BIOIR_CUTLASS_INCLUDE_DIR}/cutlass/version.h")
  file(STRINGS "${BIOIR_CUTLASS_INCLUDE_DIR}/cutlass/version.h"
       BIOIR_CUTLASS_VERSION_LINES
       REGEX "^#define CUTLASS_(MAJOR|MINOR|PATCH) ")
  string(REGEX REPLACE "[^0-9;]" "" BIOIR_CUTLASS_VERSION_PARTS
                       "${BIOIR_CUTLASS_VERSION_LINES}")
  string(REPLACE ";" "." BIOIR_CUTLASS_VERSION "${BIOIR_CUTLASS_VERSION_PARTS}")
  message(STATUS "CUTLASS version: ${BIOIR_CUTLASS_VERSION}")
endif()

add_library(bioir_cutlass_headers INTERFACE)
target_include_directories(
  bioir_cutlass_headers SYSTEM INTERFACE "${BIOIR_CUTLASS_INCLUDE_DIR}")

message(STATUS "CUTLASS include path: ${BIOIR_CUTLASS_INCLUDE_DIR}")
