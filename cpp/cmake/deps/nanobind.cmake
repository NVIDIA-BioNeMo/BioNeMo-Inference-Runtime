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

# nanobind is an explicit, pinned PEP 517 build dependency. Resolve it through
# the selected Python interpreter so an unpacked sdist builds from a populated
# wheelhouse without CMake making a hidden network request.
set(BIOIR_NANOBIND_VERSION "2.10.2")
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  RESULT_VARIABLE BIOIR_NANOBIND_CMAKE_DIR_RESULT
  OUTPUT_VARIABLE BIOIR_NANOBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

if(NOT BIOIR_NANOBIND_CMAKE_DIR_RESULT EQUAL 0 OR
   BIOIR_NANOBIND_CMAKE_DIR STREQUAL "")
  message(
    FATAL_ERROR
      "nanobind ${BIOIR_NANOBIND_VERSION} is required for the selected "
      "Python interpreter (${Python_EXECUTABLE}). Install the build-system "
      "requirements from pyproject.toml before configuring CMake.")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${BIOIR_NANOBIND_CMAKE_DIR}")
find_package(nanobind ${BIOIR_NANOBIND_VERSION} EXACT CONFIG REQUIRED)
