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

# Prefer the nanobind package installed for the selected Python interpreter.
# Fall back to the same FetchContent pattern/version used by the historical
# TensorRT-BioNeMo nanobind build.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  RESULT_VARIABLE TRTBNM_NANOBIND_CMAKE_DIR_RESULT
  OUTPUT_VARIABLE TRTBNM_NANOBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

if(TRTBNM_NANOBIND_CMAKE_DIR_RESULT EQUAL 0)
  list(PREPEND CMAKE_PREFIX_PATH "${TRTBNM_NANOBIND_CMAKE_DIR}")
  find_package(nanobind CONFIG QUIET)
endif()

if(NOT COMMAND nanobind_add_module)
  include(FetchContent)
  FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.10.2
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(nanobind)
endif()
