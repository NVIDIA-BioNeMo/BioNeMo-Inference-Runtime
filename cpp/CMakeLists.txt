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

cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

project(bionemo_ir_cutedsl_kernels LANGUAGES CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

get_filename_component(BIOIR_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.."
                       ABSOLUTE)
set(BIOIR_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(BIOIR_PYTHON_PACKAGE_DIR "${BIOIR_REPO_ROOT}/bionemo_ir")

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/setup_dependencies.cmake")

add_subdirectory(kernels)
