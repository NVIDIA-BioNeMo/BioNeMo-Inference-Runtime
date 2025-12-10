#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(tensorrt_bionemo LANGUAGES CXX)

set(TRT_ROOT_DIR
    "/usr"
    CACHE PATH "Path to TensorRT installation")

get_filename_component(TRT_BIONEMO_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} PATH)
set(TRT_BIONEMO_THIRDPARTY_DIR ${TRT_BIONEMO_ROOT_DIR}/3rdparty)
set(TRT_BIONEMO_CPP_DIR ${TRT_BIONEMO_ROOT_DIR}/cpp)
set(TRT_BIONEMO_PYTHON_DIR ${TRT_BIONEMO_ROOT_DIR}/tensorrt_bionemo)

enable_language(C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fno-inline")
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss -DENABLE_MULTI_DEVICE=1")
if(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=medium")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-relax")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-relax")
endif()
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS}")

# Append local cmake module path for CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH "${TRT_BIONEMO_CPP_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH
     "${TRT_BIONEMO_THIRDPARTY_DIR}/TensorRT-LLM/cpp/cmake/modules")

# Append local cmake dependency path CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${TRT_BIONEMO_CPP_DIR}/cmake/deps")

# Setup rapids-cmake
include(SetupRapidsCMake)

include(cmake/setup_dependencies.cmake)

# Take code TRT_LLM
file(GLOB TRTLLM_SRCS ${TRT_LLM_CPP_DIR}/tensorrt_llm/common/*.cpp
     ${TRT_LLM_CPP_DIR}/tensorrt_llm/runtime/*.cpp)

# Build TRT-BioNeMo
include_directories(${TRT_BIONEMO_CPP_DIR} ${TRT_BIONEMO_CPP_DIR}/include)
add_subdirectory(tensorrt_bionemo)

# kernels shared library between torch and trt
install(TARGETS kernels_tensorrt_bionemo
        LIBRARY DESTINATION ${TRT_BIONEMO_PYTHON_DIR}/libs)

# Library for TensorRT plugins
install(TARGETS nvinfer_plugin_tensorrt_bionemo
        LIBRARY DESTINATION ${TRT_BIONEMO_PYTHON_DIR}/libs)

# Library for torch extensions
install(TARGETS torch_crsc_tensorrt_bionemo
        LIBRARY DESTINATION ${TRT_BIONEMO_PYTHON_DIR}/libs)
