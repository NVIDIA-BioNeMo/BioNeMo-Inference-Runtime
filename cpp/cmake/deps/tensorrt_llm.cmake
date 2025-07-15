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

find_package(Python REQUIRED)

execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "import importlib; pkg=importlib.util.find_spec('tensorrt_llm'); print(pkg.submodule_search_locations[0])"
  RESULT_VARIABLE FOUND_STATUS
  OUTPUT_VARIABLE TRT_LLM_PY_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if("${TRT_LLM_PY_DIR}" MATCHES "tensorrt_llm")
  # trtllm package is not installed, set its path from 3rdparty/TensorRT-LLM
  message(STATUS "Found tensorrt_llm package at ${TRT_LLM_PY_DIR}")
else()
  message(
    WARNING
      "Not found tensorrt_llm package, set its path from 3rdparty/TensorRT-LLM")
  set(TRT_LLM_PY_DIR ${TRT_BIONEMO_THIRDPARTY_DIR}/TensorRT-LLM/tensorrt_llm)
endif()

set(TRT_LLM_CPP_DIR ${TRT_BIONEMO_THIRDPARTY_DIR}/TensorRT-LLM/cpp)
set(TRT_LLM_CPP_DIR
    ${TRT_LLM_CPP_DIR}
    PARENT_SCOPE)

set(TRT_LLM_LIBS_DIR ${TRT_LLM_PY_DIR}/libs)

set(TRT_LIB_DIR ${TRT_ROOT_DIR}/lib)
set(TRT_INCLUDE_DIR ${TRT_ROOT_DIR}/include)
find_package(TRT_LLM 1.0.0 MODULE REQUIRED)
