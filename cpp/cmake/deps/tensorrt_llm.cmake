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

set(TRT_LLM_ROOT_DIR ${TRT_BIONEMO_THIRDPARTY_DIR}/TensorRT-LLM)
set(TRT_LLM_CPP_DIR ${TRT_LLM_ROOT_DIR}/cpp)
set(TRT_LLM_CPP_DIR
    ${TRT_LLM_CPP_DIR}
    PARENT_SCOPE)
set(TRT_LLM_PY_DIR ${TRT_LLM_ROOT_DIR}/tensorrt_llm)
set(TRT_LLM_LIBS_DIR ${TRT_LLM_PY_DIR}/libs)
set(TRT_LLM_CPP_BUILD_DIR ${TRT_LLM_CPP_DIR}/build_${CMAKE_BUILD_TYPE})

set(TRT_LIB_DIR ${TRT_ROOT_DIR}/lib)
set(TRT_INCLUDE_DIR ${TRT_ROOT_DIR}/include)
find_package(TRT_LLM 0.18.0 MODULE REQUIRED)
