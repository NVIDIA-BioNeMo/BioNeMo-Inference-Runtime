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

set(TRT_LLM_INCLUDE_DIR ${TRT_LLM_CPP_DIR})
mark_as_advanced(TRT_LLM_INCLUDE_DIR)

function(read_version var str)
  message(STATUS "str: ${str}")
  string(REGEX MATCH "__version__ = \"([0-9]+\\.[0-9]+\\.[0-9]+)" _ ${str})
  set(${var}
      ${CMAKE_MATCH_1}
      PARENT_SCOPE)
endfunction()

file(STRINGS "${TRT_LLM_PY_DIR}/version.py" _TRT_LLM_VERSION_FILE
     REGEX "__version__ = ")
read_version(TRT_LLM_VERSION "${_TRT_LLM_VERSION_FILE}")
message(STATUS "TRT_LLM_VERSION: ${TRT_LLM_VERSION}, ${TRT_LLM_INCLUDE_DIR}")
unset(_TRT_LLM_VERSION_FILE)

macro(find_trtllm_library libname)
  if(NOT TARGET TRT_LLM::${libname})
    find_library(
      TRT_LLM_${libname}_LIBRARY
      NAMES ${libname} REQUIRED
      PATHS ${TRT_LLM_LIBS_DIR})
    add_library(TRT_LLM::${libname} SHARED IMPORTED GLOBAL)
    list(APPEND _include_dirs "${TRT_LLM_INCLUDE_DIR}"
         "${TRT_LLM_INCLUDE_DIR}/include")
    set_target_properties(
      TRT_LLM::${libname}
      PROPERTIES IMPORTED_LOCATION "${TRT_LLM_${libname}_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${_include_dirs}")
  endif()
endmacro()

find_trtllm_library(tensorrt_llm)
find_trtllm_library(th_common)
find_trtllm_library(nvinfer_plugin_tensorrt_llm)
find_trtllm_library(tensorrt_llm_ucx_wrapper)
#find_trtllm_library(tensorrt_llm_nvrtc_wrapper)
find_trtllm_library(decoder_attention_0)
find_trtllm_library(decoder_attention_1)

# Generate TRT_LLM_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TRT_LLM
  FOUND_VAR TRT_LLM_FOUND
  VERSION_VAR TRT_LLM_VERSION
  REQUIRED_VARS TRT_LLM_INCLUDE_DIR)
