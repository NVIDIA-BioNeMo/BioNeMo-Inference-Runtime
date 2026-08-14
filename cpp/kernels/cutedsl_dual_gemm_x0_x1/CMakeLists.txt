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

set(BIOIR_DUAL_GEMM_X0_X1_CUBIN_HEADER
    "${CMAKE_CURRENT_LIST_DIR}/cubins/embedded_cubins.h")
if(NOT EXISTS "${BIOIR_DUAL_GEMM_X0_X1_CUBIN_HEADER}")
  message(
    FATAL_ERROR
      "Missing ${BIOIR_DUAL_GEMM_X0_X1_CUBIN_HEADER}. "
      "Run cpp/tools/prepare_cubins.py --kernel dual_gemm_x0_x1 in the "
      "private CuTeDSL development tree before configuring this public build.")
endif()

target_sources(
  _cutedsl_kernels PRIVATE bindings.cpp launcher.cpp
                           "${BIOIR_DUAL_GEMM_X0_X1_CUBIN_HEADER}")
set_source_files_properties("${BIOIR_DUAL_GEMM_X0_X1_CUBIN_HEADER}"
                            PROPERTIES HEADER_FILE_ONLY TRUE)

target_include_directories(_cutedsl_kernels PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
