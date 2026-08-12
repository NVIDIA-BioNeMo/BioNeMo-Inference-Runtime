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

add_subdirectory(cutedsl_adaln_layernorm_sigmoid)
add_subdirectory(cutedsl_gated_sigmoid)
add_subdirectory(cutedsl_outer_product_mean)
add_subdirectory(cutedsl_triangle_attention)
add_subdirectory(cutedsl_pairwise_attention)
add_subdirectory(cutedsl_dual_gemm_x_x)
add_subdirectory(cutedsl_dual_gemm_x0_x1)
add_subdirectory(cutedsl_pair_weighted_averaging)

install(TARGETS _cutedsl_kernels LIBRARY DESTINATION tensorrt_bionemo/libs)
