/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Shared embedded-CUBIN loading, caching, and preloading runtime.
 */

#ifndef TENSORRT_BIONEMO_CPP_KERNELS_CUBIN_RUNTIME_H_
#define TENSORRT_BIONEMO_CPP_KERNELS_CUBIN_RUNTIME_H_

#include "cubin_launch.h"

#include <cstddef>
#include <cstdint>

namespace trtbnm::cutedsl
{

struct EmbeddedCubinImage
{
  std::int32_t target_sm;
  std::int32_t kernel_sm;
  std::uint32_t dynamic_smem_bytes;
  bool non_portable_cluster_size_allowed;
  char const* launch_abi;
  std::int32_t const* supported_sms;
  std::size_t supported_sm_count;
  char const* variant_id;
  char const* kernel_symbol;
  void const* data;
  std::size_t size;
};

bool cubin_supports_sm(EmbeddedCubinImage const& image, std::int32_t device_sm);

inline constexpr std::size_t kTmaMaxRank = 5;

/* Host-side TMA descriptor recipe baked into a CUBIN at build time.
 *
 * Arch-generic: any SM90+ native kernel encodes its descriptors from these
 * fields, so families share this definition and only the set of descriptors
 * they carry (named per tensor operand) stays family-specific.
 */
struct TmaDescriptorInfo
{
  CUtensorMapDataType data_type;
  std::uint32_t rank;
  std::uint32_t global_dim_order[kTmaMaxRank];
  std::uint32_t box_dims[kTmaMaxRank];
  std::uint32_t element_strides[kTmaMaxRank];
  CUtensorMapInterleave interleave;
  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2_promotion;
  CUtensorMapFloatOOBfill oob_fill;
};

void check_cuda_driver(CUresult result, char const* operation);

CUcontext current_cuda_context();

std::int32_t cuda_sm_for_context(CUcontext context);

std::int32_t cuda_device_for_context(CUcontext context);

std::int32_t cuda_multiprocessor_count_for_context(CUcontext context);

std::int32_t current_cuda_sm();

cubin_kernel_t
load_embedded_kernel(CUcontext context, EmbeddedCubinImage const& image, bool configure_function_attributes = true);

using CubinPreloadFunction = std::size_t (*)(CUcontext context, std::int32_t device_sm);

class CubinPreloadRegistration
{
  public:
  CubinPreloadRegistration(char const* family, CubinPreloadFunction function);
};

std::size_t preload_registered_kernels();

std::size_t preload_registered_kernels_if_context_active();

} // namespace trtbnm::cutedsl

#endif /* TENSORRT_BIONEMO_CPP_KERNELS_CUBIN_RUNTIME_H_ */
