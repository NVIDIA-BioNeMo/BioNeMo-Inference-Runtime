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

/* Pair-weighted-averaging CUBIN configuration, device ABI, and launcher interface.
 */

#ifndef TENSORRT_BIONEMO_CPP_KERNELS_CUTEDSL_PAIR_WEIGHTED_AVERAGING_LAUNCHER_H_
#define TENSORRT_BIONEMO_CPP_KERNELS_CUTEDSL_PAIR_WEIGHTED_AVERAGING_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace trtbnm::cutedsl::pair_weighted_averaging::embedded
{
struct CubinImage;
}

/* The five lowered parameters were read from EIATTR_KPARAM_INFO. They are the
 * device-kernel ABI, not the high-level CuTeDSL callable signature.
 */
namespace trtbnm::cutedsl::pair_weighted_averaging::abi
{

inline constexpr std::size_t kParameterCount = 5;

struct SM80Params
{
  cute_tensor_s3_d3_t w;
  cute_tensor_s3_d3_t v;
  cute_tensor_s3_d3_t g;
  cute_tensor_s0_d0_t weight;
  cute_tensor_s3_d3_t output;
};

inline void pack_sm80_kernel_params(SM80Params* params, void* kernel_params[kParameterCount])
{
  kernel_params[0] = &params->w;
  kernel_params[1] = &params->v;
  kernel_params[2] = &params->g;
  kernel_params[3] = &params->weight;
  kernel_params[4] = &params->output;
}

static_assert(kParameterCount == 5);
static_assert(std::is_standard_layout_v<SM80Params>);
static_assert(sizeof(cute_tensor_s3_d3_t) == 48);
static_assert(alignof(cute_tensor_s3_d3_t) == 8);
static_assert(sizeof(cute_tensor_s0_d0_t) == 8);
static_assert(alignof(cute_tensor_s0_d0_t) == 8);
static_assert(offsetof(SM80Params, w) == 0);
static_assert(offsetof(SM80Params, v) == 48);
static_assert(offsetof(SM80Params, g) == 96);
static_assert(offsetof(SM80Params, weight) == 144);
static_assert(offsetof(SM80Params, output) == 152);
static_assert(sizeof(SM80Params) == 200);
static_assert(alignof(SM80Params) == 8);

} // namespace trtbnm::cutedsl::pair_weighted_averaging::abi

namespace trtbnm::cutedsl::pair_weighted_averaging
{

enum class DType : std::uint8_t
{
  kFloat16,
  kBFloat16,
};

struct KernelConfig
{
  std::int32_t target_sm;
  std::int32_t D;
  std::int32_t c_m;
  DType dtype;
  std::int32_t n_anchor;
  std::int32_t s_anchor;
  EmbeddedCubinImage cubin;
  embedded::CubinImage const* embedded_image;
};

struct LaunchParams
{
  Tensor4View w;
  Tensor4View v;
  Tensor4View g;
  Tensor2View weight;
  Tensor4View output;
  std::uint64_t stream{};
};

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  std::int32_t I,
  std::int32_t J,
  std::int32_t S,
  std::int32_t D,
  std::int32_t c_m,
  DType dtype);

std::uint32_t dynamic_smem_bytes(KernelConfig const& config);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace trtbnm::cutedsl::pair_weighted_averaging

#endif /* TENSORRT_BIONEMO_CPP_KERNELS_CUTEDSL_PAIR_WEIGHTED_AVERAGING_LAUNCHER_H_ */
