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

/* Dual-GEMM x_x CUBIN configuration, device ABI, and launcher interface.
 */

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_DUAL_GEMM_X_X_LAUNCHER_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_DUAL_GEMM_X_X_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace bioir::cutedsl::dual_gemm_x_x::embedded
{
struct CubinImage;
}

/* Direct CUDA Driver launch ABIs for Ampere and Hopper dual GEMM.
 * These are the lowered device-kernel ABIs read from EIATTR_KPARAM_INFO,
 * not the high-level CuTeDSL __call__ signatures.
 */
namespace bioir::cutedsl::dual_gemm_x_x::abi
{

inline constexpr std::size_t kSM80MaxParameterCount = 9;
inline constexpr std::size_t kSM90MaxParameterCount = 15;

struct SM80Params
{
  cute_tensor_s1_d1_t x;
  cute_tensor_s1_d0_t w0;
  cute_tensor_s1_d0_t w1;
  cute_tensor_s1_d0_t bias0;
  cute_tensor_s1_d0_t bias1;
  cute_tensor_s1_d0_t actual_seqlen;
  cute_tensor_s2_d1_t output;
  std::int32_t i_dim;
  std::int32_t raster_factor;
};

constexpr std::size_t sm80_parameter_count(bool has_bias, bool has_mask)
{
  return 6U + (has_bias ? 2U : 0U) + (has_mask ? 1U : 0U);
}

/* Optional operands are compiled out of the parameter bank, so compact the
 * live pointers instead of leaving null placeholders.
 */
inline std::size_t
pack_sm80_kernel_params(SM80Params* params, bool has_bias, bool has_mask, void* kernel_params[kSM80MaxParameterCount])
{
  std::size_t count = 0;
  kernel_params[count++] = &params->x;
  kernel_params[count++] = &params->w0;
  kernel_params[count++] = &params->w1;
  if (has_bias)
  {
    kernel_params[count++] = &params->bias0;
    kernel_params[count++] = &params->bias1;
  }
  if (has_mask)
    kernel_params[count++] = &params->actual_seqlen;
  kernel_params[count++] = &params->output;
  kernel_params[count++] = &params->i_dim;
  kernel_params[count++] = &params->raster_factor;
  return count;
}

/* Hopper tensor maps and coordinates are all passed by value. x0 and x1 get
 * separate maps and coordinates even though the x_x family points both at the
 * same activation.
 */
struct SM90Params
{
  CUtensorMap x0_tma;
  CoordTensorS1 x0_coord;
  CUtensorMap x1_tma;
  CoordTensorS1 x1_coord;
  CUtensorMap w0_tma;
  CoordTensorS1 w0_coord;
  CUtensorMap w1_tma;
  CoordTensorS1 w1_coord;
  CUtensorMap output_tma;
  CoordTensorS2 output_coord;
  cute_tensor_s1_d0_t bias0;
  cute_tensor_s1_d0_t bias1;
  cute_tensor_s1_d0_t actual_seqlen;
  std::int32_t i_dim;
  std::uint8_t tiled_mma;
};

constexpr std::size_t sm90_parameter_count(bool has_bias, bool has_mask)
{
  return 12U + (has_bias ? 2U : 0U) + (has_mask ? 1U : 0U);
}

inline std::size_t
pack_sm90_kernel_params(SM90Params* params, bool has_bias, bool has_mask, void* kernel_params[kSM90MaxParameterCount])
{
  std::size_t count = 0;
  kernel_params[count++] = &params->x0_tma;
  kernel_params[count++] = &params->x0_coord;
  kernel_params[count++] = &params->x1_tma;
  kernel_params[count++] = &params->x1_coord;
  kernel_params[count++] = &params->w0_tma;
  kernel_params[count++] = &params->w0_coord;
  kernel_params[count++] = &params->w1_tma;
  kernel_params[count++] = &params->w1_coord;
  kernel_params[count++] = &params->output_tma;
  kernel_params[count++] = &params->output_coord;
  if (has_bias)
  {
    kernel_params[count++] = &params->bias0;
    kernel_params[count++] = &params->bias1;
  }
  if (has_mask)
    kernel_params[count++] = &params->actual_seqlen;
  kernel_params[count++] = &params->i_dim;
  kernel_params[count++] = &params->tiled_mma;
  return count;
}

static_assert(sm80_parameter_count(true, true) == kSM80MaxParameterCount);
static_assert(sm90_parameter_count(true, true) == kSM90MaxParameterCount);
static_assert(std::is_standard_layout_v<SM80Params>);
static_assert(std::is_standard_layout_v<SM90Params>);
static_assert(offsetof(SM80Params, x) == 0);
static_assert(offsetof(SM90Params, x0_tma) == 0);
static_assert(sizeof(CUtensorMap) == 128);
static_assert(alignof(CUtensorMap) >= 64);
static_assert(sizeof(CoordTensorS1) == 4);
static_assert(alignof(CoordTensorS1) == alignof(std::int32_t));
static_assert(offsetof(CoordTensorS1, dynamic_shapes) == 0);
static_assert(sizeof(CoordTensorS2) == 8);
static_assert(alignof(CoordTensorS2) == alignof(std::int32_t));
static_assert(offsetof(CoordTensorS2, dynamic_shapes) == 0);
static_assert(sizeof(std::uint8_t) == 1);

} // namespace bioir::cutedsl::dual_gemm_x_x::abi

namespace bioir::cutedsl::dual_gemm_x_x
{

enum class DType : std::uint8_t
{
  kFloat16,
  kBFloat16,
};

struct KernelConfig
{
  std::int32_t target_sm;
  std::int32_t K;
  std::int32_t N;
  std::int32_t bucket;
  DType dtype;
  bool transpose_out;
  bool has_bias;
  bool has_mask;
  EmbeddedCubinImage cubin;
  embedded::CubinImage const* embedded_image;
};

struct LaunchParams
{
  Tensor2View x;
  Tensor2View w0;
  Tensor2View w1;
  Tensor2View output;
  Tensor1View bias0;
  Tensor1View bias1;
  Tensor1View actual_seqlen;
  std::int32_t i_dim{};
  std::uint64_t stream{};
};

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  std::int32_t K,
  std::int32_t N,
  std::int32_t S,
  DType dtype,
  bool transpose_out,
  bool has_bias,
  bool has_mask);

std::uint32_t dynamic_smem_bytes(KernelConfig const& config);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace bioir::cutedsl::dual_gemm_x_x

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_DUAL_GEMM_X_X_LAUNCHER_H_ */
