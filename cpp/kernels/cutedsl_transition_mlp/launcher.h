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

/* Transition MLP CUBIN configuration, device ABI, and launcher interface. */

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_TRANSITION_MLP_LAUNCHER_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_TRANSITION_MLP_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace bioir::cutedsl::transition_mlp::embedded
{
struct CubinImage;
}

/* Hopper device ABI transition_mlp_sm90_v1, read from EIATTR_KPARAM_INFO. With residual, bias and mask:
 *
 *   ord  0  0x000  0x01  qk_tiled_mma    std::uint8_t
 *   ord  1  0x001  0x01  pv_tiled_mma    std::uint8_t
 *   ord  2  0x030  0x80  x_tma           CUtensorMap
 *   ord  3  0x0b0  0x04  x_coord         CoordTensorS1
 *   ord  4  0x0f0  0x80  w1_tma          CUtensorMap
 *   ord  5  0x170  0x80  w2_tma          CUtensorMap
 *   ord  6  0x1f0  0x80  residual_tma    CUtensorMap
 *   ord  7  0x270  0x04  residual_coord  CoordTensorS1
 *   ord  8  0x2b0  0x80  output_tma      CUtensorMap
 *   ord  9  0x330  0x04  output_coord    CoordTensorS1
 *   ord 10  0x338  0x08  b1              cute_tensor_s0_d0_t
 *   ord 11  0x340  0x08  b2              cute_tensor_s0_d0_t
 *   ord 12  0x348  0x10  mask            cute_tensor_s1_d0_t
 *   ord 13  0x358  0x04  num_tiles       std::int32_t
 *
 * W1 and W2 have static shapes, so their coordinate tensors lower to no parameter, and the
 * static-length biases lower to bare pointers. Without the residual, residual_tma and
 * residual_coord are absent; without bias, b1 and b2; without the mask, mask; the remaining
 * ordinals shift down. The shared-memory layouts in the kernel signature are compile-time objects
 * and occupy no slot.
 *
 * Each 0x80 TMA slot carries CuTe DSL 4.5.2's 0x40-byte non-executable CopyAtom payload followed by
 * zero padding: plain loads of x, W1, W2 and the residual, and one store of the output, none multicast.
 */
namespace bioir::cutedsl::transition_mlp::abi
{

/* Ampere device ABI transition_mlp_sm80_v1, verified from EIATTR_KPARAM_INFO.
 * Dynamic row operands carry one extent and one stride (24 bytes); static
 * weights and biases carry only a pointer (8 bytes). The optional mask carries
 * one extent (16 bytes). Layout, copy and MMA objects occupy no parameter slot.
 * With all optional operands: x@0x00, w1@0x18, b1@0x20, w2@0x28,
 * b2@0x30, residual@0x38, mask@0x50, output@0x60, ending at 0x78.
 */
struct SM80Params
{
  cute_tensor_s1_d1_t x;
  cute_tensor_s0_d0_t w1;
  cute_tensor_s0_d0_t b1;
  cute_tensor_s0_d0_t w2;
  cute_tensor_s0_d0_t b2;
  cute_tensor_s1_d1_t residual;
  cute_tensor_s1_d0_t mask;
  cute_tensor_s1_d1_t output;
};

constexpr std::size_t sm80_parameter_count(bool has_residual, bool has_bias, bool has_mask)
{
  return 4U + (has_residual ? 1U : 0U) + (has_bias ? 2U : 0U) + (has_mask ? 1U : 0U);
}

inline constexpr std::size_t kSM80MaxParameterCount = sm80_parameter_count(true, true, true);

inline std::size_t pack_sm80_kernel_params(
  SM80Params* params, bool has_residual, bool has_bias, bool has_mask, void* kernel_params[kSM80MaxParameterCount])
{
  std::size_t count = 0;
  kernel_params[count++] = &params->x;
  kernel_params[count++] = &params->w1;
  if (has_bias)
    kernel_params[count++] = &params->b1;
  kernel_params[count++] = &params->w2;
  if (has_bias)
    kernel_params[count++] = &params->b2;
  if (has_residual)
    kernel_params[count++] = &params->residual;
  if (has_mask)
    kernel_params[count++] = &params->mask;
  kernel_params[count++] = &params->output;
  return count;
}

static_assert(std::is_standard_layout_v<SM80Params>);
static_assert(sizeof(SM80Params::x) == 0x18 && sizeof(SM80Params::output) == 0x18);
static_assert(sizeof(SM80Params::w1) == 0x08 && sizeof(SM80Params::w2) == 0x08);
static_assert(sizeof(SM80Params::b1) == 0x08 && sizeof(SM80Params::b2) == 0x08);
static_assert(sizeof(SM80Params::mask) == 0x10 && sizeof(SM80Params::residual) == 0x18);
static_assert(offsetof(SM80Params, output) == 0x60 && sizeof(SM80Params) == 0x78);
static_assert(sm80_parameter_count(true, true, true) == 8U);
static_assert(sm80_parameter_count(false, false, false) == 4U);

struct SM90Params
{
  std::uint8_t qk_tiled_mma;
  std::uint8_t pv_tiled_mma;
  CUtensorMap x_tma;
  CoordTensorS1 x_coord;
  CUtensorMap w1_tma;
  CUtensorMap w2_tma;
  CUtensorMap residual_tma;
  CoordTensorS1 residual_coord;
  CUtensorMap output_tma;
  CoordTensorS1 output_coord;
  cute_tensor_s0_d0_t b1;
  cute_tensor_s0_d0_t b2;
  cute_tensor_s1_d0_t mask;
  std::int32_t num_tiles;
};

constexpr std::size_t sm90_parameter_count(bool has_residual, bool has_bias, bool has_mask)
{
  return 9U + (has_residual ? 2U : 0U) + (has_bias ? 2U : 0U) + (has_mask ? 1U : 0U);
}

inline constexpr std::size_t kSM90MaxParameterCount = sm90_parameter_count(true, true, true);

/* params and kernel_params must stay alive until the launch call returns. */
inline std::size_t pack_sm90_kernel_params(
  SM90Params* params, bool has_residual, bool has_bias, bool has_mask, void* kernel_params[kSM90MaxParameterCount])
{
  std::size_t count = 0;
  kernel_params[count++] = &params->qk_tiled_mma;
  kernel_params[count++] = &params->pv_tiled_mma;
  kernel_params[count++] = &params->x_tma;
  kernel_params[count++] = &params->x_coord;
  kernel_params[count++] = &params->w1_tma;
  kernel_params[count++] = &params->w2_tma;
  if (has_residual)
  {
    kernel_params[count++] = &params->residual_tma;
    kernel_params[count++] = &params->residual_coord;
  }
  kernel_params[count++] = &params->output_tma;
  kernel_params[count++] = &params->output_coord;
  if (has_bias)
  {
    kernel_params[count++] = &params->b1;
    kernel_params[count++] = &params->b2;
  }
  if (has_mask)
    kernel_params[count++] = &params->mask;
  kernel_params[count++] = &params->num_tiles;
  return count;
}

/* The driver takes one pointer per parameter, so host struct padding is irrelevant; only each
 * parameter's width must match the CUBIN.
 */
static_assert(std::is_standard_layout_v<SM90Params>);
static_assert(sizeof(CUtensorMap) == 0x80 && alignof(CUtensorMap) >= 64);
static_assert(sizeof(SM90Params::qk_tiled_mma) == 0x01, "unexpected transition MLP MMA token width");
static_assert(sizeof(SM90Params::pv_tiled_mma) == 0x01, "unexpected transition MLP MMA token width");
static_assert(sizeof(SM90Params::x_coord) == 0x04, "unexpected transition MLP coordinate width");
static_assert(sizeof(SM90Params::residual_coord) == 0x04, "unexpected transition MLP coordinate width");
static_assert(sizeof(SM90Params::output_coord) == 0x04, "unexpected transition MLP coordinate width");
static_assert(sizeof(SM90Params::b1) == 0x08, "unexpected transition MLP bias width");
static_assert(sizeof(SM90Params::b2) == 0x08, "unexpected transition MLP bias width");
static_assert(sizeof(SM90Params::mask) == 0x10, "unexpected transition MLP mask width");
static_assert(sizeof(SM90Params::num_tiles) == 0x04, "unexpected transition MLP tile-count width");
static_assert(sm90_parameter_count(true, true, true) == 14U);
static_assert(sm90_parameter_count(false, false, false) == 9U);

} // namespace bioir::cutedsl::transition_mlp::abi

namespace bioir::cutedsl::transition_mlp
{

enum class DType : std::uint8_t
{
  kBFloat16,
};

/* Launch fields generated with the corresponding CUBIN. bucket is the pseudo sequence length the
 * image was tuned for, or 0 for an image that serves every size.
 */
struct KernelSpec
{
  std::int32_t target_sm;
  std::int32_t kernel_sm;
  std::int32_t width;
  std::int32_t hidden;
  std::int32_t bucket;
  bool is_silu_gate;
  bool is_three_way;
  bool has_bias;
  bool has_mask;
  bool has_residual;
  std::uint32_t tile_m;
  std::uint32_t num_threads;
};

struct KernelConfig
{
  KernelSpec spec;
  DType dtype;
  EmbeddedCubinImage cubin;
  embedded::CubinImage const* embedded_image;
};

inline std::uint32_t dynamic_smem_bytes(KernelConfig const& config)
{
  if (config.cubin.dynamic_smem_bytes == 0)
    throw std::invalid_argument("transition MLP CUBIN has no dynamic shared-memory metadata");
  return config.cubin.dynamic_smem_bytes;
}

/* x, residual and output are [rows, width]; w1 is [hidden, width], or [2 * hidden, width] holding
 * value rows then gate rows for the SwiGLU, or [3 * hidden, width] holding value, gate and second-value
 * rows for the 3-way SwiGLU; w2 is [width, hidden]. b1, b2, mask and residual are read only when the
 * CUBIN has them. output may alias residual.
 */
struct LaunchParams
{
  Tensor2View x;
  Tensor2View w1;
  Tensor1View b1;
  Tensor2View w2;
  Tensor1View b2;
  Tensor2View residual;
  Tensor1View mask;
  Tensor2View output;
  std::uint64_t stream{};
};

std::vector<KernelSpec> kernel_specs();

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  DType dtype,
  bool is_silu_gate,
  bool is_three_way,
  bool has_bias,
  bool has_mask,
  bool has_residual,
  std::int32_t width,
  std::int32_t hidden,
  std::int32_t bucket);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace bioir::cutedsl::transition_mlp

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_TRANSITION_MLP_LAUNCHER_H_ */
