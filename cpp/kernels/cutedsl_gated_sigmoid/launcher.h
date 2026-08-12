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

/* Gated-sigmoid CUBIN configuration, device ABI, and launcher interface. */

#ifndef TENSORRT_BIONEMO_CPP_KERNELS_CUTEDSL_GATED_SIGMOID_LAUNCHER_H_
#define TENSORRT_BIONEMO_CPP_KERNELS_CUTEDSL_GATED_SIGMOID_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace trtbnm::cutedsl::gated_sigmoid::embedded
{
struct CubinImage;
}

/* Direct CUDA Driver launch ABI for the Ampere-style gated-sigmoid GEMM.
 *
 * This is the device-kernel ABI, not the high-level CuTeDSL __call__ ABI, and
 * was read back from the compiled CUBINs' EIATTR_KPARAM_INFO. Two details are
 * not visible from the Python signature:
 *
 *  - `rasterization_factor` is ordinal 5, computed on the host. The Python
 *    __call__ takes 7 arguments; the device kernel takes 8 parameters.
 *  - `has_bias=false` REMOVES the bias slot rather than passing null, so the
 *    parameter bank is 108 bytes instead of 124 and the ordinals after it
 *    shift down by one. The bias flag is therefore an ABI axis.
 *
 * The layout/tiled-copy/tiled-mma arguments in the @cute.kernel signature are
 * compile-time objects and are traced away; they occupy no parameter slot.
 */
namespace trtbnm::cutedsl::gated_sigmoid::abi
{

inline constexpr std::size_t kSM80ParameterCountWithBias = 8;
inline constexpr std::size_t kSM80ParameterCountNoBias = 7;
inline constexpr std::size_t kSM80MaxParameterCount = kSM80ParameterCountWithBias;

struct SM80Params
{
  cute_tensor_s2_d1_t s;
  cute_tensor_s2_d1_t weight;
  cute_tensor_s1_d0_t bias;
  cute_tensor_s2_d1_t mha_out;
  cute_tensor_s2_d1_t output;
  std::int32_t rasterization_factor;
  std::int32_t mult;
  std::int32_t inner;
};

/* Both params and kernel_params must remain alive until the CUDA launch call
 * returns. Returns the number of populated entries, which depends on has_bias.
 */
inline std::size_t
pack_sm80_kernel_params(SM80Params* params, void* kernel_params[kSM80MaxParameterCount], bool has_bias)
{
  std::size_t index = 0;
  kernel_params[index++] = &params->s;
  kernel_params[index++] = &params->weight;
  if (has_bias)
    kernel_params[index++] = &params->bias;
  kernel_params[index++] = &params->mha_out;
  kernel_params[index++] = &params->output;
  kernel_params[index++] = &params->rasterization_factor;
  kernel_params[index++] = &params->mult;
  kernel_params[index++] = &params->inner;
  return index;
}

/* Offsets of the lowered parameter bank, with bias present. */
static_assert(sizeof(cute_tensor_s2_d1_t) == 24, "gated-sigmoid s2_d1 operand size changed");
static_assert(sizeof(cute_tensor_s1_d0_t) == 16, "gated-sigmoid s1_d0 bias size changed");
static_assert(offsetof(SM80Params, s) == 0, "unexpected gated-sigmoid s offset");
static_assert(offsetof(SM80Params, weight) == 24, "unexpected gated-sigmoid weight offset");
static_assert(offsetof(SM80Params, bias) == 48, "unexpected gated-sigmoid bias offset");
static_assert(offsetof(SM80Params, mha_out) == 64, "unexpected gated-sigmoid mha_out offset");
static_assert(offsetof(SM80Params, output) == 88, "unexpected gated-sigmoid output offset");
static_assert(offsetof(SM80Params, rasterization_factor) == 112, "unexpected gated-sigmoid raster offset");
static_assert(offsetof(SM80Params, mult) == 116, "unexpected gated-sigmoid mult offset");
static_assert(offsetof(SM80Params, inner) == 120, "unexpected gated-sigmoid inner offset");

/* The lowered parameter bank ends at 124 bytes, which is what
 * EIATTR_KPARAM_INFO reports. This backing struct is 128: its int64 strides
 * force 8-byte alignment, so four tail padding bytes follow `inner`. That is
 * harmless because every parameter is handed to the driver through its own
 * pointer in kernel_params[], never as one blob -- but assert both numbers so a
 * layout change cannot hide behind the padding.
 */
static_assert(offsetof(SM80Params, inner) + sizeof(SM80Params::inner) == 124, "gated-sigmoid parameter bank changed");
static_assert(sizeof(SM80Params) == 128, "gated-sigmoid SM80 backing struct size changed");
static_assert(alignof(SM80Params) == 8, "gated-sigmoid SM80 backing struct alignment changed");

} // namespace trtbnm::cutedsl::gated_sigmoid::abi

namespace trtbnm::cutedsl::gated_sigmoid
{

enum class DType : std::uint8_t
{
  kFloat16,
  kBFloat16,
};

/* One compiled tile configuration.
 *
 * Unlike the attention families there is no hand-written spec table to keep in
 * sync with the JSON configs: K and N are symbolic in this kernel, so the tile
 * IS the runtime key. make_kernel_config() looks the caller's tile up directly
 * in the generated registry, which the builder emits from the same JSON the
 * Python interface reads. There is no second copy of the geometry to drift.
 */
struct KernelSpec
{
  std::int32_t target_sm;
  std::int32_t m_block_size;
  std::int32_t n_block_size;
  std::int32_t k_block_size;
  std::int32_t num_stages;
  std::int32_t raster_factor;
  std::int32_t atom_layout_mnk[3];
  std::int32_t num_threads;
};

struct KernelConfig
{
  KernelSpec spec;
  DType dtype;
  bool has_bias;
  EmbeddedCubinImage cubin;
  embedded::CubinImage const* embedded_image;
};

inline std::uint32_t dynamic_smem_bytes(KernelConfig const& config)
{
  if (config.cubin.dynamic_smem_bytes == 0)
    throw std::invalid_argument("Gated-sigmoid CUBIN has no dynamic shared-memory metadata");
  return config.cubin.dynamic_smem_bytes;
}

struct LaunchParams
{
  Tensor2View s;
  Tensor2View weight;
  Tensor1View bias;
  Tensor2View mha_out;
  Tensor2View output;
  /* Broadcast multiplicity: mha_out has `mult` times as many rows as s. */
  std::int32_t mult{1};
  /* Rows per sample; equals s.shape[0] when mult == 1. */
  std::int32_t inner{1};
  std::uint64_t stream{};
};

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  DType dtype,
  bool has_bias,
  std::int32_t m_block_size,
  std::int32_t n_block_size,
  std::int32_t k_block_size,
  std::int32_t num_stages,
  std::int32_t raster_factor,
  std::int32_t atom_layout_m,
  std::int32_t atom_layout_n,
  std::int32_t atom_layout_k);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace trtbnm::cutedsl::gated_sigmoid

#endif /* TENSORRT_BIONEMO_CPP_KERNELS_CUTEDSL_GATED_SIGMOID_LAUNCHER_H_ */
