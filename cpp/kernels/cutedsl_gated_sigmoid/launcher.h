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

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_GATED_SIGMOID_LAUNCHER_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_GATED_SIGMOID_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace bioir::cutedsl::gated_sigmoid::embedded
{
struct CubinImage;
}

/* Direct CUDA Driver launch ABIs for the gated-sigmoid kernels.
 *
 * These are the device-kernel ABIs, not the high-level CuTeDSL __call__ ABIs,
 * read back from the compiled CUBINs' EIATTR_KPARAM_INFO. Details not visible
 * from the Python signatures:
 *
 *  - An absent bias, residual or mask REMOVES its slot rather than passing
 *    null, and the ordinals after it shift down. Each flag is an ABI axis.
 *  - SM80: `rasterization_factor` follows `output` and is computed on the host.
 *  - SM90: the WGMMA tiled MMA keeps a runtime `accumulate` flag, a 1-byte
 *    slot after `output` that starts false. The grid decomposition (`n_it`,
 *    `n_nt`, `n_chunks`), `N` and the K-block count follow `chunk` and are
 *    computed on the host.
 *
 * The layout/tiled-copy/tiled-mma arguments in the @cute.kernel signatures are
 * compile-time objects and are traced away; they occupy no parameter slot.
 */
namespace bioir::cutedsl::gated_sigmoid::abi
{

inline constexpr std::size_t kSM80MaxParameterCount = 10;
inline constexpr std::size_t kSM90MaxParameterCount = 16;

inline constexpr std::size_t optional_operand_count(bool has_bias, bool has_residual, bool has_mask)
{
  return static_cast<std::size_t>(has_bias) + static_cast<std::size_t>(has_residual)
    + static_cast<std::size_t>(has_mask);
}

inline constexpr std::size_t sm80_parameter_count(bool has_bias, bool has_residual, bool has_mask)
{
  return 7 + optional_operand_count(has_bias, has_residual, has_mask);
}

inline constexpr std::size_t sm90_parameter_count(bool has_bias, bool has_residual, bool has_mask)
{
  return 13 + optional_operand_count(has_bias, has_residual, has_mask);
}

struct SM80Params
{
  cute_tensor_s2_d1_t s;
  cute_tensor_s2_d1_t weight;
  cute_tensor_s1_d0_t bias;
  cute_tensor_s2_d1_t mha_out;
  cute_tensor_s2_d1_t residual;
  cute_tensor_s1_d0_t mask;
  cute_tensor_s2_d1_t output;
  std::int32_t rasterization_factor;
  std::int32_t mult;
  std::int32_t inner;
};

struct SM90Params
{
  cute_tensor_s2_d1_t s;
  cute_tensor_s2_d1_t weight;
  cute_tensor_s1_d0_t bias;
  cute_tensor_s2_d1_t mha_out;
  cute_tensor_s2_d1_t residual;
  cute_tensor_s1_d0_t mask;
  cute_tensor_s2_d1_t output;
  std::uint8_t mma_accumulate;
  std::int32_t mult;
  std::int32_t inner;
  std::int32_t chunk;
  std::int32_t n_it;
  std::int32_t n_nt;
  std::int32_t n_chunks;
  std::int32_t N;
  std::int32_t k_blocks;
};

/* Both params and kernel_params must remain alive until the CUDA launch call
 * returns. Each returns the number of populated entries.
 */
template <typename Params>
inline std::size_t pack_operands(Params* params, void** kernel_params, bool has_bias, bool has_residual, bool has_mask)
{
  std::size_t index = 0;
  kernel_params[index++] = &params->s;
  kernel_params[index++] = &params->weight;
  if (has_bias)
    kernel_params[index++] = &params->bias;
  kernel_params[index++] = &params->mha_out;
  if (has_residual)
    kernel_params[index++] = &params->residual;
  if (has_mask)
    kernel_params[index++] = &params->mask;
  kernel_params[index++] = &params->output;
  return index;
}

inline std::size_t pack_sm80_kernel_params(
  SM80Params* params, void* kernel_params[kSM80MaxParameterCount], bool has_bias, bool has_residual, bool has_mask)
{
  std::size_t index = pack_operands(params, kernel_params, has_bias, has_residual, has_mask);
  kernel_params[index++] = &params->rasterization_factor;
  kernel_params[index++] = &params->mult;
  kernel_params[index++] = &params->inner;
  return index;
}

inline std::size_t pack_sm90_kernel_params(
  SM90Params* params, void* kernel_params[kSM90MaxParameterCount], bool has_bias, bool has_residual, bool has_mask)
{
  std::size_t index = pack_operands(params, kernel_params, has_bias, has_residual, has_mask);
  kernel_params[index++] = &params->mma_accumulate;
  kernel_params[index++] = &params->mult;
  kernel_params[index++] = &params->inner;
  kernel_params[index++] = &params->chunk;
  kernel_params[index++] = &params->n_it;
  kernel_params[index++] = &params->n_nt;
  kernel_params[index++] = &params->n_chunks;
  kernel_params[index++] = &params->N;
  kernel_params[index++] = &params->k_blocks;
  return index;
}

static_assert(sizeof(cute_tensor_s2_d1_t) == 24, "gated-sigmoid s2_d1 operand size changed");
static_assert(sizeof(cute_tensor_s1_d0_t) == 16, "gated-sigmoid s1_d0 operand size changed");
static_assert(sizeof(SM80Params) == 168, "gated-sigmoid SM80 backing struct size changed");
static_assert(sizeof(SM90Params) == 192, "gated-sigmoid SM90 backing struct size changed");

} // namespace bioir::cutedsl::gated_sigmoid::abi

namespace bioir::cutedsl::gated_sigmoid
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
enum class LaunchAbi : std::uint8_t
{
  kSM80,
  kSM90,
};

struct KernelSpec
{
  std::int32_t target_sm;
  LaunchAbi launch_abi;
  std::int32_t m_block_size;
  std::int32_t n_block_size;
  std::int32_t k_block_size;
  std::int32_t num_stages;
  std::int32_t raster_factor;
  std::int32_t atom_layout_mnk[3];
  std::int32_t num_threads;
  std::int32_t unroll;
};

struct KernelConfig
{
  KernelSpec spec;
  DType dtype;
  bool has_bias;
  bool has_residual;
  bool has_mask;
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
  Tensor2View residual;
  Tensor1View mask;
  Tensor2View output;
  /* Broadcast multiplicity: mha_out has `mult` times as many rows as s. */
  std::int32_t mult{1};
  /* Rows per sample; equals s.shape[0] when mult == 1. */
  std::int32_t inner{1};
  /* SM90 only: samples per CTA; 0 runs every sample in one CTA. */
  std::int32_t chunk{0};
  std::uint64_t stream{};
};

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  DType dtype,
  bool has_bias,
  bool has_residual,
  bool has_mask,
  std::int32_t m_block_size,
  std::int32_t n_block_size,
  std::int32_t k_block_size,
  std::int32_t num_stages,
  std::int32_t raster_factor,
  std::int32_t atom_layout_m,
  std::int32_t atom_layout_n,
  std::int32_t atom_layout_k,
  std::int32_t unroll);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace bioir::cutedsl::gated_sigmoid

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_GATED_SIGMOID_LAUNCHER_H_ */
