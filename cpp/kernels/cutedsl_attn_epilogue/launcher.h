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

/* Fused attention epilogue CUBIN ABI and launcher interface. */

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_ATTN_EPILOGUE_LAUNCHER_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_ATTN_EPILOGUE_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace bioir::cutedsl::attn_epilogue::embedded
{
struct CubinImage;
}

/* Hopper device ABI from EIATTR_KPARAM_INFO (launch ABI attn_epilogue_sm90_v1):
 *
 *   ord  0  0x030  0x80  o_tma        non-executable TMA load atom
 *   ord  1  0x0b0  0x08  o_coord      CoordTensorS2 {J, B*I}
 *   ord  2  0x0f0  0x80  g_tma        non-executable TMA load atom
 *   ord  3  0x170  0x08  g_coord      CoordTensorS2 {J, B*I}
 *   ord  4  0x1b0  0x80  w_tma        non-executable TMA load atom
 *   ord  5  0x230  0x80  z_tma        non-executable TMA load atom
 *   ord  6  0x2b0  0x08  z_coord      CoordTensorS2 {J, B*I}
 *   ord  7  0x2f0  0x80  d_tma        non-executable TMA store atom
 *   ord  8  0x370  0x08  d_coord      CoordTensorS2 {J, B*I}
 *   ord  9  0x378  0x04  gj           J tiles per pair row
 *   ord 10  0x37c  0x04  total_tiles  B*I * gj
 *   ord 11  0x380  0x01  tiled_mma    std::uint8_t
 *
 * With the output-projection bias, its static-shape [C] tensor lowers to one
 * pointer at ord 9 (0x378, 0x08), and gj, total_tiles and tiled_mma move to
 * 0x380, 0x384 and 0x388.
 *
 * With the output gate, its y_tma atom and CoordTensorS2 {J, B*I / mult}
 * y_coord follow z_coord as ords 7 and 8, and every later parameter moves
 * down two ordinals.
 *
 * Without the residual, z_tma and z_coord are absent and every later
 * parameter moves up two ordinals. Such a kernel has no output gate.
 *
 * The weight's extents are static, so its coordinate tensor lowers to nothing.
 * Every 0x80 atom slot carries CuTe's 0x40-byte non-executable CopyAtom payload
 * followed by zero padding.
 *
 * Launch ABI attn_epilogue_sm90_streamed_v1 shares this bank. Its kernel
 * streams Wo in tile_n-channel slices, so total_tiles counts
 * B*I * gj * C / tile_n tiles.
 */
namespace bioir::cutedsl::attn_epilogue::abi
{

struct SM90Params
{
  CUtensorMap o_tma;
  CoordTensorS2 o_coord;
  CUtensorMap g_tma;
  CoordTensorS2 g_coord;
  CUtensorMap w_tma;
  CUtensorMap z_tma;
  CoordTensorS2 z_coord;
  CUtensorMap y_tma;
  CoordTensorS2 y_coord;
  CUtensorMap d_tma;
  CoordTensorS2 d_coord;
  cute_tensor_s0_d0_t bias;
  std::int32_t gj;
  std::int32_t total_tiles;
  std::uint8_t tiled_mma;
};

constexpr std::size_t sm90_parameter_count(bool has_bias, bool has_output_gate, bool has_residual)
{
  return 10U + (has_residual ? 2U : 0U) + (has_bias ? 1U : 0U) + (has_output_gate ? 2U : 0U);
}

inline constexpr std::size_t kSM90MaxParameterCount = sm90_parameter_count(true, true, true);

/* Returns the number of packed parameters. */
inline std::size_t pack_sm90_kernel_params(
  SM90Params* params,
  bool has_bias,
  bool has_output_gate,
  bool has_residual,
  void* kernel_params[kSM90MaxParameterCount])
{
  std::size_t count = 0;
  kernel_params[count++] = &params->o_tma;
  kernel_params[count++] = &params->o_coord;
  kernel_params[count++] = &params->g_tma;
  kernel_params[count++] = &params->g_coord;
  kernel_params[count++] = &params->w_tma;
  if (has_residual)
  {
    kernel_params[count++] = &params->z_tma;
    kernel_params[count++] = &params->z_coord;
  }
  if (has_output_gate)
  {
    kernel_params[count++] = &params->y_tma;
    kernel_params[count++] = &params->y_coord;
  }
  kernel_params[count++] = &params->d_tma;
  kernel_params[count++] = &params->d_coord;
  if (has_bias)
    kernel_params[count++] = &params->bias;
  kernel_params[count++] = &params->gj;
  kernel_params[count++] = &params->total_tiles;
  kernel_params[count++] = &params->tiled_mma;
  return count;
}

/* Ampere device ABI from EIATTR_KPARAM_INFO (launch ABI attn_epilogue_sm80_v1),
 * every operand a by-value CuTe tensor over the kernel's (J, K or C, B*I) modes:
 *
 *   ord 0  0x00  0x18  o     cute_tensor_s2_d1_t  extents {J, B*I}, stride {B*I}
 *   ord 1  0x18  0x20  g     cute_tensor_s2_d2_t  extents {J, B*I}, strides {J, B*I}
 *   ord 2  0x38  0x08  w     cute_tensor_s0_d0_t
 *   ord 3  0x40  0x08  bias  cute_tensor_s0_d0_t  (biased CUBINs only)
 *   ord 4  0x48  0x20  d     cute_tensor_s2_d2_t  extents {J, B*I}, strides {J, B*I}
 *   ord 5  0x68  0x20  z     cute_tensor_s2_d2_t
 *   ord 6  0x88  0x20  y     cute_tensor_s2_d2_t  extents {J, B*I / mult} (output-gated CUBINs only)
 *
 * o's J stride is the static H*D. Without the bias, d, z and y move up one
 * ordinal. Without the residual, z is absent; such a kernel has no output
 * gate. The launch grid is (ceil(J / tile_j), B*I, 1).
 *
 * Launch ABI attn_epilogue_sm80_tiled_v1 shares this bank. Its kernel splits
 * the C channels over grid.z, so the grid is (ceil(J / tile_j), B*I,
 * C / tile_n).
 */
struct SM80Params
{
  cute_tensor_s2_d1_t o;
  cute_tensor_s2_d2_t g;
  cute_tensor_s0_d0_t w;
  cute_tensor_s0_d0_t bias;
  cute_tensor_s2_d2_t d;
  cute_tensor_s2_d2_t z;
  cute_tensor_s2_d2_t y;
};

constexpr std::size_t sm80_parameter_count(bool has_bias, bool has_output_gate, bool has_residual)
{
  return 4U + (has_bias ? 1U : 0U) + (has_residual ? 1U : 0U) + (has_output_gate ? 1U : 0U);
}

inline constexpr std::size_t kSM80MaxParameterCount = sm80_parameter_count(true, true, true);

/* Returns the number of packed parameters. */
inline std::size_t pack_sm80_kernel_params(
  SM80Params* params,
  bool has_bias,
  bool has_output_gate,
  bool has_residual,
  void* kernel_params[kSM80MaxParameterCount])
{
  std::size_t count = 0;
  kernel_params[count++] = &params->o;
  kernel_params[count++] = &params->g;
  kernel_params[count++] = &params->w;
  if (has_bias)
    kernel_params[count++] = &params->bias;
  kernel_params[count++] = &params->d;
  if (has_residual)
    kernel_params[count++] = &params->z;
  if (has_output_gate)
    kernel_params[count++] = &params->y;
  return count;
}

static_assert(std::is_standard_layout_v<SM80Params>);
static_assert(sizeof(SM80Params::o) == 0x18, "unexpected attention epilogue SM80 o width");
static_assert(sizeof(SM80Params::g) == 0x20, "unexpected attention epilogue SM80 g width");
static_assert(sizeof(SM80Params::w) == 0x08, "unexpected attention epilogue SM80 weight width");
static_assert(sizeof(SM80Params::d) == 0x20, "unexpected attention epilogue SM80 destination width");

static_assert(std::is_standard_layout_v<SM90Params>);
static_assert(sizeof(CUtensorMap) == 0x80);
static_assert(sizeof(SM90Params::o_coord) == 0x08, "unexpected attention epilogue coordinate width");
static_assert(sizeof(SM90Params::bias) == 0x08, "unexpected attention epilogue bias width");
static_assert(sizeof(SM90Params::gj) == 0x04, "unexpected attention epilogue tile-count width");
static_assert(sizeof(SM90Params::tiled_mma) == 0x01, "unexpected attention epilogue MMA token width");

} // namespace bioir::cutedsl::attn_epilogue::abi

namespace bioir::cutedsl::attn_epilogue
{

/* Launch fields generated with the corresponding CUBIN. */
struct KernelSpec
{
  std::int32_t target_sm;
  /* Device ABI generation, distinct from target_sm. */
  std::int32_t kernel_sm;
  std::int32_t heads;
  std::int32_t head_dim;
  std::int32_t channels;
  bool has_bias;
  bool has_output_gate;
  /* Accumulates into z. Without it the kernel writes the bare projection and
   * has no output gate. */
  bool has_residual;
  std::uint32_t tile_j;
  /* Output channels per tile; divides channels. Layers wider than 128
   * channels stream Wo (SM90) or split the channels over grid.z (SM80) in
   * tiles of tile_n, which may cover every channel. */
  std::uint32_t tile_n;
  std::uint32_t num_threads;
  /* The folded-rows tuning anchor this CUBIN was tuned for. */
  std::int32_t bucket;
};

struct KernelConfig
{
  KernelSpec spec;
  EmbeddedCubinImage cubin;
  embedded::CubinImage const* embedded_image;
};

/* Operands with the pair rows folded, each with a unit-stride last mode:
 * o [B*I, J, H, D], g [B*I, J, H*D], w [C, H*D], b [C] for a biased CUBIN,
 * d [B*I, J, C], z like d for a CUBIN with the residual, and
 * y [B*I / mult, J, C] for an output-gated CUBIN, whose row bi serves rows
 * bi * mult to bi * mult + mult - 1. d may alias z.
 */
struct LaunchParams
{
  Tensor4View o;
  Tensor3View g;
  Tensor2View w;
  Tensor1View b;
  Tensor3View d;
  Tensor3View z;
  Tensor3View y;
  std::uint64_t stream{};
};

std::vector<KernelSpec> kernel_specs();

/* Select the CUBIN whose bucket is nearest ``rows``, the lower on a tie. */
KernelConfig make_kernel_config(
  std::int32_t target_sm,
  std::int32_t heads,
  std::int32_t head_dim,
  std::int32_t channels,
  bool has_bias,
  bool has_output_gate,
  std::int32_t rows = 0,
  bool has_residual = true);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace bioir::cutedsl::attn_epilogue

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_ATTN_EPILOGUE_LAUNCHER_H_ */
