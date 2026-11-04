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

/* SM90 and SM100/SM103 TriMul KF K1 CUBIN configuration, device ABIs, and launcher interface. */

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_TRIMUL_KF_K1_LAUNCHER_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_TRIMUL_KF_K1_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace bioir::cutedsl::trimul_kf_k1::embedded
{
struct CubinImage;
}

/* Hopper device ABI trimul_kf_k1_sm90_v4, read from EIATTR_KPARAM_INFO for variant K1_0:
 *
 *   ord  0  0x030  0x80  x_tma          CUtensorMap
 *   ord  1  0x0b0  0x04  x_coord        CoordTensorS1   rows
 *   ord  2  0x0f0  0x80  w_proj_tma     CUtensorMap
 *   ord  3  0x170  0x80  w_gate_tma     CUtensorMap
 *   ord  4  0x1f0  0x80  a_tma          CUtensorMap
 *   ord  5  0x270  0x08  a_coord        CoordTensorS2   (N * P, B)
 *   ord  6  0x2b0  0x80  b_tma          CUtensorMap
 *   ord  7  0x330  0x08  b_coord        CoordTensorS2   (N * P, B)
 *   ord  8  0x338  0x10  seqlen         cute_tensor_s1_d0_t
 *   ord  9  0x348  0x08  vec            cute_tensor_s0_d0_t
 *   ord 10  0x350  0x01  tiled_mma      std::uint8_t
 *   ord 11  0x354  0x04  n              std::int32_t
 *   ord 12  0x358  0x04  num_tiles      std::int32_t
 *   ord 13  0x35c  0x04  eps            float
 *   ord 14  0x360  0x04  rows           std::int32_t
 *   ord 15  0x370  0x80  x_chunk_tma    CUtensorMap
 *   ord 16  0x3f0  0x08  x_chunk_coord  CoordTensorS2   (N, rows / N)
 *
 * The ping-pong variants K1_1 and K1_2 read one interleaved [4D, C] weight fold through w_proj_tma,
 * so w_gate_tma is absent, and a second MMA token, mma_out, follows tiled_mma. K1_2 also passes its
 * row-statistics view, stats (extent = rows), right after vec. The weights have static shapes, so
 * their coordinate tensors lower to no parameter; the shared-memory layouts are compile-time objects
 * and occupy no slot either.
 *
 * a and b are [B, D, P, P] planes stored flat over their first N * P positions: P = N for a dense
 * image; P = N rounded up to 64 for a padded one, which loads x one 64-position row chunk at a time
 * through x_chunk_tma. A dense image declares the chunk-load pair too, never reads it, and receives
 * x_tma and x_coord in it.
 *
 * v4 tiles each batch on its own (v3 tiled a launch's rows flat): num_tiles counts every batch's
 * whole 128-row tiles, or pairs of row chunks, so one launch serves any number of batches.
 *
 * Each 0x80 TMA slot carries CuTe DSL 4.5.2's 0x40-byte non-executable CopyAtom payload followed
 * by zero padding: plain loads of x, its row chunks and the weights, and stores of a and b.
 */
namespace bioir::cutedsl::trimul_kf_k1::abi
{

struct SM90Params
{
  CUtensorMap x_tma;
  CoordTensorS1 x_coord;
  CUtensorMap w_proj_tma;
  CUtensorMap w_gate_tma;
  CUtensorMap a_tma;
  CoordTensorS2 a_coord;
  CUtensorMap b_tma;
  CoordTensorS2 b_coord;
  cute_tensor_s1_d0_t seqlen;
  cute_tensor_s0_d0_t vec;
  cute_tensor_s1_d0_t stats;
  std::uint8_t tiled_mma;
  std::uint8_t mma_out;
  std::int32_t n;
  std::int32_t num_tiles;
  float eps;
  std::int32_t rows;
  CUtensorMap x_chunk_tma;
  CoordTensorS2 x_chunk_coord;
};

constexpr std::size_t sm90_parameter_count(std::int32_t kernel_variant)
{
  return kernel_variant == 2 ? 18U : 17U;
}

inline constexpr std::size_t kSM90MaxParameterCount = 18;

/* params and kernel_params must stay alive until the launch call returns. */
inline std::size_t pack_sm90_kernel_params(
  SM90Params* params, std::int32_t kernel_variant, bool padded, void* kernel_params[kSM90MaxParameterCount])
{
  bool const pingpong = kernel_variant != 0;
  std::size_t count = 0;
  kernel_params[count++] = &params->x_tma;
  kernel_params[count++] = &params->x_coord;
  kernel_params[count++] = &params->w_proj_tma;
  if (!pingpong)
    kernel_params[count++] = &params->w_gate_tma;
  kernel_params[count++] = &params->a_tma;
  kernel_params[count++] = &params->a_coord;
  kernel_params[count++] = &params->b_tma;
  kernel_params[count++] = &params->b_coord;
  kernel_params[count++] = &params->seqlen;
  kernel_params[count++] = &params->vec;
  if (kernel_variant == 2)
    kernel_params[count++] = &params->stats;
  kernel_params[count++] = &params->tiled_mma;
  if (pingpong)
    kernel_params[count++] = &params->mma_out;
  kernel_params[count++] = &params->n;
  kernel_params[count++] = &params->num_tiles;
  kernel_params[count++] = &params->eps;
  kernel_params[count++] = &params->rows;
  if (padded)
  {
    kernel_params[count++] = &params->x_chunk_tma;
    kernel_params[count++] = &params->x_chunk_coord;
  }
  else
  {
    kernel_params[count++] = &params->x_tma;
    kernel_params[count++] = &params->x_coord;
  }
  return count;
}

/* The driver takes one pointer per parameter, so host struct padding is irrelevant; only each
 * parameter's width must match the CUBIN.
 */
static_assert(std::is_standard_layout_v<SM90Params>);
static_assert(sizeof(CUtensorMap) == 0x80 && alignof(CUtensorMap) >= 64);
static_assert(sizeof(SM90Params::x_coord) == 0x04, "unexpected trimul KF K1 row-coordinate width");
static_assert(sizeof(SM90Params::a_coord) == 0x08, "unexpected trimul KF K1 a/b coordinate width");
static_assert(sizeof(SM90Params::seqlen) == 0x10, "unexpected trimul KF K1 seqlen width");
static_assert(sizeof(SM90Params::vec) == 0x08, "unexpected trimul KF K1 vec width");
static_assert(sizeof(SM90Params::stats) == 0x10, "unexpected trimul KF K1 stats width");
static_assert(sizeof(SM90Params::tiled_mma) == 0x01, "unexpected trimul KF K1 MMA token width");
static_assert(sizeof(SM90Params::eps) == 0x04, "unexpected trimul KF K1 eps width");
static_assert(sizeof(SM90Params::x_chunk_coord) == 0x08, "unexpected trimul KF K1 row-chunk coordinate width");
static_assert(sm90_parameter_count(0) == 17U && sm90_parameter_count(1) == 17U && sm90_parameter_count(2) == 18U);

/* Blackwell device ABI trimul_kf_k1_sm100_v1, read from EIATTR_KPARAM_INFO. K1_B0 (14 parameters):
 *
 *   ord  0  0x000  0x03  tiled_mma      SM100TiledMma
 *   ord  1  0x040  0x80  x_tma          CUtensorMap
 *   ord  2  0x0c0  0x08  x_coord        CoordTensorS2   (n * n, nb); padded: CoordTensorS3 (n, n, nb), 0x0c
 *   ord  3  0x100  0x80  w_proj_tma     CUtensorMap
 *   ord  4  0x180  0x80  w_gate_tma     CUtensorMap
 *   ord  5  0x200  0x80  planes_tma     CUtensorMap
 *   ord  6  0x280  0x08  planes_coord   CoordTensorS2   (n * P, nb)
 *   ord  7  0x288  0x10  seqlen         cute_tensor_s1_d0_t
 *   ord  8  0x298  0x08  vec            cute_tensor_s0_d0_t
 *   ord  9  0x2a0  0x04  n              std::int32_t
 *   ord 10  0x2a4  0x04  P              std::int32_t
 *   ord 11  0x2a8  0x04  work_per_batch std::int32_t
 *   ord 12  0x2ac  0x04  num_work       std::int32_t
 *   ord 13  0x2b0  0x04  eps            float
 *
 * K1_B1 (16 parameters) stores a and b apart and counts tiles, not work items:
 *
 *   tiled_mma, x_tma, x_coord (CoordTensorS2: (n * n, nb), padded (n, nb * n)), w_proj_tma, w_gate_tma,
 *   a_tma, a_coord, b_tma, b_coord (CoordTensorS2 (n * n, nb), padded CoordTensorS3 (n, n, nb)), seqlen,
 *   vec, n, nb, tiles_per_batch, num_tiles, eps
 *
 * The weights have static shapes, so their coordinate tensors lower to no parameter. Every TMA slot
 * carries CuTe DSL 4.5.2's tagged Blackwell atom (finalize_sm100_tma_atom). The tiled MMA's three bytes
 * are runtime state the kernel initializes itself; the launch passes zeros.
 */
struct SM100TiledMma
{
  std::uint8_t fields[3]{};
};

struct SM100Params
{
  SM100TiledMma tiled_mma;
  CUtensorMap x_tma;
  CoordTensorS2 x_coord2;
  CoordTensorS3 x_coord3;
  CUtensorMap w_proj_tma;
  CUtensorMap w_gate_tma;
  CUtensorMap planes_tma;
  CoordTensorS2 planes_coord;
  CUtensorMap a_tma;
  CUtensorMap b_tma;
  CoordTensorS2 ab_coord2;
  CoordTensorS3 ab_coord3;
  cute_tensor_s1_d0_t seqlen;
  cute_tensor_s0_d0_t vec;
  std::int32_t n;
  std::int32_t P;
  std::int32_t nb;
  std::int32_t per_batch;
  std::int32_t total;
  float eps;
};

constexpr std::size_t sm100_parameter_count(std::int32_t kernel_variant)
{
  return kernel_variant == 0 ? 14U : 16U;
}

inline constexpr std::size_t kSM100MaxParameterCount = 16;

/* params and kernel_params must stay alive until the launch call returns. */
inline std::size_t pack_sm100_kernel_params(
  SM100Params* params, std::int32_t kernel_variant, bool padded, void* kernel_params[kSM100MaxParameterCount])
{
  std::size_t count = 0;
  kernel_params[count++] = &params->tiled_mma;
  kernel_params[count++] = &params->x_tma;
  if (kernel_variant == 0)
  {
    kernel_params[count++] = padded ? static_cast<void*>(&params->x_coord3) : static_cast<void*>(&params->x_coord2);
    kernel_params[count++] = &params->w_proj_tma;
    kernel_params[count++] = &params->w_gate_tma;
    kernel_params[count++] = &params->planes_tma;
    kernel_params[count++] = &params->planes_coord;
    kernel_params[count++] = &params->seqlen;
    kernel_params[count++] = &params->vec;
    kernel_params[count++] = &params->n;
    kernel_params[count++] = &params->P;
  }
  else
  {
    void* ab_coord = padded ? static_cast<void*>(&params->ab_coord3) : static_cast<void*>(&params->ab_coord2);
    kernel_params[count++] = &params->x_coord2;
    kernel_params[count++] = &params->w_proj_tma;
    kernel_params[count++] = &params->w_gate_tma;
    kernel_params[count++] = &params->a_tma;
    kernel_params[count++] = ab_coord;
    kernel_params[count++] = &params->b_tma;
    kernel_params[count++] = ab_coord;
    kernel_params[count++] = &params->seqlen;
    kernel_params[count++] = &params->vec;
    kernel_params[count++] = &params->n;
    kernel_params[count++] = &params->nb;
  }
  kernel_params[count++] = &params->per_batch;
  kernel_params[count++] = &params->total;
  kernel_params[count++] = &params->eps;
  return count;
}

static_assert(std::is_standard_layout_v<SM100Params>);
static_assert(sizeof(SM100TiledMma) == 0x03 && alignof(SM100TiledMma) == 1);
static_assert(sizeof(CoordTensorS2) == 0x08 && sizeof(CoordTensorS3) == 0x0c);
static_assert(sizeof(cute_tensor_s1_d0_t) == 0x10 && sizeof(cute_tensor_s0_d0_t) == 0x08);
static_assert(sm100_parameter_count(0) == 14U && sm100_parameter_count(1) == kSM100MaxParameterCount);

} // namespace bioir::cutedsl::trimul_kf_k1::abi

namespace bioir::cutedsl::trimul_kf_k1
{

enum class DType : std::uint8_t
{
  kBFloat16,
};

/* The SM100 K1_B0 tile (the configs' TrimulKFK1Tile); all zeros for every other image. */
struct KernelTile
{
  std::uint32_t bnc{};
  std::uint32_t epi_n{};
  std::uint32_t ctas{};
  std::uint32_t w_slots{};
  std::uint32_t s_stages{};
  std::uint32_t store_warps{};
  std::uint32_t epi_groups{};
};

inline bool operator==(KernelTile const& left, KernelTile const& right)
{
  return left.bnc == right.bnc && left.epi_n == right.epi_n && left.ctas == right.ctas && left.w_slots == right.w_slots
    && left.s_stages == right.s_stages && left.store_warps == right.store_warps && left.epi_groups == right.epi_groups;
}

/* Launch fields generated with the corresponding CUBIN. On SM90 (kernel_sm 90) it runs the configs'
 * K1_<kernel_variant>: variants 1 and 2 read the interleaved weight fold, and variant 2 also writes the
 * row statistics. On SM100 and SM103 (kernel_sm 100) kernel_variant 0 is K1_B0, which takes a tile, and 1
 * is K1_B1. padded writes a and b as [N, N] views of [P, P] planes.
 */
struct KernelSpec
{
  std::int32_t target_sm;
  std::int32_t kernel_sm;
  std::int32_t C;
  std::int32_t D;
  std::int32_t kernel_variant;
  bool padded;
  std::uint32_t num_threads;
  KernelTile tile;
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
    throw std::invalid_argument("trimul KF K1 CUBIN has no dynamic shared-memory metadata");
  return config.cubin.dynamic_smem_bytes;
}

/* K1 operands, all flat and contiguous: x is [rows * C] with rows = nb * n * n; seqlen holds one
 * int32 prefix length per (b, i) row; w_in is the [2D * C] projection fold, or the [4D * C]
 * interleaved fold; w_gate_in the [2D * C] gate fold, read by K1_0 and both SM100 variants; vec_in the
 * fp32 [8D] folded bias terms, which carry the LayerNorm shift and any projection biases; a and b the
 * bf16 [nb * D * P * P] channel-major outputs; planes, read by SM100 K1_B0 in their place, the
 * [2 * nb * D * P * P] a planes of every batch followed by the b planes; stats the fp32 row statistics,
 * written by K1_2 alone and sized for rows rounded up to whole 128-row tiles.
 */
struct LaunchParams
{
  FlatTensorView x;
  Tensor1View seqlen;
  Tensor1View w_in;
  Tensor1View w_gate_in;
  Tensor1View vec_in;
  FlatTensorView a;
  FlatTensorView b;
  FlatTensorView planes;
  FlatTensorView stats;
  std::int32_t rows{};
  std::int32_t n{};
  std::int32_t nb{};
  float eps{};
  std::uint64_t stream{};
};

std::vector<KernelSpec> kernel_specs();

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  DType dtype,
  std::int32_t C,
  std::int32_t D,
  std::int32_t kernel_variant,
  bool padded,
  KernelTile const& tile = {});

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace bioir::cutedsl::trimul_kf_k1

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_TRIMUL_KF_K1_LAUNCHER_H_ */
