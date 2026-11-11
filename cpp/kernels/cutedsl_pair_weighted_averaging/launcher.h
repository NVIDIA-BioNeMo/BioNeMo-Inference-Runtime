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

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_PAIR_WEIGHTED_AVERAGING_LAUNCHER_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_PAIR_WEIGHTED_AVERAGING_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace bioir::cutedsl::pair_weighted_averaging::embedded
{
struct CubinImage;
}

/* The lowered parameters were read from EIATTR_KPARAM_INFO. They are the
 * device-kernel ABI, not the high-level CuTeDSL callable signature.
 */
namespace bioir::cutedsl::pair_weighted_averaging::abi
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

/* Hopper device ABI from EIATTR_KPARAM_INFO (launch ABI pair_weighted_averaging_sm90, kernel_variant 0):
 *
 *   ord  0  0x030  0x80  w_tma       non-executable TMA load atom
 *   ord  1  0x0b0  0x0c  w_coord     CoordTensorS3 {I, N, B}
 *   ord  2  0x0f0  0x80  v_tma       non-executable TMA load atom
 *   ord  3  0x170  0x0c  v_coord     CoordTensorS3 {N, S, B}
 *   ord  4  0x1b0  0x80  g_tma       non-executable TMA load atom
 *   ord  5  0x230  0x0c  g_coord     CoordTensorS3 {I, S, B}
 *   ord  6  0x270  0x80  wo_tma      static extents: no coordinate tensor
 *   ord  7  0x2f0  0x30  output      cute_tensor_s3_d3_t {data, {I, S, B}, {s_i, s_s, s_b}}
 *   ord  8  0x320  0x01  mma1        value WGMMA accumulate flag
 *   ord  9  0x321  0x01  mma2        projection WGMMA accumulate flag
 *   ord 10  0x322  0x01  mma32       slice helper WGMMA accumulate flag
 */
inline constexpr std::size_t kSM90ParameterCount = 11;

struct SM90Params
{
  CUtensorMap w_tma;
  CoordTensorS3 w_coord;
  CUtensorMap v_tma;
  CoordTensorS3 v_coord;
  CUtensorMap g_tma;
  CoordTensorS3 g_coord;
  CUtensorMap wo_tma;
  cute_tensor_s3_d3_t output;
  std::uint8_t mma1;
  std::uint8_t mma2;
  std::uint8_t mma32;
};

inline void pack_sm90_kernel_params(SM90Params* params, void* kernel_params[kSM90ParameterCount])
{
  kernel_params[0] = &params->w_tma;
  kernel_params[1] = &params->w_coord;
  kernel_params[2] = &params->v_tma;
  kernel_params[3] = &params->v_coord;
  kernel_params[4] = &params->g_tma;
  kernel_params[5] = &params->g_coord;
  kernel_params[6] = &params->wo_tma;
  kernel_params[7] = &params->output;
  kernel_params[8] = &params->mma1;
  kernel_params[9] = &params->mma2;
  kernel_params[10] = &params->mma32;
}

static_assert(kSM90ParameterCount == 11);
static_assert(std::is_standard_layout_v<SM90Params>);
static_assert(sizeof(CUtensorMap) == 0x80);
static_assert(sizeof(CoordTensorS3) == 0x0c);

/* Hopper D=8 device ABI from EIATTR_KPARAM_INFO (launch ABI pair_weighted_averaging_sm90, kernel_variant 1):
 *
 *   ord  0  0x030  0x80  w_tma         non-executable TMA load atom
 *   ord  1  0x0b0  0x0c  w_coord       CoordTensorS3 {I, N, B}
 *   ord  2  0x0f0  0x80  v_tma         non-executable TMA load atom
 *   ord  3  0x170  0x0c  v_coord       CoordTensorS3 {N, S, B}
 *   ord  4  0x1b0  0x80  g_tma         non-executable TMA load atom
 *   ord  5  0x230  0x0c  g_coord       CoordTensorS3 {I, S, B}
 *   ord  6  0x270  0x80  wo_tma        static extents: no coordinate tensor
 *   ord  7  0x2f0  0x80  output_tma    non-executable TMA store atom
 *   ord  8  0x370  0x0c  output_coord  CoordTensorS3 {I, S, B}
 *   ord  9  0x37c  0x01  mma_v         value WGMMA accumulate flag
 *   ord 10  0x37d  0x01  mma_p         projection WGMMA accumulate flag
 */
inline constexpr std::size_t kSM90D8ParameterCount = 11;

struct SM90D8Params
{
  CUtensorMap w_tma;
  CoordTensorS3 w_coord;
  CUtensorMap v_tma;
  CoordTensorS3 v_coord;
  CUtensorMap g_tma;
  CoordTensorS3 g_coord;
  CUtensorMap wo_tma;
  CUtensorMap output_tma;
  CoordTensorS3 output_coord;
  std::uint8_t mma_v;
  std::uint8_t mma_p;
};

inline void pack_sm90_d8_kernel_params(SM90D8Params* params, void* kernel_params[kSM90D8ParameterCount])
{
  kernel_params[0] = &params->w_tma;
  kernel_params[1] = &params->w_coord;
  kernel_params[2] = &params->v_tma;
  kernel_params[3] = &params->v_coord;
  kernel_params[4] = &params->g_tma;
  kernel_params[5] = &params->g_coord;
  kernel_params[6] = &params->wo_tma;
  kernel_params[7] = &params->output_tma;
  kernel_params[8] = &params->output_coord;
  kernel_params[9] = &params->mma_v;
  kernel_params[10] = &params->mma_p;
}

static_assert(kSM90D8ParameterCount == 11);
static_assert(std::is_standard_layout_v<SM90D8Params>);

/* Blackwell device ABIs from EIATTR_KPARAM_INFO (launch ABI pair_weighted_averaging_sm100). Each tiled MMA is an
 * SM100TiledMmaToken the host passes clear: 0x20 bytes on one CTA, 0x40 on a CTA pair.
 *
 * kernel_variant 0, the D=32 kernel. Its TMA epilogue (every tile but the direct one) declares:
 *
 *   ord  0  0x000  0x20  mma           value GEMM tiled MMA
 *   ord  1  0x020  0x20  mma_o         Wo GEMM tiled MMA
 *   ord  2  0x040  0x80  w_tma         non-executable TMA load atom
 *   ord  3  0x0c0  0x0c  w_coord       CoordTensorS3 {I, Jp, B}
 *   ord  4  0x100  0x80  v_tma         rank 5: one head's d slice per box
 *   ord  5  0x180  0x0c  v_coord       CoordTensorS3 {N, S, B}
 *   ord  6  0x1c0  0x80  g_tma         rank 5
 *   ord  7  0x240  0x0c  g_coord       CoordTensorS3 {I, S, B}
 *   ord  8  0x280  0x80  wo_tma        static extents: no coordinate tensor
 *   ord  9  0x300  0x80  output_tma    non-executable TMA store atom
 *   ord 10  0x380  0x0c  output_coord  CoordTensorS3 {I, S, B}
 *   ord 11  0x38c  0x04  S             MSA rows
 *
 * The direct tile (TILE_S 4 over 192 threads) loads g and stores the output from registers, so after v_coord it
 * declares wo_tma (ord 6, 0x1c0), g and output as cute_tensor_s3_d3_t {data, {B, S, I}, {s_b, s_s, s_i}} (ords 7
 * and 8, 0x240 and 0x270, 0x30 each) and S (ord 9, 0x2a0). The pair tile (TILE_I 256) is the same kernel on a CTA
 * pair: its 0x40-byte MMA tokens move every later parameter up by 0x40.
 *
 * The split tile (TILE_S 8) stops at the gated value Z [B, S, I, H*D] and declares one MMA token, then w_tma and
 * w_coord (ords 1 and 2, 0x040 and 0x0c0), v_tma and v_coord (ords 3 and 4, 0x100 and 0x180), g and Z as
 * cute_tensor_s3_d3_t (ords 5 and 6, 0x190 and 0x1c0) and S (ord 7, 0x1f0).
 */
inline constexpr std::size_t kSM100D32ParameterCount = 12;

/* The D=32 kernel's device kernels: the TMA epilogue, the register epilogue of the direct and pair tiles, and the
 * split tile's gated value.
 */
enum class SM100D32Kernel : std::uint8_t
{
  kTmaEpilogue,
  kRegisterEpilogue,
  kGatedValue,
};

struct SM100D32Params
{
  SM100TiledMmaToken mma;
  SM100TiledMmaToken mma_o;
  CUtensorMap w_tma;
  CoordTensorS3 w_coord;
  CUtensorMap v_tma;
  CoordTensorS3 v_coord;
  CUtensorMap g_tma;
  CoordTensorS3 g_coord;
  CUtensorMap wo_tma;
  CUtensorMap output_tma;
  CoordTensorS3 output_coord;
  cute_tensor_s3_d3_t g;
  cute_tensor_s3_d3_t output;
  std::int32_t S;
};

/* Packs the kernel's own bank into the first entries; the direct and split tiles leave the rest unset. */
inline void pack_sm100_d32_kernel_params(
  SM100D32Params* params, SM100D32Kernel kernel, void* kernel_params[kSM100D32ParameterCount])
{
  std::size_t count = 0;
  kernel_params[count++] = &params->mma;
  if (kernel != SM100D32Kernel::kGatedValue)
    kernel_params[count++] = &params->mma_o;
  kernel_params[count++] = &params->w_tma;
  kernel_params[count++] = &params->w_coord;
  kernel_params[count++] = &params->v_tma;
  kernel_params[count++] = &params->v_coord;
  switch (kernel)
  {
  case SM100D32Kernel::kTmaEpilogue:
    kernel_params[count++] = &params->g_tma;
    kernel_params[count++] = &params->g_coord;
    kernel_params[count++] = &params->wo_tma;
    kernel_params[count++] = &params->output_tma;
    kernel_params[count++] = &params->output_coord;
    break;
  case SM100D32Kernel::kRegisterEpilogue:
    kernel_params[count++] = &params->wo_tma;
    kernel_params[count++] = &params->g;
    kernel_params[count++] = &params->output;
    break;
  case SM100D32Kernel::kGatedValue:
    kernel_params[count++] = &params->g;
    kernel_params[count++] = &params->output;
    break;
  }
  kernel_params[count++] = &params->S;
}

/* kernel_variant 1, the D=8 kernel, one bank for every tile and cluster. It loads v transposed, [B, S, H*D, N]. A CTA
 * pair's 0x40-byte MMA tokens move every later parameter up by 0x40.
 *
 *   ord  0  0x000  0x20  mma_v         value GEMM tiled MMA
 *   ord  1  0x020  0x20  mma_p         projection tiled MMA, Z from TMEM
 *   ord  2  0x040  0x20  mma_ps        projection tiled MMA, Z rows past TMEM from shared memory
 *   ord  3  0x080  0x80  w_tma         over the logical N: the loads zero-fill w's pad
 *   ord  4  0x100  0x0c  w_coord       CoordTensorS3 {I, N, B}
 *   ord  5  0x140  0x80  v_tma         rank 5: one head's d rows of v^T per box
 *   ord  6  0x1c0  0x0c  v_coord       CoordTensorS3 {N, S, B}
 *   ord  7  0x200  0x80  wo_tma        static extents: no coordinate tensor
 *   ord  8  0x280  0x80  g_tma         read by the tiles that stage g in shared memory; every image declares it
 *   ord  9  0x300  0x0c  g_coord       CoordTensorS3 {I, S, B}
 *   ord 10  0x310  0x30  g             cute_tensor_s3_d3_t, the other tiles' per-head loads
 *   ord 11  0x340  0x30  output        cute_tensor_s3_d3_t
 */
inline constexpr std::size_t kSM100D8ParameterCount = 12;

struct SM100D8Params
{
  SM100TiledMmaToken mma_v;
  SM100TiledMmaToken mma_p;
  SM100TiledMmaToken mma_ps;
  CUtensorMap w_tma;
  CoordTensorS3 w_coord;
  CUtensorMap v_tma;
  CoordTensorS3 v_coord;
  CUtensorMap wo_tma;
  CUtensorMap g_tma;
  CoordTensorS3 g_coord;
  cute_tensor_s3_d3_t g;
  cute_tensor_s3_d3_t output;
};

inline void pack_sm100_d8_kernel_params(SM100D8Params* params, void* kernel_params[kSM100D8ParameterCount])
{
  kernel_params[0] = &params->mma_v;
  kernel_params[1] = &params->mma_p;
  kernel_params[2] = &params->mma_ps;
  kernel_params[3] = &params->w_tma;
  kernel_params[4] = &params->w_coord;
  kernel_params[5] = &params->v_tma;
  kernel_params[6] = &params->v_coord;
  kernel_params[7] = &params->wo_tma;
  kernel_params[8] = &params->g_tma;
  kernel_params[9] = &params->g_coord;
  kernel_params[10] = &params->g;
  kernel_params[11] = &params->output;
}

static_assert(kSM100D8ParameterCount == 12);
static_assert(std::is_standard_layout_v<SM100D32Params>);
static_assert(std::is_standard_layout_v<SM100D8Params>);

} // namespace bioir::cutedsl::pair_weighted_averaging::abi

namespace bioir::cutedsl::pair_weighted_averaging
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

} // namespace bioir::cutedsl::pair_weighted_averaging

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_PAIR_WEIGHTED_AVERAGING_LAUNCHER_H_ */
