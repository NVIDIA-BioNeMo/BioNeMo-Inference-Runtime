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

/* SM90 transition with its LayerNorm: CUBIN configuration, device ABI, and launcher interface. */

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_TRANSITION_MLP_W_NORM_LAUNCHER_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_TRANSITION_MLP_W_NORM_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace bioir::cutedsl::transition_mlp_w_norm::embedded
{
struct CubinImage;
}

/* Hopper device ABI transition_mlp_w_norm_sm90_v3, read from EIATTR_KPARAM_INFO:
 *
 *   ord  0  0x00  0x10  x          FlatTensor            bf16 [rows * width]
 *   ord  1  0x10  0x08  mask       std::uint64_t         address of bf16 [rows], or 0: every row kept
 *   ord  2  0x18  0x10  image      FlatTensor            bf16 [(blocks + 1) * width * hidden]
 *   ord  3  0x28  0x08  vec        cute_tensor_s0_d0_t   fp32 [blocks * hidden + width], static extent
 *   ord  4  0x30  0x10  output     FlatTensor            bf16 [rows * width]
 *   ord  5  0x40  0x08  n_valid    std::uint64_t         address of int32 [n_batch], or 0: flat rows
 *   ord  6  0x48  0x04  rows       std::int32_t
 *   ord  7  0x4c  0x04  num_tiles  std::int32_t          ceil(rows / tile_m)
 *   ord  8  0x50  0x04  n          std::int32_t          pair map side, 0 for flat rows
 *   ord  9  0x54  0x04  n_batch    std::int32_t          pair map items, 0 for flat rows
 *   ord 10  0x58  0x04  eps        float
 *   ord 11  0x5c  0x04  residual   std::int32_t          1: x + mask * update; 0: mask * update
 *   ord 12  0x60  0x04  split      std::int32_t          CTAs per row tile: 1, or a cluster of split CTAs
 *   ord 13  0x64  0x04  tile_wgs   std::int32_t          warpgroups per row tile; the CTA's others idle
 *
 * blocks is 1 for ReLU and 2 for SwiGLU, whose W1 makes a value and a gate per hidden unit. The device code never
 * reads num_tiles, and it reads the extents only as the compiled signature declares them: each CTA walks its tiles
 * from rows, or from the per-item table it builds from n_valid.
 */
namespace bioir::cutedsl::transition_mlp_w_norm::abi
{

/* A flat operand compiled with a 64-bit dynamic extent: the address, then the extent. */
struct FlatTensor
{
  CUdeviceptr data;
  std::int64_t extent;
};

struct SM90Params
{
  FlatTensor x;
  std::uint64_t mask;
  FlatTensor image;
  cute_tensor_s0_d0_t vec;
  FlatTensor output;
  std::uint64_t n_valid;
  std::int32_t rows;
  std::int32_t num_tiles;
  std::int32_t n;
  std::int32_t n_batch;
  float eps;
  std::int32_t residual;
  std::int32_t split;
  std::int32_t tile_wgs;
};

inline constexpr std::size_t kSM90ParameterCount = 14;

/* params and kernel_params must stay alive until the launch call returns. */
inline void pack_sm90_kernel_params(SM90Params* params, void* kernel_params[kSM90ParameterCount])
{
  kernel_params[0] = &params->x;
  kernel_params[1] = &params->mask;
  kernel_params[2] = &params->image;
  kernel_params[3] = &params->vec;
  kernel_params[4] = &params->output;
  kernel_params[5] = &params->n_valid;
  kernel_params[6] = &params->rows;
  kernel_params[7] = &params->num_tiles;
  kernel_params[8] = &params->n;
  kernel_params[9] = &params->n_batch;
  kernel_params[10] = &params->eps;
  kernel_params[11] = &params->residual;
  kernel_params[12] = &params->split;
  kernel_params[13] = &params->tile_wgs;
}

/* The driver takes one pointer per parameter, so only each parameter's width must match the CUBIN; the host layout
 * mirrors the device offsets anyway.
 */
static_assert(std::is_standard_layout_v<SM90Params>);
static_assert(sizeof(FlatTensor) == 0x10 && offsetof(FlatTensor, extent) == 0x08);
static_assert(sizeof(cute_tensor_s0_d0_t) == 0x08);
static_assert(offsetof(SM90Params, mask) == 0x10);
static_assert(offsetof(SM90Params, image) == 0x18);
static_assert(offsetof(SM90Params, vec) == 0x28);
static_assert(offsetof(SM90Params, output) == 0x30);
static_assert(offsetof(SM90Params, n_valid) == 0x40);
static_assert(offsetof(SM90Params, rows) == 0x48);
static_assert(offsetof(SM90Params, num_tiles) == 0x4c);
static_assert(offsetof(SM90Params, n) == 0x50);
static_assert(offsetof(SM90Params, n_batch) == 0x54);
static_assert(offsetof(SM90Params, eps) == 0x58);
static_assert(offsetof(SM90Params, residual) == 0x5c);
static_assert(offsetof(SM90Params, split) == 0x60);
static_assert(offsetof(SM90Params, tile_wgs) == 0x64);

/* Blackwell transition_mlp_w_norm_sm100_v1, checked against EIATTR_KPARAM_INFO:
 * ord 0/1: 3-byte tiled MMA states; ord 2: X atom at 0x40 (128 B);
 * ord 3: X row extent at 0xc0 (4 B); ord 4/5: static W1/W2 atoms at
 * 0x100/0x180 (128 B each); ord 6: raw X at 0x200 (16 B);
 * ord 7/8: mask/vector pointers at 0x210/0x218 (8 B each);
 * ord 9: output at 0x220 (16 B); ord 10/11: eps/residual at 0x230/0x234.
 * W1 and W2 have static shapes, so neither has a coordinate parameter.
 */
/* The pinned MmaF16BF16Op builds its atom from three Boolean(false) values.
 * PTX reads only the low bits of bytes 1 and 2 (operand negation). This
 * kernel overwrites ACCUMULATE before every MMA, so byte 0 is never read.
 * All-zero bytes therefore reproduce the source constructor's state.
 */
struct SM100TiledMmaDescriptor
{
  std::uint8_t fields[3]{};
};

struct SM100Params
{
  SM100TiledMmaDescriptor tiled_mma_g1;
  SM100TiledMmaDescriptor tiled_mma_g2;
  CUtensorMap x_tma;
  CoordTensorS1 x_coord;
  CUtensorMap w1_tma;
  CUtensorMap w2_tma;
  cute_tensor_s1_d0_t x;
  std::uint64_t mask;
  cute_tensor_s0_d0_t vec;
  cute_tensor_s1_d0_t output;
  float eps;
  std::int32_t residual;
};

inline constexpr std::size_t kSM100ParameterCount = 12;
inline constexpr std::size_t kSM100OutputParameterCount = 14;

inline void pack_sm100_kernel_params(SM100Params* params, void* kernel_params[kSM100ParameterCount])
{
  kernel_params[0] = &params->tiled_mma_g1;
  kernel_params[1] = &params->tiled_mma_g2;
  kernel_params[2] = &params->x_tma;
  kernel_params[3] = &params->x_coord;
  kernel_params[4] = &params->w1_tma;
  kernel_params[5] = &params->w2_tma;
  kernel_params[6] = &params->x;
  kernel_params[7] = &params->mask;
  kernel_params[8] = &params->vec;
  kernel_params[9] = &params->output;
  kernel_params[10] = &params->eps;
  kernel_params[11] = &params->residual;
}

/* Output ABI v2 inserts the store tensor map and its row extent after W2:
 * ordinal 6: 128 bytes at 0x200; ordinal 7: 4 bytes at 0x280.
 * The remaining v1 arguments move to ordinals 8..13 (0x288..0x2bc).
 */
inline void pack_sm100_output_kernel_params(
  SM100Params* params,
  CUtensorMap* output_tma,
  CoordTensorS1* output_coord,
  void* kernel_params[kSM100OutputParameterCount])
{
  void* original[kSM100ParameterCount]{};
  pack_sm100_kernel_params(params, original);
  for (std::size_t i = 0; i < 6; ++i)
    kernel_params[i] = original[i];
  kernel_params[6] = output_tma;
  kernel_params[7] = output_coord;
  for (std::size_t i = 6; i < kSM100ParameterCount; ++i)
    kernel_params[i + 2] = original[i];
}

static_assert(std::is_standard_layout_v<SM100Params>);
static_assert(sizeof(SM100TiledMmaDescriptor) == 3);
static_assert(sizeof(CoordTensorS1) == 4);
static_assert(sizeof(cute_tensor_s1_d0_t) == 16);
static_assert(offsetof(SM100Params, tiled_mma_g2) == 0x3);
/* CUDA headers may align CUtensorMap more strictly than the device bank. The
 * Driver receives a pointer per argument, so each value's size, not this
 * aggregate's inter-field padding, is the ABI contract.
 */
static_assert(sizeof(CUtensorMap) == 128);
static_assert(alignof(CUtensorMap) >= 64);
static_assert(sizeof(cute_tensor_s0_d0_t) == 8);

} // namespace bioir::cutedsl::transition_mlp_w_norm::abi

namespace bioir::cutedsl::transition_mlp_w_norm
{

enum class DType : std::uint8_t
{
  kBFloat16,
};

/* The hidden activation: relu(h), or silu(gate) * value with W1's first hidden rows making the value and the next the
 * gate.
 */
enum class Activation : std::uint8_t
{
  kReLU,
  kSiLUGate,
};

/* Launch fields generated with the corresponding CUBIN: a persistent grid of up to ctas_per_sm CTAs per SM, each of
 * num_threads threads (num_threads / 128 warpgroups), walks tile_m-row tiles. layouts holds a bit per (tile_wgs,
 * split) pair the kernel takes, bit 8 log2(tile_wgs) + split - 1: a row tile of tile_wgs warpgroups, tile_m tile_wgs
 * / (num_threads / 128) rows, the CTA's other warpgroups idle, and split CTAs per tile, a cluster of split CTAs when
 * it is above 1. Each shape has two images: one takes every layout, and the persistent one only the full tile at
 * split 1, with that layout fixed while tracing so that its code carries none of the others' arithmetic.
 */
struct KernelSpec
{
  std::int32_t target_sm;
  std::int32_t kernel_sm;
  std::int32_t width;
  std::int32_t hidden;
  Activation activation;
  std::uint32_t num_threads;
  std::uint32_t tile_m;
  std::uint32_t ctas_per_sm;
  std::uint32_t layouts;
  bool is_persistent;
};

struct KernelConfig
{
  KernelSpec spec;
  DType dtype;
  EmbeddedCubinImage cubin;
  embedded::CubinImage const* embedded_image;
};

std::uint32_t dynamic_smem_bytes(KernelConfig const& config);

/* Operands, all flat and contiguous: x and output are rows of width bf16 values. image is the bf16 weight image and
 * vec the fp32 biases, both as the op packs them. mask holds one bf16 per row and scales the update; a null view
 * keeps every row. A row whose mask is 0 comes out equal to x with the residual and 0 without it.
 *
 * A null n_valid view computes every row. Otherwise x is the pair maps of n_batch items, rows = n_batch * n * n, and
 * n_valid holds one int32 per item: only the rows (b, i, j) with i, j < n_valid[b] are read and written (the kernel
 * clamps each length to 0 .. n), so the others keep what output held. output is x itself (in place) or overlaps no
 * operand.
 *
 * split and tile_wgs choose the launch layout, a pair KernelSpec::layouts holds; tile_wgs 0 is the CTA's every
 * warpgroup. split 1 with every warpgroup is the persistent grid, and the output of split 1 does not depend on
 * tile_wgs. A split above 1 sums the tile's partials in a fixed order, so its output is the same every launch. On
 * SM100 a split counts CTA pairs: the image that takes every layout runs each row tile on a cluster of 2 split CTAs.
 */
struct LaunchParams
{
  FlatTensorView x;
  FlatTensorView mask;
  FlatTensorView image;
  Tensor1View vec;
  FlatTensorView output;
  Tensor1View n_valid;
  std::int64_t rows{};
  std::int32_t n{};
  std::int32_t n_batch{};
  float eps{};
  bool residual{true};
  std::int32_t split{1};
  std::int32_t tile_wgs{};
  std::uint64_t stream{};
};

std::vector<KernelSpec> kernel_specs();

/* Whether the kernel takes split CTAs per row tile of tile_wgs warpgroups (0: every warpgroup). */
bool supports_layout(KernelSpec const& spec, std::int32_t split, std::int32_t tile_wgs);

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  DType dtype,
  std::int32_t width,
  std::int32_t hidden,
  Activation activation,
  bool persistent = false);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace bioir::cutedsl::transition_mlp_w_norm

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_TRANSITION_MLP_W_NORM_LAUNCHER_H_ */
