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

#include "launcher.h"

#include "trimul_kf_k1_registry.h"

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace bioir::cutedsl::trimul_kf_k1
{
namespace
{

constexpr char kSM90LaunchAbi[] = "trimul_kf_k1_sm90_v4";
constexpr char kSM100LaunchAbi[] = "trimul_kf_k1_sm100_v1";
/* The SM100 K1_B0 variant, whose image carries a tile; K1_B1 is variant 1. */
constexpr std::int32_t kSM100TiledVariant = 0;
/* Rows per K1 tile, which the persistent CTAs stride over; a tile never straddles two batches. */
constexpr std::uint64_t kTileRows = 128;
/* Rows, and positions per a/b plane, one launch addresses: TMA coordinates are int32. */
constexpr std::int64_t kMaxLaunchRows = std::numeric_limits<std::int32_t>::max() - static_cast<std::int64_t>(kTileRows);
/* The ping-pong K1 variant that hands its row statistics to K3. */
constexpr std::int32_t kStatsVariant = 2;
/* The kernel's AB_PAD_ALIGN and AB_CHUNK, which the CUBIN build checks against these values. */
constexpr std::int64_t kAbPadAlign = 64;
constexpr std::int64_t kAbChunk = 64;

/* A positive device-side int32 count. */
std::int32_t checked_i32(std::uint64_t value, char const* name)
{
  if (value == 0 || value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::invalid_argument(std::string(name) + " does not fit a positive int32");
  return static_cast<std::int32_t>(value);
}

std::int64_t ab_pitch(std::int64_t n, bool padded)
{
  return padded ? (n + kAbPadAlign - 1) / kAbPadAlign * kAbPadAlign : n;
}

/* Bare device pointers require explicit cross-device validation. */
void check_device(std::int32_t operand_device, std::int32_t device, char const* name)
{
  if (operand_device == kUnknownDevice)
  {
    throw std::invalid_argument(
      std::string(name) + " has no CUDA device: it is a host tensor, or its view was built without one");
  }
  if (operand_device != device)
  {
    throw std::invalid_argument(
      std::string(name) + " is on CUDA device " + std::to_string(operand_device) + " but the launch targets device "
      + std::to_string(device));
  }
}

/* A flat operand holding exactly ``elements`` values. */
void validate_flat(
  Tensor1View const& view, std::int64_t elements, std::uint64_t alignment, std::int32_t device, char const* name)
{
  validate_tensor(view, name, alignment);
  check_device(view.device, device, name);
  if (view.shape[0] != elements)
  {
    throw std::invalid_argument(
      std::string(name) + " must hold " + std::to_string(elements) + " elements, got " + std::to_string(view.shape[0]));
  }
}

void validate_flat(
  FlatTensorView const& view, std::int64_t elements, std::uint64_t alignment, std::int32_t device, char const* name)
{
  validate_flat_tensor(view, name, alignment);
  check_device(view.device, device, name);
  if (view.extent != elements)
  {
    throw std::invalid_argument(
      std::string(name) + " must hold " + std::to_string(elements) + " elements, got " + std::to_string(view.extent));
  }
}

void validate_image(KernelConfig const& config)
{
  if (config.cubin.data == nullptr || config.cubin.size == 0)
    throw std::invalid_argument("embedded CUBIN image must not be empty");
  if (config.cubin.variant_id == nullptr || config.cubin.variant_id[0] == '\0')
    throw std::invalid_argument("embedded CUBIN variant_id must not be empty");
  if (config.cubin.kernel_symbol == nullptr || config.cubin.kernel_symbol[0] == '\0')
    throw std::invalid_argument("kernel_symbol must not be empty");
  if (config.embedded_image == nullptr)
    throw std::invalid_argument("trimul KF K1 CUBIN is missing its registry record");
  if (!cubin_supports_sm(config.cubin, config.spec.target_sm))
    throw std::invalid_argument(
      "embedded CUBIN does not support configured device SM" + std::to_string(config.spec.target_sm));
  KernelSpec const& spec = config.spec;
  if (spec.C <= 0 || spec.D <= 0 || spec.num_threads == 0)
    throw std::invalid_argument("trimul KF K1 CUBIN has invalid launch geometry");
  char const* const abi = config.cubin.launch_abi;
  if (config.cubin.kernel_sm == 90 && abi != nullptr && std::strcmp(abi, kSM90LaunchAbi) == 0)
  {
    if (!config.embedded_image->sm90.enabled)
      throw std::invalid_argument("trimul KF K1 CUBIN is missing its Hopper launch metadata");
    if (spec.kernel_variant < 0 || spec.kernel_variant > 2)
      throw std::invalid_argument("trimul KF K1 CUBIN has an invalid Hopper variant");
    if (spec.padded && config.embedded_image->sm90.x_chunk.rank == 0)
      throw std::invalid_argument("padded trimul KF K1 CUBIN is missing its x row-chunk TMA metadata");
    return;
  }
  if (config.cubin.kernel_sm == 100 && abi != nullptr && std::strcmp(abi, kSM100LaunchAbi) == 0)
  {
    embedded::SM100LaunchInfo const& sm100 = config.embedded_image->sm100;
    if (!sm100.enabled)
      throw std::invalid_argument("trimul KF K1 CUBIN is missing its Blackwell launch metadata");
    if (spec.kernel_variant < 0 || spec.kernel_variant > 1)
      throw std::invalid_argument("trimul KF K1 CUBIN has an invalid Blackwell variant");
    bool const tiled = spec.kernel_variant == kSM100TiledVariant;
    if (tiled != (spec.tile.ctas != 0) || (tiled && spec.tile.ctas > 2))
      throw std::invalid_argument("trimul KF K1 CUBIN has an invalid Blackwell tile");
    if (
      sm100.cluster_dims[0] != (tiled ? spec.tile.ctas : 1U) || sm100.cluster_dims[1] != 1
      || sm100.cluster_dims[2] != 1)
      throw std::invalid_argument("trimul KF K1 CUBIN records a cluster its tile does not launch");
    if ((tiled ? sm100.planes.rank : sm100.a.rank) == 0)
      throw std::invalid_argument("trimul KF K1 CUBIN is missing its a/b store TMA metadata");
    return;
  }
  throw std::invalid_argument("trimul KF K1 CUBIN has an incompatible launch ABI");
}

/* A flat row-major [rows, cols] operand as a rank-2 TMA source. */
TmaTensorSource matrix_source(std::uint64_t data, std::int64_t rows, std::int64_t cols)
{
  return TmaTensorSource{
    data,
    {static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(cols), 0, 0},
    {static_cast<std::uint64_t>(cols), 1, 0, 0},
  };
}

/* The first positions of every [P, P] plane of a flat channel-major [nb, D, P * P] buffer as the
 * kernel's (positions, D, nb) TMA source.
 */
TmaTensorSource channel_major_source(
  FlatTensorView const& view, std::int64_t positions, std::int64_t plane, std::int64_t D, std::int64_t nb)
{
  return TmaTensorSource{
    view.data,
    {static_cast<std::uint64_t>(positions), static_cast<std::uint64_t>(D), static_cast<std::uint64_t>(nb), 0},
    {1, static_cast<std::uint64_t>(plane), static_cast<std::uint64_t>(D * plane), 0},
  };
}

/* Flat x [rows, C] as the padded kernel's (n, C, rows / n) TMA source: one x row of n positions
 * per (b, i), so a row chunk reaching past n, or a chunk row past the batch, loads zeros.
 */
TmaTensorSource row_chunk_source(FlatTensorView const& view, std::int64_t n, std::int64_t C, std::int64_t x_rows)
{
  return TmaTensorSource{
    view.data,
    {static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(C), static_cast<std::uint64_t>(x_rows), 0},
    {static_cast<std::uint64_t>(C), 1, static_cast<std::uint64_t>(n * C), 0},
  };
}

void launch_k1(
  cubin_kernel_t loaded,
  KernelConfig const& config,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  std::int32_t const device = cuda_device_for_context(context);
  KernelSpec const& spec = config.spec;
  std::int64_t const C = spec.C;
  std::int64_t const D = spec.D;
  std::int64_t const n = params.n;
  std::int64_t const nb = params.nb;
  std::int64_t const rows = params.rows;
  if (n <= 0 || nb <= 0 || rows != checked_mul(nb, checked_mul(n, n, "trimul KF K1 rows"), "trimul KF K1 rows"))
    throw std::invalid_argument("trimul KF K1 needs rows == nb * n * n with positive n and nb");
  bool const pingpong = spec.kernel_variant != 0;
  std::int64_t const w_rows = (pingpong ? 4 : 2) * D;
  std::int64_t const pitch = ab_pitch(n, spec.padded);
  std::int64_t const ab_positions = n * pitch;
  /* Each batch's rows fill whole 128-row tiles, the last running past them, as do its row statistics. */
  std::uint64_t const batch_tiles = ceil_div(static_cast<std::uint64_t>(n * n), kTileRows);
  std::int64_t const tile_rows
    = checked_mul(nb, static_cast<std::int64_t>(batch_tiles * kTileRows), "trimul KF K1 rows");
  if (tile_rows > kMaxLaunchRows || ab_positions > kMaxLaunchRows)
  {
    throw std::invalid_argument(
      "trimul KF K1 addresses at most " + std::to_string(kMaxLaunchRows)
      + " tile rows and plane positions a launch; got " + std::to_string(tile_rows) + " rows, "
      + std::to_string(ab_positions) + " positions");
  }
  std::int64_t const plane = pitch * pitch;
  std::int64_t const ab_elements = checked_mul(nb, checked_mul(D, plane, "trimul KF K1 a/b"), "trimul KF K1 a/b");
  /* Tiles never straddle two batches: a dense image tiles each batch's rows, a padded one its planes in
   * pairs of row chunks.
   */
  std::uint64_t const num_tiles = spec.padded
    ? static_cast<std::uint64_t>(nb) * ceil_div(static_cast<std::uint64_t>(n * (pitch / kAbChunk)), 2)
    : static_cast<std::uint64_t>(nb) * batch_tiles;

  validate_flat(params.x, checked_mul(rows, C, "trimul KF K1 x"), 16, device, "x");
  validate_tensor(params.seqlen, "seqlen", 4);
  check_device(params.seqlen.device, device, "seqlen");
  if (params.seqlen.shape[0] < nb * n)
    throw std::invalid_argument("seqlen needs one length per (b, i) row");
  validate_flat(params.w_in, w_rows * C, 16, device, "w_in");
  if (!pingpong)
    validate_flat(params.w_gate_in, 2 * D * C, 16, device, "w_gate_in");
  validate_flat(params.vec_in, 8 * D, 16, device, "vec_in");
  validate_flat(params.a, ab_elements, 16, device, "a");
  validate_flat(params.b, ab_elements, 16, device, "b");
  if (spec.kernel_variant == kStatsVariant)
  {
    validate_flat_tensor(params.stats, "stats", 16);
    check_device(params.stats.device, device, "stats");
    /* Batch b's row p at b * (N * N rounded up to whole 128-row tiles) + p, in either layout. */
    if (params.stats.extent < 2 * tile_rows)
      throw std::invalid_argument("stats needs two values per row, each batch rounded up to whole 128-row tiles");
  }

  embedded::SM90LaunchInfo const& metadata = config.embedded_image->sm90;
  CUtensorMapDataType const expected_dtype = tma_data_type(true);
  abi::SM90Params device_params{};
  /* The stores cover the first n * P positions of each plane: the n rows, pad columns included (K2
   * never reads them); the pad rows stay unwritten.
   */
  TmaTensorSource const x_source = matrix_source(params.x.data, rows, C);
  TmaTensorSource const w_proj_source = matrix_source(params.w_in.data, w_rows, C);
  TmaTensorSource const a_source = channel_major_source(params.a, ab_positions, plane, D, nb);
  TmaTensorSource const b_source = channel_major_source(params.b, ab_positions, plane, D, nb);
  encode_tma_descriptor(device_params.x_tma, metadata.x, expected_dtype, x_source, "x");
  encode_tma_descriptor(device_params.w_proj_tma, metadata.w_proj, expected_dtype, w_proj_source, "w_in");
  encode_tma_descriptor(device_params.a_tma, metadata.a, expected_dtype, a_source, "a");
  encode_tma_descriptor(device_params.b_tma, metadata.b, expected_dtype, b_source, "b");
  finalize_sm90_tma_atom(device_params.x_tma, sm90_tma_operation_tag(x_source));
  finalize_sm90_tma_atom(device_params.w_proj_tma, sm90_tma_operation_tag(w_proj_source));
  finalize_sm90_tma_atom(device_params.a_tma, sm90_tma_operation_tag(a_source));
  finalize_sm90_tma_atom(device_params.b_tma, sm90_tma_operation_tag(b_source));
  if (!pingpong)
  {
    TmaTensorSource const w_gate_source = matrix_source(params.w_gate_in.data, 2 * D, C);
    encode_tma_descriptor(device_params.w_gate_tma, metadata.w_gate, expected_dtype, w_gate_source, "w_gate_in");
    finalize_sm90_tma_atom(device_params.w_gate_tma, sm90_tma_operation_tag(w_gate_source));
  }
  if (spec.padded)
  {
    TmaTensorSource const x_chunk_source = row_chunk_source(params.x, n, C, nb * n);
    encode_tma_descriptor(device_params.x_chunk_tma, metadata.x_chunk, expected_dtype, x_chunk_source, "x row chunks");
    finalize_sm90_tma_atom(device_params.x_chunk_tma, sm90_tma_operation_tag(x_chunk_source));
    device_params.x_chunk_coord = CoordTensorS2{{static_cast<std::int32_t>(n), static_cast<std::int32_t>(nb * n)}};
  }
  device_params.x_coord = CoordTensorS1{{static_cast<std::int32_t>(rows)}};
  device_params.a_coord = CoordTensorS2{{static_cast<std::int32_t>(ab_positions), static_cast<std::int32_t>(nb)}};
  device_params.b_coord = device_params.a_coord;
  device_params.seqlen = make_tensor1_descriptor(params.seqlen);
  device_params.vec.data = static_cast<CUdeviceptr>(params.vec_in.data);
  if (spec.kernel_variant == kStatsVariant)
  {
    device_params.stats.data = static_cast<CUdeviceptr>(params.stats.data);
    device_params.stats.dynamic_shapes[0] = static_cast<std::int32_t>(rows);
  }
  device_params.n = static_cast<std::int32_t>(n);
  device_params.num_tiles = static_cast<std::int32_t>(checked_u32(num_tiles, "trimul KF K1 tile count"));
  device_params.eps = params.eps;
  device_params.rows = static_cast<std::int32_t>(rows);

  void* kernel_params[abi::kSM90MaxParameterCount]{};
  std::size_t const parameter_count
    = abi::pack_sm90_kernel_params(&device_params, spec.kernel_variant, spec.padded, kernel_params);
  if (parameter_count != abi::sm90_parameter_count(spec.kernel_variant))
    throw std::logic_error("trimul KF K1 parameter packer produced the wrong ABI count");

  for (std::uint32_t dimension : metadata.cluster_dims)
  {
    if (dimension != 1)
      throw std::invalid_argument("trimul KF K1 CUBIN must launch unit clusters");
  }
  std::int32_t const multiprocessor_count = cuda_multiprocessor_count_for_context(context);
  if (multiprocessor_count <= 0)
    throw std::invalid_argument("current CUDA device has no active multiprocessors");
  /* Persistent CTAs stride over the row tiles, or, in K1_2, take runs of (tile, channel group) units, at
   * most one CTA a unit; K1 waits on its predecessor before reading anything.
   */
  std::uint64_t const work_units = spec.kernel_variant == kStatsVariant
    ? num_tiles * static_cast<std::uint64_t>(2 * D / std::min<std::int64_t>(D, 64))
    : num_tiles;
  cubin_launch_config_t launch_config{};
  launch_config.grid_x = checked_u32(
    std::min<std::uint64_t>(work_units, static_cast<std::uint64_t>(multiprocessor_count)), "trimul KF K1 grid.x");
  launch_config.grid_y = 1;
  launch_config.grid_z = 1;
  launch_config.block_x = metadata.block_dims[0];
  launch_config.block_y = metadata.block_dims[1];
  launch_config.block_z = metadata.block_dims[2];
  launch_config.cluster_x = metadata.cluster_dims[0];
  launch_config.cluster_y = metadata.cluster_dims[1];
  launch_config.cluster_z = metadata.cluster_dims[2];
  launch_config.cluster_scheduling_policy = metadata.cluster_scheduling_policy;
  launch_config.dynamic_smem_bytes = smem_bytes;
  launch_config.stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream));
  launch_config.programmatic_stream_serialization = allows_programmatic_launch(launch_config.stream) ? 1 : 0;
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(trimul_kf_k1_sm90)");
}

KernelTile image_tile(embedded::CubinImage const& image)
{
  return KernelTile{
    image.bnc,
    image.epi_n,
    image.ctas,
    image.w_slots,
    image.s_stages,
    image.store_warps,
    image.epi_groups,
  };
}

/* The SM100/SM103 K1: x as dense (n * n, C, nb) rows, or padded row chunks, (n, C, n, nb) for K1_B0 and
 * (n, C, nb * n) for K1_B1; the weights as [2D, C] matrices; and the planes as K1_B0's (n * P, D, a|b, nb)
 * view of one buffer, or K1_B1's separate (n * n, D, nb) or, padded, (n, D, n, nb) views. One launch
 * covers every batch.
 */
void launch_k1_sm100(
  cubin_kernel_t loaded,
  KernelConfig const& config,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  std::int32_t const device = cuda_device_for_context(context);
  KernelSpec const& spec = config.spec;
  bool const tiled = spec.kernel_variant == kSM100TiledVariant;
  std::int64_t const C = spec.C;
  std::int64_t const D = spec.D;
  std::int64_t const n = params.n;
  std::int64_t const nb = params.nb;
  if (n <= 0 || nb <= 0)
    throw std::invalid_argument("trimul KF K1 needs positive n and nb");
  std::int64_t const nn = checked_mul(n, n, "trimul KF K1 rows");
  std::int64_t const rows = checked_mul(nb, nn, "trimul KF K1 rows");
  std::int64_t const pitch = ab_pitch(n, spec.padded);
  std::int64_t const ab_positions = checked_mul(n, pitch, "trimul KF K1 a/b");
  std::int64_t const plane = checked_mul(pitch, pitch, "trimul KF K1 a/b");
  std::int64_t const ab_elements = checked_mul(nb, checked_mul(D, plane, "trimul KF K1 a/b"), "trimul KF K1 a/b");
  /* Tiles never straddle two batches: a dense image tiles each batch's n * n positions, a padded one its n
   * plane rows in pairs of 64-position chunks, which is the same ceil(n * P / 128).
   */
  static_assert(2 * kAbChunk == static_cast<std::int64_t>(kTileRows));
  std::uint64_t const tiles_per_batch = ceil_div(static_cast<std::uint64_t>(ab_positions), kTileRows);
  if (rows > kMaxLaunchRows || checked_mul(nb, ab_positions, "trimul KF K1 a/b") > kMaxLaunchRows)
  {
    throw std::invalid_argument(
      "trimul KF K1 addresses at most " + std::to_string(kMaxLaunchRows) + " rows and plane positions a launch");
  }

  validate_flat(params.x, checked_mul(rows, C, "trimul KF K1 x"), 16, device, "x");
  validate_tensor(params.seqlen, "seqlen", 4);
  check_device(params.seqlen.device, device, "seqlen");
  if (params.seqlen.shape[0] < nb * n)
    throw std::invalid_argument("seqlen needs one length per (b, i) row");
  validate_flat(params.w_in, 2 * D * C, 16, device, "w_in");
  validate_flat(params.w_gate_in, 2 * D * C, 16, device, "w_gate_in");
  validate_flat(params.vec_in, 8 * D, 16, device, "vec_in");
  if (tiled)
  {
    validate_flat(params.planes, 2 * ab_elements, 16, device, "planes");
  }
  else
  {
    validate_flat(params.a, ab_elements, 16, device, "a");
    validate_flat(params.b, ab_elements, 16, device, "b");
  }

  embedded::SM100LaunchInfo const& metadata = config.embedded_image->sm100;
  CUtensorMapDataType const expected_dtype = tma_data_type(true);
  abi::SM100Params device_params{};
  auto const encode
    = [&](CUtensorMap& descriptor, TmaDescriptorInfo const& info, TmaTensorSource const& source, char const* name)
  {
    encode_tma_descriptor(descriptor, info, expected_dtype, source, name);
    finalize_sm100_tma_atom(descriptor);
  };
  auto const u64 = [](std::int64_t value) { return static_cast<std::uint64_t>(value); };
  auto const i32 = [](std::int64_t value) { return static_cast<std::int32_t>(value); };

  TmaTensorSource x_source{};
  if (!spec.padded)
    x_source = TmaTensorSource{params.x.data, {u64(nn), u64(C), u64(nb), 0}, {u64(C), 1, u64(nn * C), 0}};
  else if (tiled)
    x_source = TmaTensorSource{params.x.data, {u64(n), u64(C), u64(n), u64(nb)}, {u64(C), 1, u64(n * C), u64(nn * C)}};
  else
    x_source = TmaTensorSource{params.x.data, {u64(n), u64(C), u64(nb * n), 0}, {u64(C), 1, u64(n * C), 0}};
  encode(device_params.x_tma, metadata.x, x_source, "x");
  encode(device_params.w_proj_tma, metadata.w_proj, matrix_source(params.w_in.data, 2 * D, C), "w_in");
  encode(device_params.w_gate_tma, metadata.w_gate, matrix_source(params.w_gate_in.data, 2 * D, C), "w_gate_in");
  if (spec.padded && tiled)
    device_params.x_coord3 = CoordTensorS3{{i32(n), i32(n), i32(nb)}};
  else if (spec.padded)
    device_params.x_coord2 = CoordTensorS2{{i32(n), i32(nb * n)}};
  else
    device_params.x_coord2 = CoordTensorS2{{i32(nn), i32(nb)}};

  if (tiled)
  {
    TmaTensorSource const planes_source{
      params.planes.data,
      {u64(ab_positions), u64(D), 2, u64(nb)},
      {1, u64(plane), u64(nb * D * plane), u64(D * plane)},
    };
    encode(device_params.planes_tma, metadata.planes, planes_source, "planes");
    device_params.planes_coord = CoordTensorS2{{i32(ab_positions), i32(nb)}};
  }
  else
  {
    auto const ab_source = [&](FlatTensorView const& view)
    {
      return spec.padded
        ? TmaTensorSource{view.data, {u64(n), u64(D), u64(n), u64(nb)}, {1, u64(plane), u64(pitch), u64(D * plane)}}
        : TmaTensorSource{view.data, {u64(nn), u64(D), u64(nb), 0}, {1, u64(nn), u64(D * nn), 0}};
    };
    encode(device_params.a_tma, metadata.a, ab_source(params.a), "a");
    encode(device_params.b_tma, metadata.b, ab_source(params.b), "b");
    if (spec.padded)
      device_params.ab_coord3 = CoordTensorS3{{i32(n), i32(n), i32(nb)}};
    else
      device_params.ab_coord2 = CoordTensorS2{{i32(nn), i32(nb)}};
  }
  device_params.seqlen = make_tensor1_descriptor(params.seqlen);
  device_params.vec.data = static_cast<CUdeviceptr>(params.vec_in.data);
  device_params.n = i32(n);
  device_params.P = i32(pitch);
  device_params.nb = i32(nb);
  device_params.eps = params.eps;

  std::int32_t const multiprocessor_count = cuda_multiprocessor_count_for_context(context);
  if (multiprocessor_count <= 0)
    throw std::invalid_argument("current CUDA device has no active multiprocessors");
  std::uint64_t const ctas = tiled ? spec.tile.ctas : 1U;
  std::uint64_t grid_x = 0;
  if (tiled)
  {
    /* A work tile is one CTA tile, or a pair of a batch's consecutive tiles for a CTA pair, of 2D / bnc
     * units. Each persistent cluster owns a contiguous share of the units; with fewer tiles than
     * clusters, the shares are whole tiles or the most equal parts of each tile that fit. This matches
     * the source kernel's host launch.
     */
    std::uint64_t const work_per_batch = ceil_div(tiles_per_batch, ctas);
    std::uint64_t const num_tiles = work_per_batch * static_cast<std::uint64_t>(nb);
    std::uint64_t const units_per_tile = 2 * static_cast<std::uint64_t>(spec.D) / spec.tile.bnc;
    std::uint64_t const num_units = num_tiles * units_per_tile;
    std::uint64_t const max_workers = static_cast<std::uint64_t>(multiprocessor_count) / ctas;
    std::uint64_t workers = std::min(num_units, max_workers);
    if (num_tiles < max_workers)
    {
      workers = num_tiles;
      for (std::uint64_t parts = 2; parts <= units_per_tile; ++parts)
        if (units_per_tile % parts == 0 && parts * num_tiles <= max_workers)
          workers = parts * num_tiles;
    }
    device_params.per_batch = checked_i32(work_per_batch, "trimul KF K1 work per batch");
    device_params.total = checked_i32(num_units, "trimul KF K1 units");
    grid_x = workers * ctas;
  }
  else
  {
    std::uint64_t const num_tiles = tiles_per_batch * static_cast<std::uint64_t>(nb);
    device_params.per_batch = checked_i32(tiles_per_batch, "trimul KF K1 tiles per batch");
    device_params.total = checked_i32(num_tiles, "trimul KF K1 tiles");
    grid_x = std::min<std::uint64_t>(num_tiles, static_cast<std::uint64_t>(multiprocessor_count));
  }

  void* kernel_params[abi::kSM100MaxParameterCount]{};
  std::size_t const parameter_count
    = abi::pack_sm100_kernel_params(&device_params, spec.kernel_variant, spec.padded, kernel_params);
  if (parameter_count != abi::sm100_parameter_count(spec.kernel_variant))
    throw std::logic_error("trimul KF K1 SM100 parameter packer produced the wrong ABI count");

  cubin_launch_config_t launch_config{};
  launch_config.grid_x = checked_u32(grid_x, "trimul KF K1 grid.x");
  launch_config.grid_y = 1;
  launch_config.grid_z = 1;
  launch_config.block_x = metadata.block_dims[0];
  launch_config.block_y = metadata.block_dims[1];
  launch_config.block_z = metadata.block_dims[2];
  launch_config.cluster_x = metadata.cluster_dims[0];
  launch_config.cluster_y = metadata.cluster_dims[1];
  launch_config.cluster_z = metadata.cluster_dims[2];
  launch_config.cluster_scheduling_policy = metadata.cluster_scheduling_policy;
  launch_config.dynamic_smem_bytes = smem_bytes;
  launch_config.stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream));
  /* The source kernels request no programmatic dependent launch: under CUDA-graph replay a kernel could read
   * its predecessor's output mid-write.
   */
  launch_config.programmatic_stream_serialization = 0;
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(trimul_kf_k1_sm100)");
}

KernelSpec make_kernel_spec(embedded::CubinImage const& image)
{
  return KernelSpec{
    image.cubin.target_sm,
    image.cubin.kernel_sm,
    image.C,
    image.D,
    image.kernel_variant,
    image.padded,
    image.num_threads,
    image_tile(image),
  };
}

embedded::CubinImage const& find_embedded_cubin(
  std::int32_t target_sm,
  DType dtype,
  std::int32_t C,
  std::int32_t D,
  std::int32_t kernel_variant,
  bool padded,
  KernelTile const& tile)
{
  bool const is_bfloat16 = dtype == DType::kBFloat16;
  embedded::RegistryView const registry = embedded::registry();
  for (std::size_t index = 0; index < registry.count; ++index)
  {
    embedded::CubinImage const& image = registry.images[index];
    if (
      cubin_supports_sm(image.cubin, target_sm) && image.is_bfloat16 == is_bfloat16 && image.C == C && image.D == D
      && image.kernel_variant == kernel_variant && image.padded == padded
      && image_tile(image) == tile
      // an image with a split predates the unit shares and reads total as work items, not units
      && image.split == 0)
      return image;
  }
  throw std::invalid_argument(
    "No embedded trimul KF K1 CUBIN for SM" + std::to_string(target_sm) + ", C=" + std::to_string(C)
    + ", D=" + std::to_string(D) + ", variant " + std::to_string(kernel_variant) + (padded ? ", padded a/b" : "")
    + (tile.ctas != 0 ? ", bnc=" + std::to_string(tile.bnc) + ", ctas=" + std::to_string(tile.ctas) : ""));
}

std::size_t preload_kernels(CUcontext context, std::int32_t device_sm)
{
  return preload_registry_kernels(context, device_sm, embedded::registry());
}

CubinPreloadRegistration const kPreloader{
  "trimul_kf_k1",
  &preload_kernels,
};

cubin_kernel_t load_for_current_device(KernelConfig const& config, CUcontext context)
{
  std::int32_t const device_sm = cuda_sm_for_context(context);
  if (device_sm != config.spec.target_sm || !cubin_supports_sm(config.cubin, device_sm))
  {
    throw std::invalid_argument(
      "trimul KF K1 config selects SM" + std::to_string(config.spec.target_sm) + " but current device is SM"
      + std::to_string(device_sm));
  }
  return load_embedded_kernel(context, config.cubin);
}

} // namespace

std::vector<KernelSpec> kernel_specs()
{
  return map_registry(embedded::registry(), make_kernel_spec);
}

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  DType dtype,
  std::int32_t C,
  std::int32_t D,
  std::int32_t kernel_variant,
  bool padded,
  KernelTile const& tile)
{
  embedded::CubinImage const& image = find_embedded_cubin(target_sm, dtype, C, D, kernel_variant, padded, tile);
  KernelConfig config{make_kernel_spec(image), dtype, image.cubin, &image};
  /* Reject an image of an older launch ABI here rather than at launch, so callers fall back. */
  validate_image(config);
  return config;
}

void launch(KernelConfig const& config, LaunchParams const& params)
{
  validate_image(config);
  CUcontext const context = current_cuda_context();
  cubin_kernel_t const loaded = load_for_current_device(config, context);
  if (config.cubin.kernel_sm == 100)
    launch_k1_sm100(loaded, config, params, context, dynamic_smem_bytes(config));
  else
    launch_k1(loaded, config, params, context, dynamic_smem_bytes(config));
}

} // namespace bioir::cutedsl::trimul_kf_k1
