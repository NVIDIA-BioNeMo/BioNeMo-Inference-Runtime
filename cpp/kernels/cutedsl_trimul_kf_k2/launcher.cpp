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

#include "trimul_kf_k2_registry.h"

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace bioir::cutedsl::trimul_kf_k2
{
namespace
{

constexpr char kSM90LaunchAbi[] = "trimul_kf_k2_sm90_v1";
constexpr char kSM100LaunchAbi[] = "trimul_kf_k2_sm100_v1";
/* Rows of a CTA pair's tile, and the column block K2_B2's tails complete. */
constexpr std::uint64_t kPairTile = 256;
/* Contraction k-block of every variant. */
constexpr std::uint64_t kTileK = 64;

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

/* A strided operand view: its flat extent must reach the last element the TMA box walk can address. */
void validate_span(FlatTensorView const& view, std::int64_t span, std::int32_t device, char const* name)
{
  validate_flat_tensor(view, name, 16);
  check_device(view.device, device, name);
  if (view.extent < span)
  {
    throw std::invalid_argument(
      std::string(name) + " must span at least " + std::to_string(span) + " elements, got "
      + std::to_string(view.extent));
  }
}

/* Row pitch and plane stride of a and b, or of prod: both whole 16-byte TMA strides. */
void validate_strides(std::int64_t n, std::int64_t pitch, std::int64_t plane, char const* name)
{
  constexpr std::int64_t kStrideElements = 8;
  if (
    pitch < n || pitch % kStrideElements != 0 || plane % kStrideElements != 0
    || plane < checked_mul(n, pitch, "trimul KF K2 plane extent n * pitch"))
  {
    throw std::invalid_argument(
      std::string("trimul KF K2 needs ") + name + " row pitch >= n and plane stride >= n * pitch, both multiples of "
      + std::to_string(kStrideElements) + " elements; got n=" + std::to_string(n) + ", pitch=" + std::to_string(pitch)
      + ", plane=" + std::to_string(plane));
  }
}

/* Elements from the first to one past the last addressed element of l matrices of n rows by cols columns. */
std::int64_t matrices_span(
  std::int64_t n, std::int64_t cols, std::int64_t l, std::int64_t pitch, std::int64_t plane, char const* name)
{
  return checked_add(checked_add(checked_mul(l - 1, plane, name), checked_mul(n - 1, pitch, name), name), cols, name);
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
    throw std::invalid_argument("trimul KF K2 CUBIN is missing its registry record");
  if (!cubin_supports_sm(config.cubin, config.spec.target_sm))
    throw std::invalid_argument(
      "embedded CUBIN does not support configured device SM" + std::to_string(config.spec.target_sm));
  KernelSpec const& spec = config.spec;
  char const* const abi = config.cubin.launch_abi;
  if (config.cubin.kernel_sm == 100 && abi != nullptr && std::strcmp(abi, kSM100LaunchAbi) == 0)
  {
    embedded::SM100LaunchInfo const& sm100 = config.embedded_image->sm100;
    if (!sm100.enabled)
      throw std::invalid_argument("trimul KF K2 CUBIN is missing its Blackwell launch metadata");
    if (
      spec.kernel_variant < 0 || spec.kernel_variant > 2 || spec.tile_n == 0 || spec.num_threads == 0
      || (spec.cluster_m != 1 && spec.cluster_m != 2) || (spec.cluster_n != 1 && spec.cluster_n != 2)
      || spec.tile_m != 128 * spec.cluster_m || (spec.kernel_variant != 0 && spec.cluster_m != 2)
      || (spec.kernel_variant == 0 && spec.cluster_m == 2 && spec.cluster_n != 1)
      || (spec.kernel_variant == 2) != (spec.tail_w != 0) || (spec.clc && spec.kernel_variant != 1))
    {
      throw std::invalid_argument("trimul KF K2 CUBIN has an invalid Blackwell tile");
    }
    if (
      sm100.cluster_dims[0] != spec.cluster_m || sm100.cluster_dims[1] != spec.cluster_n || sm100.cluster_dims[2] != 1)
      throw std::invalid_argument("trimul KF K2 CUBIN records a cluster its tile does not launch");
    if (spec.kernel_variant == 2 && (sm100.b_tail.rank == 0 || sm100.prod_t.rank == 0))
      throw std::invalid_argument("trimul KF K2_B2 CUBIN is missing its tail TMA metadata");
    return;
  }
  if (config.cubin.kernel_sm != 90 || abi == nullptr || std::strcmp(abi, kSM90LaunchAbi) != 0)
    throw std::invalid_argument("trimul KF K2 CUBIN has an incompatible launch ABI");
  if (!config.embedded_image->sm90.enabled)
    throw std::invalid_argument("trimul KF K2 CUBIN is missing its Hopper launch metadata");
  if (
    spec.kernel_variant < 0 || spec.kernel_variant > 2 || spec.tile_m == 0 || spec.tile_n == 0 || spec.cluster_m == 0
    || spec.num_threads == 0)
  {
    throw std::invalid_argument("trimul KF K2 CUBIN has invalid launch geometry");
  }
  if (spec.dynamic && spec.kernel_variant == 1)
    throw std::invalid_argument("trimul KF K2_1 has no dynamic schedule");
}

/* Clusters of the loaded image that fit on the device at once, which sizes a dynamic image's grid:
 * fewer than num_sms / cluster size wherever clusters cannot tile every GPC. Cached per function.
 */
std::uint64_t max_active_clusters(cubin_kernel_t loaded, cubin_launch_config_t const& launch_config)
{
  static std::mutex mutex;
  static std::map<std::pair<CUfunction, std::uint32_t>, std::uint64_t> cache;
  std::pair<CUfunction, std::uint32_t> const key{loaded.function, launch_config.dynamic_smem_bytes};
  std::lock_guard<std::mutex> lock(mutex);
  auto const found = cache.find(key);
  if (found != cache.end())
    return found->second;
  CUlaunchAttribute attribute{};
  attribute.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attribute.value.clusterDim.x = launch_config.cluster_x;
  attribute.value.clusterDim.y = launch_config.cluster_y;
  attribute.value.clusterDim.z = launch_config.cluster_z;
  CUlaunchConfig query{};
  query.gridDimX = launch_config.cluster_x;
  query.gridDimY = launch_config.cluster_y;
  query.gridDimZ = launch_config.cluster_z;
  query.blockDimX = launch_config.block_x;
  query.blockDimY = launch_config.block_y;
  query.blockDimZ = launch_config.block_z;
  query.sharedMemBytes = launch_config.dynamic_smem_bytes;
  query.attrs = &attribute;
  query.numAttrs = 1;
  int clusters = 0;
  check_cuda_driver(
    cuOccupancyMaxActiveClusters(&clusters, loaded.function, &query), "cuOccupancyMaxActiveClusters(trimul_kf_k2)");
  if (clusters <= 0)
    throw std::invalid_argument("current CUDA device cannot hold one trimul KF K2 cluster");
  cache.emplace(key, static_cast<std::uint64_t>(clusters));
  return static_cast<std::uint64_t>(clusters);
}

/* l matrices of n x n, rows pitch and matrices plane elements apart, as the kernel's (n, n, l) TMA
 * source: mode 0 contiguous (rows_contiguous) or mode 1 contiguous within each matrix.
 */
TmaTensorSource matrices_source(
  FlatTensorView const& view,
  std::int64_t n,
  std::int64_t l,
  std::int64_t pitch,
  std::int64_t plane,
  bool rows_contiguous)
{
  std::uint64_t const extent = static_cast<std::uint64_t>(n);
  std::uint64_t const row_stride = static_cast<std::uint64_t>(pitch);
  std::uint64_t const plane_stride = static_cast<std::uint64_t>(plane);
  return TmaTensorSource{
    view.data,
    {extent, extent, static_cast<std::uint64_t>(l), 0},
    rows_contiguous ? std::array<std::uint64_t, 4>{1, row_stride, plane_stride, 0}
                    : std::array<std::uint64_t, 4>{row_stride, 1, plane_stride, 0},
  };
}

/* The product as the kernel's (n, pitch, l) TMA destination: its stores clip at the row pitch rather than at n, so
 * the column tiles that reach past n write the pad columns too (zeros) and every row ends on a whole line.
 */
TmaTensorSource
product_source(FlatTensorView const& view, std::int64_t n, std::int64_t l, std::int64_t pitch, std::int64_t plane)
{
  return TmaTensorSource{
    view.data,
    {static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(pitch), static_cast<std::uint64_t>(l), 0},
    {static_cast<std::uint64_t>(pitch), 1, static_cast<std::uint64_t>(plane), 0},
  };
}

void launch_sm90(
  cubin_kernel_t loaded,
  KernelConfig const& config,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  std::int32_t const device = cuda_device_for_context(context);
  KernelSpec const& spec = config.spec;
  std::int64_t const n = params.n;
  std::int64_t const l = params.l;
  std::int64_t const pitch = params.ab_pitch;
  std::int64_t const plane = params.ab_plane;
  std::int64_t const prod_pitch = params.prod_pitch;
  std::int64_t const prod_plane = params.prod_plane;
  if (n <= 0 || l <= 0)
    throw std::invalid_argument("trimul KF K2 needs positive n and l");
  validate_strides(n, pitch, plane, "a/b");
  validate_strides(n, prod_pitch, prod_plane, "prod");
  std::int64_t const ab_span = matrices_span(n, n, l, pitch, plane, "trimul KF K2 a/b span");
  validate_span(params.a, ab_span, device, "a");
  validate_span(params.b, ab_span, device, "b");
  validate_span(
    params.prod, matrices_span(n, prod_pitch, l, prod_pitch, prod_plane, "trimul KF K2 prod span"), device, "prod");
  if (spec.dynamic)
    validate_span(params.counter, 1, device, "counter");

  embedded::SM90LaunchInfo const& metadata = config.embedded_image->sm90;
  std::uint32_t const cluster_m = metadata.cluster_dims[0];
  std::uint32_t const cluster_n = metadata.cluster_dims[1];
  /* K2_0 and K2_2 launch their cluster along x, K2_1 along y. K2_2's x-cluster is a pair of
   * horizontally adjacent tiles.
   */
  if (
    metadata.cluster_dims[2] != 1 || cluster_m != spec.cluster_m
    || (spec.kernel_variant == 1 ? cluster_m != 1 : cluster_n != 1) || (spec.kernel_variant == 2 && cluster_m != 2))
    throw std::invalid_argument("trimul KF K2 CUBIN has inconsistent cluster metadata");
  std::uint64_t const k_blocks = ceil_div(static_cast<std::uint64_t>(n), kTileK);
  if (spec.defer_kmin > k_blocks)
  {
    throw std::invalid_argument(
      "trimul KF K2 tile defers " + std::to_string(spec.defer_kmin) + " k-blocks but n=" + std::to_string(n) + " has "
      + std::to_string(k_blocks));
  }
  std::uint64_t const m_tiles = ceil_div(static_cast<std::uint64_t>(n), spec.tile_m);
  std::uint64_t const n_tiles = ceil_div(static_cast<std::uint64_t>(n), spec.tile_n);
  if (n_tiles % cluster_n != 0)
    throw std::invalid_argument("trimul KF K2 clusters along N need a whole number of column-tile groups");

  /* Outgoing reads a and b row-major (K contiguous), incoming column-major. The product is always
   * row-major, at its own pitch.
   */
  CUtensorMapDataType const expected_dtype = tma_data_type(true);
  abi::SM90Params device_params{};
  TmaTensorSource const a_source = matrices_source(params.a, n, l, pitch, plane, !spec.outgoing);
  TmaTensorSource const b_source = matrices_source(params.b, n, l, pitch, plane, !spec.outgoing);
  TmaTensorSource const prod_source = product_source(params.prod, n, l, prod_pitch, prod_plane);
  encode_tma_descriptor(device_params.a_tma, metadata.a, expected_dtype, a_source, "a");
  encode_tma_descriptor(device_params.b_tma, metadata.b, expected_dtype, b_source, "b");
  encode_tma_descriptor(device_params.prod_tma, metadata.prod, expected_dtype, prod_source, "prod");
  finalize_sm90_tma_atom(device_params.a_tma, sm90_tma_operation_tag(a_source));
  finalize_sm90_tma_atom(device_params.b_tma, sm90_tma_operation_tag(b_source));
  finalize_sm90_tma_atom(device_params.prod_tma, sm90_tma_operation_tag(prod_source));
  CoordTensorS3 const coord{{static_cast<std::int32_t>(n), static_cast<std::int32_t>(n), static_cast<std::int32_t>(l)}};
  device_params.a_coord = coord;
  device_params.b_coord = coord;
  device_params.prod_coord = CoordTensorS3{
    {static_cast<std::int32_t>(n),
     static_cast<std::int32_t>(checked_u32(static_cast<std::uint64_t>(prod_pitch), "trimul KF K2 product pitch")),
     static_cast<std::int32_t>(l)}};

  std::int32_t const multiprocessor_count = cuda_multiprocessor_count_for_context(context);
  if (multiprocessor_count <= 0)
    throw std::invalid_argument("current CUDA device has no active multiprocessors");
  std::uint64_t const sms = static_cast<std::uint64_t>(multiprocessor_count);

  cubin_launch_config_t launch_config{};
  launch_config.block_x = metadata.block_dims[0];
  launch_config.block_y = metadata.block_dims[1];
  launch_config.block_z = metadata.block_dims[2];
  launch_config.cluster_x = metadata.cluster_dims[0];
  launch_config.cluster_y = metadata.cluster_dims[1];
  launch_config.cluster_z = metadata.cluster_dims[2];
  launch_config.cluster_scheduling_policy = metadata.cluster_scheduling_policy;
  launch_config.dynamic_smem_bytes = smem_bytes;
  if (spec.kernel_variant != 1)
  {
    /* Persistent clusters of cluster_m tiles, vertically adjacent for K2_0 and horizontally for
     * K2_2, walked (m, n, l) with m fastest. A static image's grid is the smallest one with the same
     * number of passes as the hardware-wide grid. A dynamic image starts every cluster that fits on
     * the device at once, each on the unit of its index, and they claim the rest.
     */
    bool const pairs_columns = spec.kernel_variant == 2;
    std::uint64_t const m_clusters = ceil_div(m_tiles, pairs_columns ? 1 : cluster_m);
    std::uint64_t const n_clusters = ceil_div(n_tiles, pairs_columns ? cluster_m : 1);
    std::uint64_t const work_units = m_clusters * n_clusters * static_cast<std::uint64_t>(l);
    device_params.schedule[0] = static_cast<std::int32_t>(checked_u32(m_clusters, "trimul KF K2 row clusters"));
    device_params.schedule[1] = static_cast<std::int32_t>(checked_u32(n_clusters, "trimul KF K2 column clusters"));
    device_params.schedule[2] = static_cast<std::int32_t>(checked_u32(work_units, "trimul KF K2 work units"));
    std::uint64_t clusters = 0;
    if (spec.dynamic)
    {
      clusters = std::min(work_units, max_active_clusters(loaded, launch_config));
      device_params.counter = params.counter.data;
    }
    else
    {
      std::uint64_t const max_active = sms / cluster_m;
      if (max_active == 0)
        throw std::invalid_argument("current CUDA device cannot hold one trimul KF K2 cluster");
      std::uint64_t const passes = ceil_div(work_units, max_active);
      clusters = ceil_div(work_units, passes);
    }
    launch_config.grid_x = cluster_m;
    launch_config.grid_y = 1;
    launch_config.grid_z = checked_u32(clusters, "trimul KF K2 grid.z");
  }
  else
  {
    /* Persistent CTAs walk a linear tile list; a clustered image rounds the grid down to whole
     * clusters laid out along y.
     */
    std::uint64_t const num_tiles = m_tiles * n_tiles * static_cast<std::uint64_t>(l);
    std::uint64_t const grid = std::min(num_tiles, sms) / cluster_n * cluster_n;
    device_params.schedule[0] = static_cast<std::int32_t>(checked_u32(m_tiles, "trimul KF K2 row tiles"));
    device_params.schedule[1] = static_cast<std::int32_t>(checked_u32(num_tiles, "trimul KF K2 tile count"));
    launch_config.grid_x = checked_u32(grid / cluster_n, "trimul KF K2 grid.x");
    launch_config.grid_y = cluster_n;
    launch_config.grid_z = 1;
  }

  void* kernel_params[abi::kSM90MaxParameterCount]{};
  std::size_t const parameter_count
    = abi::pack_sm90_kernel_params(&device_params, spec.kernel_variant, spec.dynamic, kernel_params);
  if (parameter_count != abi::sm90_parameter_count(spec.kernel_variant, spec.dynamic))
    throw std::logic_error("trimul KF K2 parameter packer produced the wrong ABI count");

  launch_config.stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream));
  launch_config.programmatic_stream_serialization = allows_programmatic_launch(launch_config.stream) ? 1 : 0;
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(trimul_kf_k2_sm90)");
}

abi::SM100FastDivmod make_sm100_fast_divmod(std::uint64_t divisor_u64)
{
  if (divisor_u64 == 0 || divisor_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::invalid_argument("trimul KF K2 tile-count divisor must be a positive int32");
  std::uint32_t const value = static_cast<std::uint32_t>(divisor_u64);
  std::uint32_t log2 = 0;
  std::uint64_t power_of_two = 1;
  while (power_of_two < value)
  {
    power_of_two <<= 1;
    ++log2;
  }
  std::uint64_t const multiplier = ((power_of_two - value) << 32) / value + 1;
  if (multiplier > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("trimul KF K2 fast-divmod multiplier overflow");
  std::uint8_t const shift_right_1 = static_cast<std::uint8_t>(std::min(log2, 1U));
  return abi::SM100FastDivmod{
    static_cast<std::int32_t>(value),
    static_cast<std::uint32_t>(multiplier),
    shift_right_1,
    static_cast<std::uint8_t>(log2 - shift_right_1),
  };
}

std::int32_t checked_i32(std::uint64_t value, char const* name)
{
  if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::invalid_argument(std::string(name) + " does not fit int32");
  return static_cast<std::int32_t>(value);
}

/* The SM100/SM103 K2 on CTA tiles (K2_B0) or CTA-pair tiles (K2_B1, K2_B2), writing a dense product. Each
 * variant's schedule repeats its source __call__: K2_B0 and K2_B2 launch the smallest persistent grid with the
 * trip count of the largest co-resident one; K2_B1 does too, or under cluster launch control one cluster per
 * work unit.
 */
void launch_sm100(
  cubin_kernel_t loaded,
  KernelConfig const& config,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  std::int32_t const device = cuda_device_for_context(context);
  KernelSpec const& spec = config.spec;
  std::int64_t const n = params.n;
  std::int64_t const l = params.l;
  std::int64_t const pitch = params.ab_pitch;
  std::int64_t const plane = params.ab_plane;
  if (n <= 0 || l <= 0)
    throw std::invalid_argument("trimul KF K2 needs positive n and l");
  if (n % 8 != 0)
    throw std::invalid_argument("SM100 trimul KF K2 needs n a multiple of 8");
  validate_strides(n, pitch, plane, "a/b");
  std::int64_t const nn = checked_mul(n, n, "trimul KF K2 product plane");
  if (params.prod_pitch != 0 && params.prod_pitch != n)
    throw std::invalid_argument("SM100 trimul KF K2 writes a dense product: its row pitch must be n");
  if (params.prod_plane != 0 && params.prod_plane != nn)
    throw std::invalid_argument("SM100 trimul KF K2 writes a dense product: its plane stride must be n * n");
  std::int64_t const ab_span = matrices_span(n, n, l, pitch, plane, "trimul KF K2 a/b span");
  validate_span(params.a, ab_span, device, "a");
  validate_span(params.b, ab_span, device, "b");
  validate_span(params.prod, checked_mul(l, nn, "trimul KF K2 prod span"), device, "prod");

  embedded::SM100LaunchInfo const& metadata = config.embedded_image->sm100;
  CUtensorMapDataType const expected_dtype = tma_data_type(true);
  abi::SM100Params device_params{};
  auto const encode
    = [&](CUtensorMap& descriptor, TmaDescriptorInfo const& info, TmaTensorSource const& source, char const* name)
  {
    encode_tma_descriptor(descriptor, info, expected_dtype, source, name);
    finalize_sm100_tma_atom(descriptor);
  };
  /* Outgoing reads a and b K-major (mode 1 contiguous), incoming M-major (mode 0); the product is row-major, its
   * transpose column-major.
   */
  TmaTensorSource const a_source = matrices_source(params.a, n, l, pitch, plane, !spec.outgoing);
  TmaTensorSource const b_source = matrices_source(params.b, n, l, pitch, plane, !spec.outgoing);
  encode(device_params.a_tma, metadata.a, a_source, "a");
  encode(device_params.b_tma, metadata.b, b_source, "b");
  encode(device_params.prod_tma, metadata.prod, matrices_source(params.prod, n, l, n, nn, false), "prod");
  std::int32_t const n32 = static_cast<std::int32_t>(n);
  device_params.coord = CoordTensorS3{{n32, n32, static_cast<std::int32_t>(l)}};
  device_params.n = n32;
  device_params.l = static_cast<std::int32_t>(l);

  std::int32_t const multiprocessor_count = cuda_multiprocessor_count_for_context(context);
  if (multiprocessor_count <= 0)
    throw std::invalid_argument("current CUDA device has no active multiprocessors");
  std::uint64_t const sms = static_cast<std::uint64_t>(multiprocessor_count);
  std::uint64_t const un = static_cast<std::uint64_t>(n);
  std::uint64_t const ul = static_cast<std::uint64_t>(l);
  std::uint64_t const cluster_size = static_cast<std::uint64_t>(spec.cluster_m) * spec.cluster_n;
  auto const static_clusters = [&](std::uint64_t work_units)
  {
    std::uint64_t const max_active = sms / cluster_size;
    if (max_active == 0)
      throw std::invalid_argument("current CUDA device cannot hold one trimul KF K2 cluster");
    return ceil_div(work_units, ceil_div(work_units, max_active));
  };
  std::uint64_t clusters = 0;
  if (spec.kernel_variant == 0)
  {
    std::uint64_t const tiles_m = ceil_div(un, spec.tile_m);
    std::uint64_t const tiles_n = ceil_div(un, static_cast<std::uint64_t>(spec.tile_n) * spec.cluster_n);
    std::uint64_t const work_units = tiles_m * tiles_n * ul;
    device_params.tiles_m = make_sm100_fast_divmod(tiles_m);
    device_params.tiles_n = make_sm100_fast_divmod(tiles_n);
    device_params.work_units = checked_i32(work_units, "trimul KF K2 work units");
    clusters = static_clusters(work_units);
  }
  else if (spec.kernel_variant == 1)
  {
    std::uint64_t const m_clusters = ceil_div(un, kPairTile);
    std::uint64_t const n_clusters = ceil_div(ceil_div(un, spec.tile_n), spec.cluster_n);
    std::uint64_t const work_units = m_clusters * n_clusters * ul;
    device_params.m_clusters = checked_i32(m_clusters, "trimul KF K2 row clusters");
    device_params.n_clusters = checked_i32(n_clusters, "trimul KF K2 column clusters");
    device_params.work_units = checked_i32(work_units, "trimul KF K2 work units");
    clusters = spec.clc ? work_units : static_clusters(work_units);
  }
  else
  {
    /* n = 256 km + rem: narrow tails when km >= 1 and 0 < rem <= tail_w, else the plain 256 x 256 tiling. */
    std::uint64_t const km = un / kPairTile;
    std::uint64_t const rem = un - km * kPairTile;
    std::uint64_t const use = km >= 1 && rem > 0 && rem <= spec.tail_w ? 1 : 0;
    std::uint64_t const jm = ceil_div(un, kPairTile);
    std::uint64_t const main_m = use != 0 ? km : jm;
    std::uint64_t const work_units = (main_m * main_m + use * (km + jm)) * ul;
    encode(device_params.b_tail_tma, metadata.b_tail, b_source, "b tail");
    encode(device_params.a_tail_tma, metadata.a_tail, a_source, "a tail");
    encode(device_params.b_as_a_tma, metadata.b_as_a, b_source, "b as A");
    encode(device_params.prod_t_tma, metadata.prod_t, matrices_source(params.prod, n, l, n, nn, true), "prod^T");
    device_params.main_m = checked_i32(main_m, "trimul KF K2 main tiles");
    device_params.work_units = checked_i32(work_units, "trimul KF K2 work units");
    device_params.use = static_cast<std::int32_t>(use);
    device_params.km = checked_i32(km, "trimul KF K2 row blocks");
    device_params.jm = checked_i32(jm, "trimul KF K2 column blocks");
    clusters = static_clusters(work_units);
  }

  void* kernel_params[abi::kSM100MaxParameterCount]{};
  std::size_t const parameter_count = abi::pack_sm100_kernel_params(&device_params, spec.kernel_variant, kernel_params);
  if (parameter_count != abi::sm100_parameter_count(spec.kernel_variant))
    throw std::logic_error("trimul KF K2 SM100 parameter packer produced the wrong ABI count");

  cubin_launch_config_t launch_config{};
  launch_config.block_x = metadata.block_dims[0];
  launch_config.block_y = metadata.block_dims[1];
  launch_config.block_z = metadata.block_dims[2];
  launch_config.cluster_x = metadata.cluster_dims[0];
  launch_config.cluster_y = metadata.cluster_dims[1];
  launch_config.cluster_z = metadata.cluster_dims[2];
  launch_config.cluster_scheduling_policy = metadata.cluster_scheduling_policy;
  launch_config.dynamic_smem_bytes = smem_bytes;
  /* K2_B0 and K2_B2 stack their clusters along z; K2_B1 spreads its CTA pairs along x. */
  if (spec.kernel_variant == 1)
  {
    launch_config.grid_x = checked_u32(clusters * spec.cluster_m, "trimul KF K2 grid.x");
    launch_config.grid_y = spec.cluster_n;
    launch_config.grid_z = 1;
    device_params.grid_x = checked_i32(launch_config.grid_x, "trimul KF K2 grid.x");
  }
  else
  {
    launch_config.grid_x = spec.cluster_m;
    launch_config.grid_y = spec.cluster_n;
    launch_config.grid_z = checked_u32(clusters, "trimul KF K2 grid.z");
  }
  launch_config.stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream));
  /* The source kernels request no programmatic dependent launch: under CUDA-graph replay a kernel could read
   * its predecessor's output mid-write.
   */
  launch_config.programmatic_stream_serialization = 0;
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(trimul_kf_k2_sm100)");
}

KernelSpec make_kernel_spec(embedded::CubinImage const& image)
{
  return KernelSpec{
    image.cubin.target_sm,
    image.cubin.kernel_sm,
    image.outgoing,
    image.kernel_variant,
    image.tile_m,
    image.tile_n,
    image.cluster_m,
    image.defer_kmin,
    image.split_epi,
    image.num_threads,
    image.dynamic,
    image.cluster_n,
    image.clc,
    image.tail_w,
  };
}

embedded::CubinImage const& find_embedded_cubin(
  std::int32_t target_sm,
  DType dtype,
  bool outgoing,
  std::int32_t kernel_variant,
  std::uint32_t tile_n,
  std::uint32_t cluster_m,
  std::uint32_t defer_kmin,
  bool split_epi,
  bool dynamic,
  std::uint32_t cluster_n,
  bool clc,
  std::uint32_t tail_w)
{
  bool const is_bfloat16 = dtype == DType::kBFloat16;
  embedded::RegistryView const registry = embedded::registry();
  for (std::size_t index = 0; index < registry.count; ++index)
  {
    embedded::CubinImage const& image = registry.images[index];
    if (
      cubin_supports_sm(image.cubin, target_sm) && image.is_bfloat16 == is_bfloat16 && image.outgoing == outgoing
      && image.kernel_variant == kernel_variant && image.tile_n == tile_n && image.cluster_m == cluster_m
      && image.defer_kmin == defer_kmin && image.split_epi == split_epi && image.dynamic == dynamic
      && image.cluster_n == cluster_n && image.clc == clc && image.tail_w == tail_w)
      return image;
  }
  throw std::invalid_argument(
    "No embedded trimul KF K2 CUBIN for SM" + std::to_string(target_sm) + ", K2_" + std::to_string(kernel_variant)
    + ", outgoing=" + (outgoing ? "true" : "false") + ", tile_n=" + std::to_string(tile_n)
    + ", cluster_m=" + std::to_string(cluster_m) + ", defer_kmin=" + std::to_string(defer_kmin)
    + ", split_epi=" + (split_epi ? "true" : "false") + ", dynamic=" + (dynamic ? "true" : "false") + ", cluster_n="
    + std::to_string(cluster_n) + ", clc=" + (clc ? "true" : "false") + ", tail_w=" + std::to_string(tail_w));
}

std::size_t preload_kernels(CUcontext context, std::int32_t device_sm)
{
  return preload_registry_kernels(context, device_sm, embedded::registry());
}

CubinPreloadRegistration const kPreloader{
  "trimul_kf_k2",
  &preload_kernels,
};

} // namespace

std::vector<KernelSpec> kernel_specs()
{
  return map_registry(embedded::registry(), make_kernel_spec);
}

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  DType dtype,
  bool outgoing,
  std::int32_t kernel_variant,
  std::uint32_t tile_n,
  std::uint32_t cluster_m,
  std::uint32_t defer_kmin,
  bool split_epi,
  bool dynamic,
  std::uint32_t cluster_n,
  bool clc,
  std::uint32_t tail_w)
{
  embedded::CubinImage const& image = find_embedded_cubin(
    target_sm,
    dtype,
    outgoing,
    kernel_variant,
    tile_n,
    cluster_m,
    defer_kmin,
    split_epi,
    dynamic,
    cluster_n,
    clc,
    tail_w);
  return KernelConfig{make_kernel_spec(image), dtype, image.cubin, &image};
}

void launch(KernelConfig const& config, LaunchParams const& params)
{
  validate_image(config);
  CUcontext const context = current_cuda_context();
  std::int32_t const device_sm = cuda_sm_for_context(context);
  if (device_sm != config.spec.target_sm || !cubin_supports_sm(config.cubin, device_sm))
  {
    throw std::invalid_argument(
      "trimul KF K2 config selects SM" + std::to_string(config.spec.target_sm) + " but current device is SM"
      + std::to_string(device_sm));
  }
  cubin_kernel_t const loaded = load_embedded_kernel(context, config.cubin);
  if (config.cubin.kernel_sm == 100)
    launch_sm100(loaded, config, params, context, dynamic_smem_bytes(config));
  else
    launch_sm90(loaded, config, params, context, dynamic_smem_bytes(config));
}

} // namespace bioir::cutedsl::trimul_kf_k2
