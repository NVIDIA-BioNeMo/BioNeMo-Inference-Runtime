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

#include "trimul_kf_k3_registry.h"

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace bioir::cutedsl::trimul_kf_k3
{
namespace
{

constexpr char kSM90LaunchAbi[] = "trimul_kf_k3_sm90_v2";
constexpr char kSM90ResidentLaunchAbi[] = "trimul_kf_k3_sm90_resident_v1";
/* Rows one launch addresses: TMA coordinates are int32. */
constexpr std::int64_t kMaxLaunchRows = std::numeric_limits<std::int32_t>::max() - 128;
/* x and output are bf16. */
constexpr std::uint64_t kElementBytes = 2;

/* Whether the image takes the weights-resident launch ABI (each CTA keeps one column tile's weights
 * resident, stores its output from registers and takes every batch item in one launch) or the
 * streaming one. Any other ABI is an image this launcher cannot run.
 */
bool resident_abi(EmbeddedCubinImage const& cubin)
{
  if (cubin.launch_abi != nullptr)
  {
    if (std::strcmp(cubin.launch_abi, kSM90ResidentLaunchAbi) == 0)
      return true;
    if (std::strcmp(cubin.launch_abi, kSM90LaunchAbi) == 0)
      return false;
  }
  throw std::invalid_argument("trimul KF K3 CUBIN has an incompatible launch ABI");
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
  if (config.cubin.kernel_sm != 90)
    throw std::invalid_argument("trimul KF K3 CUBIN has an incompatible launch ABI");
  /* The resident parameter bank always carries K1's row statistics. */
  if (resident_abi(config.cubin) && !config.spec.reads_stats)
    throw std::invalid_argument("trimul KF K3 resident CUBIN must read K1's row statistics");
  if (config.embedded_image == nullptr || !config.embedded_image->sm90.enabled)
    throw std::invalid_argument("trimul KF K3 CUBIN is missing its Hopper launch metadata");
  if (!cubin_supports_sm(config.cubin, config.spec.target_sm))
    throw std::invalid_argument(
      "embedded CUBIN does not support configured device SM" + std::to_string(config.spec.target_sm));
  KernelSpec const& spec = config.spec;
  if (
    spec.C <= 0 || spec.D <= 0 || spec.num_threads == 0 || spec.tile_m == 0 || spec.tile_ctas == 0
    || spec.kernel_variant < 0)
    throw std::invalid_argument("trimul KF K3 CUBIN has invalid launch geometry");
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

/* A flat channel-major [nb, D, n * n] buffer as the kernel's (n * n, D, nb) TMA source. */
TmaTensorSource channel_major_source(FlatTensorView const& view, std::int64_t nn, std::int64_t D, std::int64_t nb)
{
  return TmaTensorSource{
    view.data,
    {static_cast<std::uint64_t>(nn), static_cast<std::uint64_t>(D), static_cast<std::uint64_t>(nb), 0},
    {1, static_cast<std::uint64_t>(nn), static_cast<std::uint64_t>(D * nn), 0},
  };
}

/* A pitched channel-major product (rows pitch apart, channels plane apart, batch items D * plane apart)
 * as a pitched image's (n, D, n, nb) TMA source: (j, channel, i, b).
 */
TmaTensorSource pitched_channel_major_source(
  FlatTensorView const& view, std::int64_t n, std::int64_t D, std::int64_t nb, std::int64_t pitch, std::int64_t plane)
{
  std::uint64_t const extent = static_cast<std::uint64_t>(n);
  return TmaTensorSource{
    view.data,
    {extent, static_cast<std::uint64_t>(D), extent, static_cast<std::uint64_t>(nb)},
    {1, static_cast<std::uint64_t>(plane), static_cast<std::uint64_t>(pitch), static_cast<std::uint64_t>(D * plane)},
  };
}

/* A flat row-major [nb * n * n, C] operand as a pitched image's (n, C, nb * n) TMA source: (j, k, b * n + i). */
TmaTensorSource row_chunks_source(std::uint64_t data, std::int64_t n, std::int64_t C, std::int64_t pair_rows)
{
  return TmaTensorSource{
    data,
    {static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(C), static_cast<std::uint64_t>(pair_rows), 0},
    {static_cast<std::uint64_t>(C), 1, static_cast<std::uint64_t>(n * C), 0},
  };
}

/* A pitched product's row pitch and plane stride: whole 16-byte TMA strides, rows at least n long and
 * planes exactly n rows, as K2 writes them.
 */
void validate_prod_strides(std::int64_t n, std::int64_t pitch, std::int64_t plane)
{
  constexpr std::int64_t kStrideElements = 8;
  if (pitch < n || pitch % kStrideElements != 0 || plane != checked_mul(n, pitch, "trimul KF K3 prod plane"))
  {
    throw std::invalid_argument(
      "trimul KF K3 needs a pitched product's row pitch >= n, a multiple of " + std::to_string(kStrideElements)
      + " elements, and its plane stride n * pitch; got n=" + std::to_string(n) + ", pitch=" + std::to_string(pitch)
      + ", plane=" + std::to_string(plane));
  }
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
  bool const stats_variant = spec.reads_stats;
  std::int64_t const C = spec.C;
  std::int64_t const D = spec.D;
  std::int64_t const n = params.n;
  std::int64_t const nb = params.nb;
  std::int64_t const rows = params.rows;
  if (n <= 0 || nb <= 0)
    throw std::invalid_argument("trimul KF K3 needs positive n and nb");
  std::int64_t const nn = checked_mul(n, n, "trimul KF K3 plane");
  if (rows != checked_mul(nb, nn, "trimul KF K3 rows"))
    throw std::invalid_argument("trimul KF K3 needs rows == nb * n * n");
  if (rows > kMaxLaunchRows)
  {
    throw std::invalid_argument(
      "trimul KF K3 addresses at most " + std::to_string(kMaxLaunchRows) + " rows a launch; got "
      + std::to_string(rows));
  }
  bool const resident_variant = resident_abi(config.cubin);
  /* A resident image loads each slab as two 64-row units, none straddling two batch items: n * n % 64 == 0. */
  constexpr std::int64_t kResidentTokenAlign = 8;
  if (resident_variant && n % kResidentTokenAlign != 0)
  {
    throw std::invalid_argument(
      "trimul KF K3 resident CUBIN needs n a multiple of " + std::to_string(kResidentTokenAlign)
      + ", got n=" + std::to_string(n));
  }
  std::uint64_t const tile_rows = spec.tile_m;
  std::uint64_t const flat_tiles = ceil_div(static_cast<std::uint64_t>(rows), tile_rows);
  /* A pitched image tiles every one of the nb * n pair rows in chunks: tile_m columns a tile, or half a slab of
   * a resident image, two chunks a slab.
   */
  std::uint64_t const pair_rows = static_cast<std::uint64_t>(nb * n);
  std::uint64_t const num_tiles = !spec.pitched ? flat_tiles
    : resident_variant ? ceil_div(pair_rows * ceil_div(static_cast<std::uint64_t>(n), tile_rows / 2), std::uint64_t{2})
                       : pair_rows * ceil_div(static_cast<std::uint64_t>(n), tile_rows);

  std::int64_t const operand_elements = checked_mul(rows, C, "trimul KF K3 x");
  if (spec.pitched)
  {
    validate_prod_strides(n, params.prod_pitch, params.prod_plane);
    char const* const span_name = "trimul KF K3 prod span";
    std::int64_t const span = checked_add(
      checked_add(
        checked_mul(checked_mul(nb, D, span_name) - 1, params.prod_plane, span_name),
        checked_mul(n - 1, params.prod_pitch, span_name),
        span_name),
      n,
      span_name);
    validate_flat_tensor(params.prod, "prod", 16);
    check_device(params.prod.device, device, "prod");
    if (params.prod.extent < span)
    {
      throw std::invalid_argument(
        "prod must span at least " + std::to_string(span) + " elements, got " + std::to_string(params.prod.extent));
    }
  }
  else
  {
    validate_flat(
      params.prod, checked_mul(nb, checked_mul(D, nn, "trimul KF K3 prod"), "trimul KF K3 prod"), 16, device, "prod");
  }
  validate_flat(params.x, operand_elements, 16, device, "x");
  validate_flat(params.w_out, C * D, 16, device, "w_out");
  validate_flat(params.w_gate_out, C * C, 16, device, "w_gate_out");
  validate_flat(params.vec_out, 4 * C, 16, device, "vec_out");
  validate_flat(params.output, operand_elements, 16, device, "output");
  /* K3 reads x while it writes output, so the two must not share any bytes. */
  std::uint64_t const operand_bytes = static_cast<std::uint64_t>(operand_elements) * kElementBytes;
  if (params.output.data < params.x.data + operand_bytes && params.x.data < params.output.data + operand_bytes)
    throw std::invalid_argument("trimul KF K3 output must not alias x");
  if (stats_variant)
  {
    validate_flat_tensor(params.stats, "stats", 16);
    check_device(params.stats.device, device, "stats");
    /* K1 pads each batch item's statistics to whole 128-row tiles; a resident image reads every item's
     * from one launch, the others one item's or a whole number of tiles.
     */
    std::uint64_t const stats_rows = resident_variant
      ? static_cast<std::uint64_t>(nb) * ceil_div(static_cast<std::uint64_t>(nn), std::uint64_t{128}) * 128
      : flat_tiles * tile_rows;
    if (static_cast<std::uint64_t>(params.stats.extent) < 2 * stats_rows)
    {
      throw std::invalid_argument("stats needs two values per row, rounded up to whole 128-row tiles per batch item");
    }
  }
  if (spec.residual)
  {
    validate_tensor(params.seqlen, "seqlen", 4);
    check_device(params.seqlen.device, device, "seqlen");
    if (params.seqlen.shape[0] < nb * n)
      throw std::invalid_argument("seqlen needs one length per (b, i) row");
  }

  embedded::SM90LaunchInfo const& metadata = config.embedded_image->sm90;
  CUtensorMapDataType const expected_dtype = tma_data_type(true);
  TmaTensorSource const prod_source = spec.pitched
    ? pitched_channel_major_source(params.prod, n, D, nb, params.prod_pitch, params.prod_plane)
    : channel_major_source(params.prod, nn, D, nb);
  TmaTensorSource const x_source
    = spec.pitched ? row_chunks_source(params.x.data, n, C, nb * n) : matrix_source(params.x.data, rows, C);
  TmaTensorSource const w_out_source = matrix_source(params.w_out.data, C, D);
  TmaTensorSource const w_gate_source = matrix_source(params.w_gate_out.data, C, C);
  std::int32_t const tile_count = static_cast<std::int32_t>(checked_u32(num_tiles, "trimul KF K3 tile count"));

  /* Both banks share their leading TMA operands and coordinates. */
  auto encode_inputs = [&](auto& bank)
  {
    encode_tma_descriptor(bank.prod_tma, metadata.prod, expected_dtype, prod_source, "prod");
    encode_tma_descriptor(bank.x_tma, metadata.x, expected_dtype, x_source, "x");
    encode_tma_descriptor(bank.w_out_tma, metadata.w_out, expected_dtype, w_out_source, "w_out");
    encode_tma_descriptor(bank.w_gate_tma, metadata.w_gate, expected_dtype, w_gate_source, "w_gate_out");
    finalize_sm90_tma_atom(bank.prod_tma, sm90_tma_operation_tag(prod_source));
    finalize_sm90_tma_atom(bank.x_tma, sm90_tma_operation_tag(x_source));
    finalize_sm90_tma_atom(bank.w_out_tma, sm90_tma_operation_tag(w_out_source));
    finalize_sm90_tma_atom(bank.w_gate_tma, sm90_tma_operation_tag(w_gate_source));
    bank.prod_coord = CoordTensorS2{{static_cast<std::int32_t>(nn), static_cast<std::int32_t>(nb)}};
    bank.x_coord = CoordTensorS1{{static_cast<std::int32_t>(rows)}};
    /* a pitched image's (n, D, n, nb) and (n, C, nb * n) maps, D and C static */
    bank.prod_rows_coord
      = CoordTensorS3{{static_cast<std::int32_t>(n), static_cast<std::int32_t>(n), static_cast<std::int32_t>(nb)}};
    bank.x_rows_coord = CoordTensorS2{{static_cast<std::int32_t>(n), static_cast<std::int32_t>(nb * n)}};
    bank.vec.data = static_cast<CUdeviceptr>(params.vec_out.data);
    if (stats_variant)
    {
      bank.stats.data = static_cast<CUdeviceptr>(params.stats.data);
      bank.stats.dynamic_shapes[0] = static_cast<std::int32_t>(rows);
    }
    if (spec.residual)
      bank.seqlen = make_tensor1_descriptor(params.seqlen);
    bank.n = static_cast<std::int32_t>(n);
    bank.eps = params.eps;
    bank.rows = static_cast<std::int32_t>(rows);
  };

  abi::SM90Params device_params{};
  abi::SM90ResidentParams resident_params{};
  void* kernel_params[std::max(abi::kSM90MaxParameterCount, abi::kSM90ResidentMaxParameterCount)]{};
  if (resident_variant)
  {
    if (metadata.output.rank != 0)
      throw std::invalid_argument("trimul KF K3 resident CUBIN records an output map it does not take");
    encode_inputs(resident_params);
    /* The kernel stores 16-byte vectors through output's address; its extent is not read. */
    resident_params.output.data = static_cast<CUdeviceptr>(params.output.data);
    resident_params.num_slabs = tile_count;
    std::size_t const parameter_count
      = abi::pack_sm90_resident_kernel_params(&resident_params, spec.residual, spec.pitched, kernel_params);
    if (parameter_count != abi::sm90_resident_parameter_count(spec.residual))
      throw std::logic_error("trimul KF K3 resident parameter packer produced the wrong ABI count");
  }
  else
  {
    TmaTensorSource const output_source
      = spec.pitched ? row_chunks_source(params.output.data, n, C, nb * n) : matrix_source(params.output.data, rows, C);
    encode_inputs(device_params);
    encode_tma_descriptor(device_params.output_tma, metadata.output, expected_dtype, output_source, "output");
    finalize_sm90_tma_atom(device_params.output_tma, sm90_tma_operation_tag(output_source));
    device_params.output_coord = device_params.x_coord;
    device_params.output_rows_coord = device_params.x_rows_coord;
    device_params.num_tiles = tile_count;
    std::size_t const parameter_count
      = abi::pack_sm90_kernel_params(&device_params, stats_variant, spec.residual, spec.pitched, kernel_params);
    if (parameter_count != abi::sm90_parameter_count(stats_variant, spec.residual))
      throw std::logic_error("trimul KF K3 parameter packer produced the wrong ABI count");
  }

  for (std::uint32_t dimension : metadata.cluster_dims)
  {
    if (dimension != 1)
      throw std::invalid_argument("trimul KF K3 CUBIN must launch unit clusters");
  }
  std::int32_t const multiprocessor_count = cuda_multiprocessor_count_for_context(context);
  if (multiprocessor_count <= 0)
    throw std::invalid_argument("current CUDA device has no active multiprocessors");
  /* Persistent CTAs stride over the row tiles in whole groups of tile_ctas: a partial group would
   * drop its column groups (resident C=384: 3 * min(SMs / 3, slabs) CTAs, one per column tile of each lane).
   * K3 waits on K2 before reading the product.
   */
  std::uint64_t const tile_ctas = spec.tile_ctas;
  std::uint64_t const concurrent_tiles = static_cast<std::uint64_t>(multiprocessor_count) / tile_ctas;
  if (concurrent_tiles == 0)
    throw std::invalid_argument("current CUDA device cannot hold one trimul KF K3 row tile's column-group CTAs");
  cubin_launch_config_t launch_config{};
  launch_config.grid_x
    = checked_u32(std::min<std::uint64_t>(num_tiles, concurrent_tiles) * tile_ctas, "trimul KF K3 grid.x");
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
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(trimul_kf_k3_sm90)");
}

KernelSpec make_kernel_spec(embedded::CubinImage const& image)
{
  return KernelSpec{
    image.cubin.target_sm,
    image.cubin.kernel_sm,
    image.C,
    image.D,
    image.kernel_variant,
    image.residual,
    image.reads_stats,
    image.pitched,
    image.num_threads,
    image.tile_m,
    image.tile_ctas,
  };
}

embedded::CubinImage const& find_embedded_cubin(
  std::int32_t target_sm,
  DType dtype,
  std::int32_t C,
  std::int32_t D,
  std::int32_t kernel_variant,
  bool residual,
  bool pitched)
{
  bool const is_bfloat16 = dtype == DType::kBFloat16;
  embedded::RegistryView const registry = embedded::registry();
  for (std::size_t index = 0; index < registry.count; ++index)
  {
    embedded::CubinImage const& image = registry.images[index];
    if (
      cubin_supports_sm(image.cubin, target_sm) && image.is_bfloat16 == is_bfloat16 && image.C == C && image.D == D
      && image.kernel_variant == kernel_variant && image.residual == residual && image.pitched == pitched)
      return image;
  }
  throw std::invalid_argument(
    "No embedded trimul KF K3 CUBIN for SM" + std::to_string(target_sm) + ", C=" + std::to_string(C)
    + ", D=" + std::to_string(D) + ", K3_" + std::to_string(kernel_variant)
    + ", residual=" + (residual ? "true" : "false") + ", pitched=" + (pitched ? "true" : "false"));
}

std::size_t preload_kernels(CUcontext context, std::int32_t device_sm)
{
  return preload_registry_kernels(context, device_sm, embedded::registry());
}

CubinPreloadRegistration const kPreloader{
  "trimul_kf_k3",
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
  std::int32_t C,
  std::int32_t D,
  std::int32_t kernel_variant,
  bool residual,
  bool pitched)
{
  embedded::CubinImage const& image = find_embedded_cubin(target_sm, dtype, C, D, kernel_variant, residual, pitched);
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
      "trimul KF K3 config selects SM" + std::to_string(config.spec.target_sm) + " but current device is SM"
      + std::to_string(device_sm));
  }
  cubin_kernel_t const loaded = load_embedded_kernel(context, config.cubin);
  launch_sm90(loaded, config, params, context, dynamic_smem_bytes(config));
}

} // namespace bioir::cutedsl::trimul_kf_k3
