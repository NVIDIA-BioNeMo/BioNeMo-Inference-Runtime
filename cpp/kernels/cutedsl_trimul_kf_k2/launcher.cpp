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
#include <stdexcept>
#include <string>

namespace bioir::cutedsl::trimul_kf_k2
{
namespace
{

constexpr char kSM90LaunchAbi[] = "trimul_kf_k2_sm90_v1";
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

void validate_flat(FlatTensorView const& view, std::int64_t elements, std::int32_t device, char const* name)
{
  validate_flat_tensor(view, name, 16);
  check_device(view.device, device, name);
  if (view.extent != elements)
  {
    throw std::invalid_argument(
      std::string(name) + " must hold " + std::to_string(elements) + " elements, got " + std::to_string(view.extent));
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

/* Row pitch and plane stride of a and b: both whole 16-byte TMA strides. */
void validate_ab_strides(std::int64_t n, std::int64_t pitch, std::int64_t plane)
{
  constexpr std::int64_t kStrideElements = 8;
  if (
    pitch < n || pitch % kStrideElements != 0 || plane % kStrideElements != 0
    || plane < checked_mul(n, pitch, "trimul KF K2 a/b plane extent n * pitch"))
  {
    throw std::invalid_argument(
      "trimul KF K2 needs a/b row pitch >= n and plane stride >= n * pitch, both multiples of "
      + std::to_string(kStrideElements) + " elements; got n=" + std::to_string(n) + ", pitch=" + std::to_string(pitch)
      + ", plane=" + std::to_string(plane));
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
  if (
    config.cubin.kernel_sm != 90 || config.cubin.launch_abi == nullptr
    || std::strcmp(config.cubin.launch_abi, kSM90LaunchAbi) != 0)
  {
    throw std::invalid_argument("trimul KF K2 CUBIN has an incompatible launch ABI");
  }
  if (config.embedded_image == nullptr || !config.embedded_image->sm90.enabled)
    throw std::invalid_argument("trimul KF K2 CUBIN is missing its Hopper launch metadata");
  if (!cubin_supports_sm(config.cubin, config.spec.target_sm))
    throw std::invalid_argument(
      "embedded CUBIN does not support configured device SM" + std::to_string(config.spec.target_sm));
  KernelSpec const& spec = config.spec;
  if (
    spec.kernel_variant < 0 || spec.kernel_variant > 2 || spec.tile_m == 0 || spec.tile_n == 0 || spec.cluster_m == 0
    || spec.num_threads == 0)
  {
    throw std::invalid_argument("trimul KF K2 CUBIN has invalid launch geometry");
  }
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
  if (n <= 0 || l <= 0)
    throw std::invalid_argument("trimul KF K2 needs positive n and l");
  validate_ab_strides(n, pitch, plane);
  char const* const span_name = "trimul KF K2 a/b span";
  std::int64_t const ab_span = checked_add(
    checked_add(checked_mul(l - 1, plane, span_name), checked_mul(n - 1, pitch, span_name), span_name), n, span_name);
  validate_span(params.a, ab_span, device, "a");
  validate_span(params.b, ab_span, device, "b");
  validate_flat(
    params.prod, checked_mul(l, checked_mul(n, n, "trimul KF K2 prod"), "trimul KF K2 prod"), device, "prod");

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
   * dense and row-major.
   */
  CUtensorMapDataType const expected_dtype = tma_data_type(true);
  abi::SM90Params device_params{};
  TmaTensorSource const a_source = matrices_source(params.a, n, l, pitch, plane, !spec.outgoing);
  TmaTensorSource const b_source = matrices_source(params.b, n, l, pitch, plane, !spec.outgoing);
  TmaTensorSource const prod_source = matrices_source(params.prod, n, l, n, n * n, false);
  encode_tma_descriptor(device_params.a_tma, metadata.a, expected_dtype, a_source, "a");
  encode_tma_descriptor(device_params.b_tma, metadata.b, expected_dtype, b_source, "b");
  encode_tma_descriptor(device_params.prod_tma, metadata.prod, expected_dtype, prod_source, "prod");
  finalize_sm90_tma_atom(device_params.a_tma, sm90_tma_operation_tag(a_source));
  finalize_sm90_tma_atom(device_params.b_tma, sm90_tma_operation_tag(b_source));
  finalize_sm90_tma_atom(device_params.prod_tma, sm90_tma_operation_tag(prod_source));
  CoordTensorS3 const coord{{static_cast<std::int32_t>(n), static_cast<std::int32_t>(n), static_cast<std::int32_t>(l)}};
  device_params.a_coord = coord;
  device_params.b_coord = coord;
  device_params.prod_coord = coord;

  std::int32_t const multiprocessor_count = cuda_multiprocessor_count_for_context(context);
  if (multiprocessor_count <= 0)
    throw std::invalid_argument("current CUDA device has no active multiprocessors");
  std::uint64_t const sms = static_cast<std::uint64_t>(multiprocessor_count);

  cubin_launch_config_t launch_config{};
  if (spec.kernel_variant != 1)
  {
    /* Persistent clusters of cluster_m tiles, vertically adjacent for K2_0 and horizontally for
     * K2_2, walked (m, n, l) with m fastest. The grid is the smallest one with the same number of
     * passes as the hardware-wide grid.
     */
    bool const pairs_columns = spec.kernel_variant == 2;
    std::uint64_t const m_clusters = ceil_div(m_tiles, pairs_columns ? 1 : cluster_m);
    std::uint64_t const n_clusters = ceil_div(n_tiles, pairs_columns ? cluster_m : 1);
    std::uint64_t const work_units = m_clusters * n_clusters * static_cast<std::uint64_t>(l);
    std::uint64_t const max_active_clusters = sms / cluster_m;
    if (max_active_clusters == 0)
      throw std::invalid_argument("current CUDA device cannot hold one trimul KF K2 cluster");
    std::uint64_t const passes = ceil_div(work_units, max_active_clusters);
    device_params.schedule[0] = static_cast<std::int32_t>(checked_u32(m_clusters, "trimul KF K2 row clusters"));
    device_params.schedule[1] = static_cast<std::int32_t>(checked_u32(n_clusters, "trimul KF K2 column clusters"));
    device_params.schedule[2] = static_cast<std::int32_t>(checked_u32(work_units, "trimul KF K2 work units"));
    launch_config.grid_x = cluster_m;
    launch_config.grid_y = 1;
    launch_config.grid_z = checked_u32(ceil_div(work_units, passes), "trimul KF K2 grid.z");
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
  std::size_t const parameter_count = abi::pack_sm90_kernel_params(&device_params, spec.kernel_variant, kernel_params);
  if (parameter_count != abi::sm90_parameter_count(spec.kernel_variant))
    throw std::logic_error("trimul KF K2 parameter packer produced the wrong ABI count");

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
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(trimul_kf_k2_sm90)");
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
  bool split_epi)
{
  bool const is_bfloat16 = dtype == DType::kBFloat16;
  embedded::RegistryView const registry = embedded::registry();
  for (std::size_t index = 0; index < registry.count; ++index)
  {
    embedded::CubinImage const& image = registry.images[index];
    if (
      cubin_supports_sm(image.cubin, target_sm) && image.is_bfloat16 == is_bfloat16 && image.outgoing == outgoing
      && image.kernel_variant == kernel_variant && image.tile_n == tile_n && image.cluster_m == cluster_m
      && image.defer_kmin == defer_kmin && image.split_epi == split_epi)
      return image;
  }
  throw std::invalid_argument(
    "No embedded trimul KF K2 CUBIN for SM" + std::to_string(target_sm) + ", K2_" + std::to_string(kernel_variant)
    + ", outgoing=" + (outgoing ? "true" : "false") + ", tile_n=" + std::to_string(tile_n)
    + ", cluster_m=" + std::to_string(cluster_m) + ", defer_kmin=" + std::to_string(defer_kmin)
    + ", split_epi=" + (split_epi ? "true" : "false"));
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
  bool split_epi)
{
  embedded::CubinImage const& image
    = find_embedded_cubin(target_sm, dtype, outgoing, kernel_variant, tile_n, cluster_m, defer_kmin, split_epi);
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
  launch_sm90(loaded, config, params, context, dynamic_smem_bytes(config));
}

} // namespace bioir::cutedsl::trimul_kf_k2
