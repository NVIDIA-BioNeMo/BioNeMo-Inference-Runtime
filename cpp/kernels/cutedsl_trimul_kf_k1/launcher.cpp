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
#include <stdexcept>
#include <string>

namespace bioir::cutedsl::trimul_kf_k1
{
namespace
{

constexpr char kSM90LaunchAbi[] = "trimul_kf_k1_sm90_v1";
/* Rows per K1 tile, which the persistent CTAs stride over. */
constexpr std::uint64_t kTileRows = 128;
/* The ping-pong K1 variant that hands its row statistics to K3. */
constexpr std::int32_t kStatsVariant = 2;

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
    throw std::invalid_argument("trimul KF K1 CUBIN has an incompatible launch ABI");
  }
  if (config.embedded_image == nullptr || !config.embedded_image->sm90.enabled)
    throw std::invalid_argument("trimul KF K1 CUBIN is missing its Hopper launch metadata");
  if (!cubin_supports_sm(config.cubin, config.spec.target_sm))
    throw std::invalid_argument(
      "embedded CUBIN does not support configured device SM" + std::to_string(config.spec.target_sm));
  KernelSpec const& spec = config.spec;
  if (spec.C <= 0 || spec.D <= 0 || spec.num_threads == 0 || spec.kernel_variant < 0 || spec.kernel_variant > 2)
    throw std::invalid_argument("trimul KF K1 CUBIN has invalid launch geometry");
}

/* A flat row-major [rows, cols] operand as a rank-2 TMA source. */
TmaTensorSource matrix_source(Tensor1View const& view, std::int64_t rows, std::int64_t cols)
{
  return TmaTensorSource{
    view.data,
    {static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(cols), 0, 0},
    {static_cast<std::uint64_t>(cols), 1, 0, 0},
  };
}

/* A flat channel-major [nb, D, n * n] buffer as the kernel's (n * n, D, nb) TMA source. */
TmaTensorSource channel_major_source(Tensor1View const& view, std::int64_t nn, std::int64_t D, std::int64_t nb)
{
  return TmaTensorSource{
    view.data,
    {static_cast<std::uint64_t>(nn), static_cast<std::uint64_t>(D), static_cast<std::uint64_t>(nb), 0},
    {1, static_cast<std::uint64_t>(nn), static_cast<std::uint64_t>(D * nn), 0},
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
  std::int64_t const nn = n * n;
  std::int64_t const rows = params.rows;
  if (n <= 0 || nb <= 0 || rows != nb * nn)
    throw std::invalid_argument("trimul KF K1 needs rows == nb * n * n with positive n and nb");
  bool const pingpong = spec.kernel_variant != 0;
  std::int64_t const w_rows = (pingpong ? 4 : 2) * D;
  std::uint64_t const num_tiles = ceil_div(static_cast<std::uint64_t>(rows), kTileRows);

  validate_flat(params.x, rows * C, 16, device, "x");
  validate_tensor(params.seqlen, "seqlen", 4);
  check_device(params.seqlen.device, device, "seqlen");
  if (params.seqlen.shape[0] < nb * n)
    throw std::invalid_argument("seqlen needs one length per (b, i) row");
  validate_flat(params.w_in, w_rows * C, 16, device, "w_in");
  if (!pingpong)
    validate_flat(params.w_gate_in, 2 * D * C, 16, device, "w_gate_in");
  validate_flat(params.vec_in, 8 * D, 16, device, "vec_in");
  validate_flat(params.a, nb * D * nn, 16, device, "a");
  validate_flat(params.b, nb * D * nn, 16, device, "b");
  if (spec.kernel_variant == kStatsVariant)
  {
    validate_tensor(params.stats, "stats", 16);
    check_device(params.stats.device, device, "stats");
    if (static_cast<std::uint64_t>(params.stats.shape[0]) < 2 * num_tiles * kTileRows)
      throw std::invalid_argument("stats needs two values per row, rounded up to whole 128-row tiles");
  }

  embedded::SM90LaunchInfo const& metadata = config.embedded_image->sm90;
  CUtensorMapDataType const expected_dtype = tma_data_type(true);
  abi::SM90Params device_params{};
  encode_tma_descriptor(device_params.x_tma, metadata.x, expected_dtype, matrix_source(params.x, rows, C), "x");
  encode_tma_descriptor(
    device_params.w_proj_tma, metadata.w_proj, expected_dtype, matrix_source(params.w_in, w_rows, C), "w_in");
  encode_tma_descriptor(
    device_params.a_tma, metadata.a, expected_dtype, channel_major_source(params.a, nn, D, nb), "a");
  encode_tma_descriptor(
    device_params.b_tma, metadata.b, expected_dtype, channel_major_source(params.b, nn, D, nb), "b");
  /* Only the weight folds are static. */
  finalize_sm90_tma_atom(device_params.x_tma, true);
  finalize_sm90_tma_atom(device_params.w_proj_tma, sm90_static_tma_operation_tag(w_rows * C));
  finalize_sm90_tma_atom(device_params.a_tma, true);
  finalize_sm90_tma_atom(device_params.b_tma, true);
  if (!pingpong)
  {
    encode_tma_descriptor(
      device_params.w_gate_tma,
      metadata.w_gate,
      expected_dtype,
      matrix_source(params.w_gate_in, 2 * D, C),
      "w_gate_in");
    finalize_sm90_tma_atom(device_params.w_gate_tma, sm90_static_tma_operation_tag(2 * D * C));
  }
  device_params.x_coord = CoordTensorS1{{static_cast<std::int32_t>(rows)}};
  device_params.a_coord = CoordTensorS2{{static_cast<std::int32_t>(nn), static_cast<std::int32_t>(nb)}};
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
  std::size_t const parameter_count = abi::pack_sm90_kernel_params(&device_params, spec.kernel_variant, kernel_params);
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
  /* Persistent CTAs stride over the row tiles; K1 waits on its predecessor before reading anything. */
  cubin_launch_config_t launch_config{};
  launch_config.grid_x = checked_u32(
    std::min<std::uint64_t>(num_tiles, static_cast<std::uint64_t>(multiprocessor_count)), "trimul KF K1 grid.x");
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
  launch_config.programmatic_stream_serialization = 1;
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(trimul_kf_k1_sm90)");
}

KernelSpec make_kernel_spec(embedded::CubinImage const& image)
{
  return KernelSpec{
    image.cubin.target_sm,
    image.cubin.kernel_sm,
    image.C,
    image.D,
    image.kernel_variant,
    image.num_threads,
  };
}

embedded::CubinImage const&
find_embedded_cubin(std::int32_t target_sm, DType dtype, std::int32_t C, std::int32_t D, std::int32_t kernel_variant)
{
  bool const is_bfloat16 = dtype == DType::kBFloat16;
  embedded::RegistryView const registry = embedded::registry();
  for (std::size_t index = 0; index < registry.count; ++index)
  {
    embedded::CubinImage const& image = registry.images[index];
    if (
      cubin_supports_sm(image.cubin, target_sm) && image.is_bfloat16 == is_bfloat16 && image.C == C && image.D == D
      && image.kernel_variant == kernel_variant)
      return image;
  }
  throw std::invalid_argument(
    "No embedded trimul KF K1 CUBIN for SM" + std::to_string(target_sm) + ", C=" + std::to_string(C)
    + ", D=" + std::to_string(D) + ", K1_" + std::to_string(kernel_variant));
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

KernelConfig
make_kernel_config(std::int32_t target_sm, DType dtype, std::int32_t C, std::int32_t D, std::int32_t kernel_variant)
{
  embedded::CubinImage const& image = find_embedded_cubin(target_sm, dtype, C, D, kernel_variant);
  return KernelConfig{make_kernel_spec(image), dtype, image.cubin, &image};
}

void launch(KernelConfig const& config, LaunchParams const& params)
{
  validate_image(config);
  CUcontext const context = current_cuda_context();
  cubin_kernel_t const loaded = load_for_current_device(config, context);
  launch_k1(loaded, config, params, context, dynamic_smem_bytes(config));
}

} // namespace bioir::cutedsl::trimul_kf_k1
