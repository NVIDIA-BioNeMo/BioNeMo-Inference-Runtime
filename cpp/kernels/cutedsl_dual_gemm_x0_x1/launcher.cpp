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

#include "dual_gemm_x0_x1_registry.h"

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace bioir::cutedsl::dual_gemm_x0_x1
{
namespace
{

constexpr char kSM80LaunchAbi[] = "dual_gemm_x0_x1_sm80";
/* Adds actual_seqlen, residual, and i_dim to the Ampere parameter bank. */
constexpr char kSM80ResidualLaunchAbi[] = "dual_gemm_x0_x1_sm80_residual_v1";
constexpr char kSM90LaunchAbi[] = "dual_gemm_x0_x1_sm90";
constexpr char kSM90AsymmetricLaunchAbi[] = "dual_gemm_x0_x1_sm90_asym";
constexpr char kSM90ResidualLaunchAbi[] = "dual_gemm_x0_x1_sm90_residual_v1";
constexpr char kSM90AsymmetricResidualLaunchAbi[] = "dual_gemm_x0_x1_sm90_asym_residual_v1";

/* Matches the source path's unused I_dim value. */
constexpr std::int32_t kSM90UnusedIDim = 1;

enum class AsymmetricTmaAtomKind : std::uint8_t
{
  kMulticastLoad,
  kLoad,
  kStore,
};

void finalize_asymmetric_tma_atom(CUtensorMap& descriptor, AsymmetricTmaAtomKind kind)
{
  /* CuTe DSL 4.5.2 lowers each by-value non-executable TMA CopyAtom into a
   * 64-byte Hopper atom payload carried in a 128-byte kernel parameter slot.
   * cuTensorMapEncodeTiled returns the standalone CUDA tensor-map form, which
   * differs only in the atom and operation tags. Add them and clear the unused
   * upper half before the Driver copies the parameter bank. Everything else,
   * the multicast box extent included, belongs in the encoded descriptor.
   *
   * These offsets are part of launch ABI dual_gemm_x0_x1_sm90_asym. The
   * builder pins the CuTe DSL toolchain and checks the corresponding CUBIN
   * parameter sizes, so a compiler encoding change requires a new launch ABI.
   */
  constexpr std::size_t kAtomTagOffset = 8;
  constexpr std::size_t kOperationTagOffset = 10;
  constexpr std::size_t kAtomPayloadBytes = 64;
  constexpr std::uint8_t kNonExecutableAtom = 0x02U;
  constexpr std::uint8_t kNonMulticastLoad = 0x20U;
  static_assert(sizeof(CUtensorMap) == 128);

  auto* bytes = reinterpret_cast<std::uint8_t*>(&descriptor);
  bytes[kAtomTagOffset] |= kNonExecutableAtom;
  if (kind == AsymmetricTmaAtomKind::kLoad)
    bytes[kOperationTagOffset] |= kNonMulticastLoad;
  std::fill(bytes + kAtomPayloadBytes, bytes + sizeof(CUtensorMap), 0U);
}

/* A multicast load splits the tile across the cluster: every CTA fetches an
 * equal share of the outer box extent and multicasts it to its peers, so the
 * descriptor box is that much shorter than the tile the metadata records.
 */
TmaDescriptorInfo multicast_tma_descriptor(TmaDescriptorInfo info, std::uint32_t num_multicast, char const* name)
{
  if (num_multicast <= 1)
    return info;
  if (info.rank < 2)
    throw std::invalid_argument(std::string(name) + " TMA descriptor is too low-rank to multicast");
  std::uint32_t& outer_box = info.box_dims[info.rank - 1];
  if (outer_box % num_multicast != 0)
  {
    throw std::invalid_argument(
      std::string(name) + " TMA box extent " + std::to_string(outer_box) + " does not divide across "
      + std::to_string(num_multicast) + " multicasting CTAs");
  }
  outer_box /= num_multicast;
  return info;
}

/* Bare device pointers require explicit cross-device validation. */
void validate_operand_devices(KernelConfig const& config, LaunchParams const& params, std::int32_t device)
{
  auto const check = [device](std::int32_t operand_device, char const* name)
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
  };

  check(params.x0.device, "x0");
  check(params.x1.device, "x1");
  check(params.w0.device, "w0");
  check(params.w1.device, "w1");
  check(params.out.device, "out");
  if (config.fused_residual)
  {
    check(params.actual_seqlen.device, "actual_seqlen");
    check(params.residual.device, "residual");
  }
  if (config.has_bias)
  {
    check(params.bias0.device, "bias0");
    check(params.bias1.device, "bias1");
  }
}

void validate_launch(KernelConfig const& config, LaunchParams const& params)
{
  if (config.cubin.data == nullptr || config.cubin.size == 0)
    throw std::invalid_argument("embedded CUBIN image must not be empty");
  if (config.cubin.variant_id == nullptr || config.cubin.variant_id[0] == '\0')
    throw std::invalid_argument("embedded CUBIN variant_id must not be empty");
  if (config.cubin.kernel_symbol == nullptr || config.cubin.kernel_symbol[0] == '\0')
    throw std::invalid_argument("kernel_symbol must not be empty");
  bool const is_sm90 = config.cubin.kernel_sm == 90;
  bool const launch_abi_matches = config.cubin.launch_abi != nullptr
    && ((!is_sm90
         && (std::strcmp(config.cubin.launch_abi, kSM80LaunchAbi) == 0
             || std::strcmp(config.cubin.launch_abi, kSM80ResidualLaunchAbi) == 0))
        || (is_sm90
            && (std::strcmp(config.cubin.launch_abi, kSM90LaunchAbi) == 0
                || std::strcmp(config.cubin.launch_abi, kSM90AsymmetricLaunchAbi) == 0
                || std::strcmp(config.cubin.launch_abi, kSM90ResidualLaunchAbi) == 0
                || std::strcmp(config.cubin.launch_abi, kSM90AsymmetricResidualLaunchAbi) == 0)));
  if ((config.cubin.kernel_sm != 80 && config.cubin.kernel_sm != 90) || !launch_abi_matches)
  {
    throw std::invalid_argument("dual-GEMM x0_x1 CUBIN has an incompatible launch ABI");
  }
  bool const residual_launch_abi = std::strcmp(config.cubin.launch_abi, kSM80ResidualLaunchAbi) == 0
    || std::strcmp(config.cubin.launch_abi, kSM90ResidualLaunchAbi) == 0
    || std::strcmp(config.cubin.launch_abi, kSM90AsymmetricResidualLaunchAbi) == 0;
  if (residual_launch_abi != config.fused_residual)
    throw std::invalid_argument("dual-GEMM x0_x1 residual flag disagrees with its launch ABI");
  if (config.spec.kernel_sm != config.cubin.kernel_sm)
    throw std::invalid_argument("dual-GEMM x0_x1 spec disagrees with its CUBIN about the kernel ABI generation");
  if (config.cubin.non_portable_cluster_size_allowed != is_sm90)
    throw std::invalid_argument("dual-GEMM x0_x1 CUBIN has inconsistent cluster function metadata");
  if (config.spec.has_bias != config.has_bias)
    throw std::invalid_argument("dual-GEMM x0_x1 config disagrees with its CUBIN about bias");
  if (config.spec.fused_residual != config.fused_residual)
    throw std::invalid_argument("dual-GEMM x0_x1 config disagrees with its CUBIN about residual fusion");
  /* Only Hopper permits raster_factor 0. */
  if ((!is_sm90 && config.spec.raster_factor == 0) || config.spec.tile_m == 0 || config.spec.tile_n == 0)
    throw std::invalid_argument("dual-GEMM x0_x1 CUBIN has invalid launch geometry");
  if (config.spec.num_threads == 0)
    throw std::invalid_argument("dual-GEMM x0_x1 CUBIN has an invalid thread count");

  std::int32_t const configured_sm = config.spec.target_sm;
  if (!cubin_supports_sm(config.cubin, configured_sm))
  {
    throw std::invalid_argument("embedded CUBIN does not support configured device SM" + std::to_string(configured_sm));
  }

  validate_tensor(params.x0, "x0", 16);
  validate_tensor(params.x1, "x1", 16);
  validate_tensor(params.w0, "w0", 16);
  validate_tensor(params.w1, "w1", 16);
  validate_tensor(params.out, "out", 16);

  std::int32_t const M = params.x0.shape[0];
  std::int32_t const K = params.x0.shape[1];
  std::int32_t const K1 = params.x1.shape[1];
  std::int32_t const N = params.w0.shape[0];
  if (K != config.spec.K || K1 != config.spec.K1 || N != config.spec.N)
  {
    throw std::invalid_argument(
      "dual-GEMM x0_x1 operands are K0=" + std::to_string(K) + ", K1=" + std::to_string(K1) + ", N=" + std::to_string(N)
      + " but the CUBIN is K0=" + std::to_string(config.spec.K) + ", K1=" + std::to_string(config.spec.K1)
      + ", N=" + std::to_string(config.spec.N));
  }
  if (params.x1.shape[0] != M)
    throw std::invalid_argument("x0 and x1 must have the same row count");
  if (params.w1.shape[0] != N)
    throw std::invalid_argument("w0 and w1 must have the same output extent");
  if (params.w0.shape[1] != K)
    throw std::invalid_argument("w0 must be [N, K0] with K0 matching x0");
  if (params.w1.shape[1] != K1)
    throw std::invalid_argument("w1 must be [N, K1] with K1 matching x1");
  if (params.out.shape[0] != M || params.out.shape[1] != N)
    throw std::invalid_argument("out must be [M, N]");
  if (config.fused_residual)
  {
    validate_tensor(params.residual, "residual", 16);
    validate_tensor(params.actual_seqlen, "actual_seqlen", 16);
    if (params.residual.shape[0] != M || params.residual.shape[1] != N)
      throw std::invalid_argument("residual must match the [M, N] output shape");
    if (params.actual_seqlen.shape[0] <= 0 || params.i_dim <= 0 || M % params.i_dim != 0)
      throw std::invalid_argument("fused residual requires valid row lengths and I_dim");
    if (params.actual_seqlen.shape[0] != M / params.i_dim)
      throw std::invalid_argument("actual_seqlen must have one entry per output row group");
  }
  /* K0/K1/N must cover full 128-bit copy vectors. */
  if (K % 8 != 0 || K1 % 8 != 0 || N % 8 != 0)
    throw std::invalid_argument("dual-GEMM x0_x1 requires K0, K1 and N to be multiples of 8");
  if (params.x0.strides[0] < K || params.x1.strides[0] < K1 || params.out.strides[0] < N)
    throw std::invalid_argument("dual-GEMM x0_x1 row strides must cover their logical inner extents");
  if (params.x0.strides[0] % 8 != 0 || params.x1.strides[0] % 8 != 0 || params.out.strides[0] % 8 != 0)
    throw std::invalid_argument("dual-GEMM x0_x1 row strides must preserve 16-byte vector alignment");
  /* w0/w1 are compiled with a row stride equal to their K extent. */
  if (params.w0.strides[0] != K || params.w1.strides[0] != K1)
    throw std::invalid_argument("w0 and w1 must be contiguous [N, K] row-major");
  /* The compact signature is valid only when K0 == K1 == N; it shares one
   * row-stride symbol across x0, x1 and out. Any unequal extent selects the
   * wide signature with three independent strides.
   */
  if (
    config.spec.K1 == config.spec.K && config.spec.N == config.spec.K
    && (params.x1.strides[0] != params.x0.strides[0] || params.out.strides[0] != params.x0.strides[0]))
  {
    throw std::invalid_argument("compact dual-GEMM x0_x1 requires x0, x1 and out to share one row stride");
  }

  if (config.has_bias)
  {
    validate_pointer(params.bias0.data, 16, "bias0");
    validate_pointer(params.bias1.data, 16, "bias1");
    if (params.bias0.shape[0] != N || params.bias1.shape[0] != N)
      throw std::invalid_argument("bias0 and bias1 must have N elements");
  }
}

std::uint32_t ceil_div_u32(std::int32_t value, std::uint32_t divisor, char const* name)
{
  if (divisor == 0)
    throw std::invalid_argument("dual-GEMM x0_x1 launch divisor must be positive");
  std::uint64_t const extent = static_cast<std::uint64_t>(value);
  std::uint64_t const tiles = (extent + divisor - 1U) / divisor;
  if (tiles == 0 || tiles > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error(std::string(name) + " does not fit a positive uint32");
  return static_cast<std::uint32_t>(tiles);
}

void launch_sm80(
  cubin_kernel_t loaded,
  KernelConfig const& config,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  validate_operand_devices(config, params, cuda_device_for_context(context));

  KernelSpec const& spec = config.spec;
  abi::SM80Params device_params{};
  device_params.x0 = make_tensor2_s2_d1_descriptor(params.x0);
  device_params.x1 = make_tensor2_s2_d1_descriptor(params.x1);
  device_params.w0 = make_tensor2_s2_d1_descriptor(params.w0);
  device_params.w1 = make_tensor2_s2_d1_descriptor(params.w1);
  device_params.out = make_tensor2_s2_d1_descriptor(params.out);
  if (config.has_bias)
  {
    device_params.bias0 = make_tensor1_descriptor(params.bias0);
    device_params.bias1 = make_tensor1_descriptor(params.bias1);
  }
  if (config.fused_residual)
  {
    device_params.actual_seqlen = make_tensor1_descriptor(params.actual_seqlen);
    device_params.residual = make_tensor2_s2_d1_descriptor(params.residual);
    device_params.i_dim = params.i_dim;
  }
  device_params.raster_factor = static_cast<std::int32_t>(spec.raster_factor);

  void* kernel_params[abi::kSM80MaxParameterCount];
  std::size_t const parameter_count
    = abi::pack_sm80_kernel_params(&device_params, config.has_bias, config.fused_residual, kernel_params);
  std::size_t const expected_count = abi::sm80_parameter_count(config.has_bias, config.fused_residual);
  if (parameter_count != expected_count)
    throw std::logic_error("dual-GEMM x0_x1 packed an unexpected parameter count");

  cubin_launch_config_t const launch_config = abi::sm80_launch_config(
    ceil_div_u32(params.out.shape[0], spec.tile_m, "dual-GEMM x0_x1 grid.m"),
    ceil_div_u32(params.out.shape[1], spec.tile_n, "dual-GEMM x0_x1 grid.n"),
    spec.raster_factor,
    spec.num_threads,
    smem_bytes,
    reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream)));

  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(dual_gemm_x0_x1_sm80)");
}

std::uint64_t ceil_div_u64(std::uint64_t value, std::uint64_t divisor, char const* name)
{
  if (divisor == 0)
    throw std::invalid_argument(std::string(name) + " divisor must be positive");
  return (value + divisor - 1U) / divisor;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right, char const* name)
{
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
    throw std::overflow_error(std::string(name) + " overflows uint64");
  return left * right;
}

std::uint32_t checked_u32(std::uint64_t value, char const* name)
{
  if (value == 0 || value > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error(std::string(name) + " does not fit a positive uint32");
  return static_cast<std::uint32_t>(value);
}

/* Resident-weight asymmetric grid: x is the row-worker slot and y is the
 * fixed output-column tile. Two consumer warp groups cover two rows per CTA.
 */
cubin_launch_config_t make_sm90_asymmetric_launch_config(
  embedded::CubinImage const& image,
  LaunchParams const& params,
  std::uint32_t smem_bytes,
  std::int32_t multiprocessor_count)
{
  embedded::SM90LaunchInfo const& metadata = image.sm90;
  std::uint64_t const gm
    = ceil_div_u64(static_cast<std::uint64_t>(params.out.shape[0]), static_cast<std::uint64_t>(image.tile_m), "grid.m");
  std::uint64_t const gn
    = ceil_div_u64(static_cast<std::uint64_t>(params.out.shape[1]), static_cast<std::uint64_t>(image.tile_n), "grid.n");
  if (metadata.cluster_dims[0] != 1 || metadata.cluster_dims[2] != 1 || gn % metadata.cluster_dims[1] != 0)
  {
    throw std::invalid_argument(
      "dual-GEMM x0_x1 asymmetric SM90 grid requires cluster [1, y, 1] with y dividing grid.n");
  }
  std::uint64_t const per_column_cap
    = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(multiprocessor_count) / gn);
  constexpr std::uint64_t kConsumerWarpGroups = 2;
  std::uint64_t const full_workers
    = checked_multiply(kConsumerWarpGroups, per_column_cap, "asymmetric SM90 full row workers");
  std::uint64_t const target_iterations = ceil_div_u64(gm, full_workers, "asymmetric SM90 target iterations");
  std::uint64_t const row_workers_per_cta
    = checked_multiply(kConsumerWarpGroups, target_iterations, "asymmetric SM90 row workers per CTA");
  std::uint64_t const ctas_per_column = ceil_div_u64(gm, row_workers_per_cta, "asymmetric SM90 CTAs per column");

  cubin_launch_config_t launch_config{};
  launch_config.grid_x = checked_u32(ctas_per_column, "dual-GEMM x0_x1 asymmetric SM90 grid.x");
  launch_config.grid_y = checked_u32(gn, "dual-GEMM x0_x1 asymmetric SM90 grid.y");
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
  return launch_config;
}

/* Original Hopper persistent grid, flattened onto x with cluster [x, 1, 1]. */
cubin_launch_config_t make_sm90_pingpong_launch_config(
  embedded::CubinImage const& image,
  LaunchParams const& params,
  std::uint32_t smem_bytes,
  std::int32_t multiprocessor_count)
{
  embedded::SM90LaunchInfo const& metadata = image.sm90;
  if (metadata.cluster_dims[1] != 1 || metadata.cluster_dims[2] != 1)
    throw std::invalid_argument("dual-GEMM x0_x1 SM90 flattened persistent grid requires cluster dimensions [x, 1, 1]");

  std::uint64_t const gm
    = ceil_div_u64(static_cast<std::uint64_t>(params.out.shape[0]), static_cast<std::uint64_t>(image.tile_m), "grid.m");
  std::uint64_t const gn
    = ceil_div_u64(static_cast<std::uint64_t>(params.out.shape[1]), static_cast<std::uint64_t>(image.tile_n), "grid.n");
  std::uint64_t const logical_gm
    = ceil_div_u64(gm, static_cast<std::uint64_t>(metadata.cluster_dims[0]), "logical grid.m");
  std::uint64_t const logical_gn
    = ceil_div_u64(gn, static_cast<std::uint64_t>(metadata.cluster_dims[1]), "logical grid.n");
  std::uint64_t const cluster_size = checked_multiply(
    checked_multiply(
      static_cast<std::uint64_t>(metadata.cluster_dims[0]),
      static_cast<std::uint64_t>(metadata.cluster_dims[1]),
      "dual-GEMM x0_x1 SM90 cluster size"),
    static_cast<std::uint64_t>(metadata.cluster_dims[2]),
    "dual-GEMM x0_x1 SM90 cluster size");

  std::uint64_t const max_clusters = static_cast<std::uint64_t>(multiprocessor_count) / cluster_size;
  if (max_clusters == 0)
    throw std::invalid_argument("dual-GEMM x0_x1 SM90 cluster size exceeds the device's multiprocessor count");

  std::uint64_t persistent_clusters = 0;
  if (image.raster_factor > 0)
  {
    std::uint64_t const raster_factor = image.raster_factor;
    std::uint64_t const padded_gm = checked_multiply(
      ceil_div_u64(logical_gm, raster_factor, "rasterized M tiles"), raster_factor, "rasterized M tiles");
    std::uint64_t const total_iterations = checked_multiply(padded_gm, logical_gn, "SM90 total iterations");
    std::uint64_t const rounded_cap = max_clusters / raster_factor * raster_factor;
    persistent_clusters = std::min(total_iterations, std::max(rounded_cap, raster_factor));
  }
  else
  {
    persistent_clusters = std::min(checked_multiply(logical_gm, logical_gn, "SM90 tile count"), max_clusters);
  }

  cubin_launch_config_t launch_config{};
  launch_config.grid_x
    = checked_u32(checked_multiply(persistent_clusters, cluster_size, "SM90 grid.x"), "dual-GEMM x0_x1 SM90 grid.x");
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
  return launch_config;
}

cubin_launch_config_t make_sm90_launch_config(
  embedded::CubinImage const& image, LaunchParams const& params, CUcontext context, std::uint32_t smem_bytes)
{
  embedded::SM90LaunchInfo const& metadata = image.sm90;
  if (!metadata.is_native)
    throw std::invalid_argument("dual-GEMM x0_x1 SM90 CUBIN is missing its Hopper launch metadata");
  for (std::uint32_t dimension : metadata.cluster_dims)
  {
    if (dimension == 0)
      throw std::invalid_argument("dual-GEMM x0_x1 SM90 CUBIN has an invalid cluster dimension");
  }
  std::int32_t const multiprocessor_count = cuda_multiprocessor_count_for_context(context);
  if (multiprocessor_count <= 0)
    throw std::invalid_argument("current CUDA device has no active multiprocessors");

  if (
    image.cubin.launch_abi != nullptr
    && (std::strcmp(image.cubin.launch_abi, kSM90AsymmetricLaunchAbi) == 0 || std::strcmp(image.cubin.launch_abi, kSM90AsymmetricResidualLaunchAbi) == 0))
  {
    return make_sm90_asymmetric_launch_config(image, params, smem_bytes, multiprocessor_count);
  }
  return make_sm90_pingpong_launch_config(image, params, smem_bytes, multiprocessor_count);
}

void launch_sm90(
  cubin_kernel_t loaded,
  KernelConfig const& config,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  validate_operand_devices(config, params, cuda_device_for_context(context));

  embedded::CubinImage const& image = *config.embedded_image;
  embedded::SM90LaunchInfo const& metadata = image.sm90;
  CUtensorMapDataType const expected_dtype = tma_data_type(config.dtype == DType::kBFloat16);

  abi::SM90Params device_params{};
  /* x0 and x1 require distinct TMA maps and coordinates. */
  TmaTensorSource const x0_source = make_tma_tensor2_source(params.x0, false);
  TmaTensorSource const x1_source = make_tma_tensor2_source(params.x1, false);
  TmaTensorSource const w0_source = make_tma_tensor2_source(params.w0, false);
  TmaTensorSource const w1_source = make_tma_tensor2_source(params.w1, false);
  /* Output is always N-major. */
  TmaTensorSource const out_source = make_tma_tensor2_source(params.out, false);
  bool const is_asymmetric = config.cubin.launch_abi != nullptr
    && (std::strcmp(config.cubin.launch_abi, kSM90AsymmetricLaunchAbi) == 0
        || std::strcmp(config.cubin.launch_abi, kSM90AsymmetricResidualLaunchAbi) == 0);
  /* The asymmetric kernel multicasts the activation tiles over its N cluster
   * and keeps the resident weights and the output per-CTA. A unit cluster
   * leaves every operand on the plain load atom.
   */
  std::uint32_t const x_multicast = is_asymmetric ? metadata.cluster_dims[1] : 1;
  encode_tma_descriptor(
    device_params.x0_tma, multicast_tma_descriptor(metadata.x0, x_multicast, "x0"), expected_dtype, x0_source, "x0");
  encode_tma_descriptor(
    device_params.x1_tma, multicast_tma_descriptor(metadata.x1, x_multicast, "x1"), expected_dtype, x1_source, "x1");
  encode_tma_descriptor(device_params.w0_tma, metadata.w0, expected_dtype, w0_source, "w0");
  encode_tma_descriptor(device_params.w1_tma, metadata.w1, expected_dtype, w1_source, "w1");
  encode_tma_descriptor(device_params.output_tma, metadata.output, expected_dtype, out_source, "out");
  if (is_asymmetric)
  {
    AsymmetricTmaAtomKind const x_kind
      = x_multicast > 1 ? AsymmetricTmaAtomKind::kMulticastLoad : AsymmetricTmaAtomKind::kLoad;
    finalize_asymmetric_tma_atom(device_params.x0_tma, x_kind);
    finalize_asymmetric_tma_atom(device_params.x1_tma, x_kind);
    finalize_asymmetric_tma_atom(device_params.w0_tma, AsymmetricTmaAtomKind::kLoad);
    finalize_asymmetric_tma_atom(device_params.w1_tma, AsymmetricTmaAtomKind::kLoad);
    finalize_asymmetric_tma_atom(device_params.output_tma, AsymmetricTmaAtomKind::kStore);
  }

  device_params.x0_coord = make_tensor2_s2_coord(params.x0);
  device_params.x1_coord = make_tensor2_s2_coord(params.x1);
  device_params.w0_coord = make_tensor2_s2_coord(params.w0);
  device_params.w1_coord = make_tensor2_s2_coord(params.w1);
  device_params.output_coord = make_tensor2_s2_coord(params.out);
  if (config.has_bias)
  {
    device_params.bias0 = make_tensor1_descriptor(params.bias0);
    device_params.bias1 = make_tensor1_descriptor(params.bias1);
  }
  if (config.fused_residual)
  {
    device_params.actual_seqlen = make_tensor1_descriptor(params.actual_seqlen);
    device_params.residual = make_tensor2_s2_d1_descriptor(params.residual);
  }
  device_params.i_dim = config.fused_residual ? params.i_dim : kSM90UnusedIDim;
  device_params.tiled_mma = 0;

  void* kernel_params[abi::kSM90MaxParameterCount]{};
  std::size_t const parameter_count
    = abi::pack_sm90_kernel_params(&device_params, config.has_bias, config.fused_residual, kernel_params);
  if (parameter_count != abi::sm90_parameter_count(config.has_bias, config.fused_residual))
    throw std::logic_error("dual-GEMM x0_x1 SM90 parameter packer produced the wrong ABI count");

  cubin_launch_config_t const launch_config = make_sm90_launch_config(image, params, context, smem_bytes);
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(dual_gemm_x0_x1_sm90)");
}

KernelSpec make_kernel_spec(embedded::CubinImage const& image)
{
  return KernelSpec{
    image.cubin.target_sm,
    image.cubin.kernel_sm,
    image.K,
    image.K1,
    image.N,
    image.bucket,
    image.has_bias,
    image.fused_residual,
    image.tile_m,
    image.tile_n,
    image.num_threads,
    image.raster_factor,
  };
}

/* Select the nearest S anchor; ties choose the lower anchor. */
embedded::CubinImage const& find_embedded_cubin(
  std::int32_t target_sm,
  std::int32_t K,
  std::int32_t K1,
  std::int32_t N,
  std::int32_t S,
  DType dtype,
  bool has_bias,
  bool fused_residual)
{
  if (S < 0)
    throw std::invalid_argument("dual-GEMM x0_x1 S must be non-negative");

  bool const is_bfloat16 = dtype == DType::kBFloat16;
  embedded::RegistryView const registry = embedded::registry();
  embedded::CubinImage const* nearest = nullptr;
  std::uint64_t nearest_distance = 0;
  for (std::size_t index = 0; index < registry.count; ++index)
  {
    embedded::CubinImage const& image = registry.images[index];
    if (
      !cubin_supports_sm(image.cubin, target_sm) || image.K != K || image.K1 != K1 || image.N != N
      || image.is_bfloat16 != is_bfloat16 || image.has_bias != has_bias || image.fused_residual != fused_residual)
      continue;

    std::int64_t const delta = static_cast<std::int64_t>(image.bucket) - static_cast<std::int64_t>(S);
    std::uint64_t const distance = static_cast<std::uint64_t>(delta < 0 ? -delta : delta);
    if (
      nearest == nullptr || distance < nearest_distance
      || (distance == nearest_distance && image.bucket < nearest->bucket))
    {
      nearest = &image;
      nearest_distance = distance;
    }
  }
  if (nearest != nullptr)
    return *nearest;

  throw std::invalid_argument(
    "No embedded dual-GEMM x0_x1 CUBIN for SM" + std::to_string(target_sm) + ", K0=" + std::to_string(K)
    + ", K1=" + std::to_string(K1) + ", N=" + std::to_string(N) + ", S=" + std::to_string(S)
    + ", has_bias=" + (has_bias ? "true" : "false") + ", fused_residual=" + (fused_residual ? "true" : "false"));
}

std::size_t preload_kernels(CUcontext context, std::int32_t device_sm)
{
  return preload_registry_kernels(context, device_sm, embedded::registry());
}

CubinPreloadRegistration const kPreloader{
  "dual_gemm_x0_x1",
  &preload_kernels,
};

} // namespace

std::vector<KernelSpec> kernel_specs()
{
  return map_registry(embedded::registry(), make_kernel_spec);
}

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  std::int32_t K,
  std::int32_t K1,
  std::int32_t N,
  std::int32_t S,
  DType dtype,
  bool has_bias,
  bool fused_residual)
{
  embedded::CubinImage const& image = find_embedded_cubin(target_sm, K, K1, N, S, dtype, has_bias, fused_residual);
  return KernelConfig{
    make_kernel_spec(image),
    dtype,
    has_bias,
    fused_residual,
    image.cubin,
    &image,
  };
}

void launch(KernelConfig const& config, LaunchParams const& params)
{
  validate_launch(config, params);
  CUcontext const context = current_cuda_context();
  std::int32_t const device_sm = cuda_sm_for_context(context);
  std::int32_t const configured_sm = config.spec.target_sm;
  if (device_sm != configured_sm || !cubin_supports_sm(config.cubin, device_sm))
  {
    throw std::invalid_argument(
      "dual-GEMM x0_x1 config selects SM" + std::to_string(configured_sm) + " but current device is SM"
      + std::to_string(device_sm));
  }

  cubin_kernel_t const loaded = load_embedded_kernel(context, config.cubin);
  std::uint32_t const smem_bytes = dynamic_smem_bytes(config);
  if (config.spec.kernel_sm == 90)
    launch_sm90(loaded, config, params, context, smem_bytes);
  else
    launch_sm80(loaded, config, params, context, smem_bytes);
}

} // namespace bioir::cutedsl::dual_gemm_x0_x1
