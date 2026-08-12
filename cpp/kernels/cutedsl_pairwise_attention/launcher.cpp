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

#include "cubins/embedded_cubins.h"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtbnm::cutedsl::pairwise_attention
{
namespace
{

constexpr char kSM80LaunchAbi[] = "pairwise_attention_sm80";
constexpr char kSM90LaunchAbi[] = "pairwise_attention_sm90";

void validate_launch(KernelConfig const& config, LaunchParams const& params)
{
  if (config.cubin.data == nullptr || config.cubin.size == 0)
    throw std::invalid_argument("embedded CUBIN image must not be empty");
  if (config.cubin.variant_id == nullptr || config.cubin.variant_id[0] == '\0')
    throw std::invalid_argument("embedded CUBIN variant_id must not be empty");
  if (config.cubin.kernel_symbol == nullptr || config.cubin.kernel_symbol[0] == '\0')
    throw std::invalid_argument("kernel_symbol must not be empty");
  bool const is_sm80 = std::holds_alternative<KernelSpecSM80>(config.spec);
  bool const is_sm90 = std::holds_alternative<KernelSpecSM90>(config.spec);
  if (!is_sm80 && !is_sm90)
    throw std::invalid_argument("pairwise-attention config has no direct launch ABI");

  char const* expected_abi = is_sm80 ? kSM80LaunchAbi : kSM90LaunchAbi;
  std::int32_t const expected_kernel_sm = is_sm80 ? 80 : 90;
  if (
    config.cubin.kernel_sm != expected_kernel_sm || config.cubin.launch_abi == nullptr
    || std::strcmp(config.cubin.launch_abi, expected_abi) != 0)
  {
    throw std::invalid_argument("pairwise-attention CUBIN has an incompatible launch ABI");
  }
  if (is_sm90)
  {
    if (config.embedded_image == nullptr || !config.embedded_image->sm90.enabled)
      throw std::invalid_argument("native SM90 pairwise-attention CUBIN has no host launch metadata");
    embedded::SM90LaunchInfo const& sm90 = config.embedded_image->sm90;
    for (std::uint32_t dimension : sm90.block_dims)
    {
      if (dimension == 0)
        throw std::invalid_argument("native SM90 pairwise-attention CUBIN has an invalid block dimension");
    }
    for (std::uint32_t dimension : sm90.cluster_dims)
    {
      if (dimension == 0)
        throw std::invalid_argument("native SM90 pairwise-attention CUBIN has an invalid cluster dimension");
    }
  }
  std::int32_t const configured_sm = spec_target_sm(config.spec);
  if (!cubin_supports_sm(config.cubin, configured_sm))
  {
    throw std::invalid_argument("embedded CUBIN does not support configured device SM" + std::to_string(configured_sm));
  }
  if (params.mult <= 0)
    throw std::invalid_argument("mult must be positive");
  if (!std::isfinite(params.softmax_scale) || params.softmax_scale <= 0.0F)
    throw std::invalid_argument("softmax_scale must be finite and positive");
  if (!std::isfinite(params.softmax_scale_log2) || params.softmax_scale_log2 <= 0.0F)
    throw std::invalid_argument("softmax_scale_log2 must be finite and positive");

  validate_tensor(params.q, "q", 16);
  validate_tensor(params.k, "k", 16);
  validate_tensor(params.v, "v", 16);
  validate_tensor(params.output, "output", 16);
  validate_tensor(params.lse, "lse", 4);
  validate_pointer(params.actual_s_kv.data, 4, "actual_s_kv");
  validate_tensor(params.bias, "bias", 16);

  std::int32_t const batch_flat = params.q.shape[0];
  std::int32_t const seqlen_q = params.q.shape[1];
  std::int32_t const num_heads = params.q.shape[2];
  std::int32_t const seqlen_kv = params.k.shape[1];
  if (batch_flat % params.mult != 0)
    throw std::invalid_argument("q.shape[0] must be divisible by mult");
  std::int32_t const batch = batch_flat / params.mult;

  if (params.k.shape[0] != batch_flat || params.v.shape != params.k.shape)
    throw std::invalid_argument("k/v shapes must agree and use q.shape[0]");
  if (params.k.shape[2] != num_heads)
    throw std::invalid_argument("q, k, and v must have the same number of heads");
  if (params.output.shape != params.q.shape)
    throw std::invalid_argument("output shape must match q shape");
  if (params.lse.shape != params.q.shape)
    throw std::invalid_argument("lse dynamic shape must match q [B*mult, Sq, H]");
  if (params.actual_s_kv.shape[0] != batch)
    throw std::invalid_argument("actual_s_kv shape must equal q.shape[0] / mult");
  if (
    params.bias.shape[0] != batch || params.bias.shape[1] != num_heads || params.bias.shape[2] != seqlen_q
    || params.bias.shape[3] < seqlen_kv)
  {
    throw std::invalid_argument("bias shape must be [B, H, Sq, Sk_padded]");
  }
  if (params.bias.shape[3] % 8 != 0)
    throw std::invalid_argument("bias.shape[3] must be padded to a multiple of 8");
}

cubin_launch_config_t make_sm90_launch_config(
  KernelSpecSM90 const& spec,
  embedded::SM90LaunchInfo const& metadata,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  if (!spec.persistent)
    throw std::invalid_argument("non-persistent native SM90 pairwise attention is not supported");

  std::uint64_t const m_tiles
    = ceil_div(static_cast<std::uint64_t>(params.q.shape[1]), static_cast<std::uint64_t>(spec.tile_m));
  std::uint64_t const heads = static_cast<std::uint64_t>(params.q.shape[2]);
  std::uint64_t const batch_flat = static_cast<std::uint64_t>(params.q.shape[0]);
  std::uint64_t total_blocks = m_tiles * heads * batch_flat;
  std::uint64_t active_sms = static_cast<std::uint64_t>(cuda_multiprocessor_count_for_context(context));
  if (spec.raster_factor > 0)
  {
    std::uint64_t const raster_factor = spec.raster_factor;
    if (active_sms >= raster_factor)
      active_sms = active_sms / raster_factor * raster_factor;
    total_blocks = ceil_div(m_tiles, raster_factor) * raster_factor * heads * batch_flat;
  }
  std::uint64_t const grid_x = std::min(active_sms, total_blocks);

  cubin_launch_config_t launch_config{};
  launch_config.grid_x = checked_u32(grid_x, "pairwise-attention SM90 grid.x");
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

/* Reject operands that do not live on the device the launch targets.
 *
 * Each descriptor carries a bare address, so the kernel cannot tell a foreign
 * pointer from a local one and instead faults or reads whatever occupies that
 * address on the target device. On SM90 the TMA encoding consumes the address
 * before the launch, so the same mistake corrupts the descriptor itself.
 */
void validate_operand_devices(LaunchParams const& params, std::int32_t device)
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

  check(params.q.device, "q");
  check(params.k.device, "k");
  check(params.v.device, "v");
  check(params.actual_s_kv.device, "actual_s_kv");
  check(params.bias.device, "bias");
  check(params.output.device, "output");
  check(params.lse.device, "lse");
}

void launch_sm80(
  cubin_kernel_t loaded,
  KernelSpecSM80 const& spec,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  validate_operand_devices(params, cuda_device_for_context(context));

  abi::SM80Params device_params{};
  device_params.q = make_tensor3_descriptor(params.q);
  device_params.k = make_tensor3_descriptor(params.k);
  device_params.v = make_tensor3_descriptor(params.v);
  device_params.actual_s_kv = make_tensor1_descriptor(params.actual_s_kv);
  device_params.bias = make_tensor4_descriptor(params.bias);
  device_params.output = make_tensor3_descriptor(params.output);
  device_params.lse = make_tensor3_descriptor(params.lse);
  device_params.softmax_scale = params.softmax_scale;
  device_params.softmax_scale_log2 = params.softmax_scale_log2;
  device_params.mult = params.mult;
  device_params.seqlen_q = params.q.shape[1];
  device_params.seqlen_k = params.k.shape[1];

  void* kernel_params[abi::kSM80ParameterCount];
  abi::pack_sm80_kernel_params(&device_params, kernel_params);
  cubin_launch_config_t const launch_config = abi::sm80_launch_config(
    static_cast<std::uint32_t>(device_params.seqlen_q),
    spec.tile_m,
    static_cast<std::uint32_t>(params.q.shape[0]),
    static_cast<std::uint32_t>(params.q.shape[2]),
    spec.num_threads,
    smem_bytes,
    reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream)));

  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr),
    "launch_cubin_kernel(pairwise_attention_sm80)");
}

void launch_sm90(
  cubin_kernel_t loaded,
  KernelSpecSM90 const& spec,
  embedded::SM90LaunchInfo const& metadata,
  DType dtype,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  validate_operand_devices(params, cuda_device_for_context(context));

  if (metadata.block_dims[0] != spec.num_threads)
    throw std::invalid_argument("native SM90 block metadata disagrees with the kernel spec");

  abi::SM90Params device_params{};
  CUtensorMapDataType const expected_dtype = tma_data_type(dtype == DType::kBFloat16);
  TmaTensorSource const q_source = make_tma_tensor3_source(params.q, spec.head_dim);
  TmaTensorSource const k_source = make_tma_tensor3_source(params.k, spec.head_dim);
  TmaTensorSource const v_source = make_tma_tensor3_source(params.v, spec.head_dim);
  TmaTensorSource const bias_source = make_tma_bias_source(params.bias);
  TmaTensorSource const output_source = make_tma_tensor3_source(params.output, spec.head_dim);
  encode_tma_descriptor(device_params.q_tma, metadata.q, expected_dtype, q_source, "q");
  encode_tma_descriptor(device_params.k_tma, metadata.k, expected_dtype, k_source, "k");
  encode_tma_descriptor(device_params.v_tma, metadata.v, expected_dtype, v_source, "v");
  encode_tma_descriptor(device_params.bias_tma, metadata.bias, expected_dtype, bias_source, "bias");
  encode_tma_descriptor(device_params.output_tma, metadata.output, expected_dtype, output_source, "output");

  device_params.q_coord = make_sm90_tensor3_coord(params.q);
  device_params.k_coord = make_sm90_tensor3_coord(params.k);
  device_params.v_coord = make_sm90_tensor3_coord(params.v);
  device_params.bias_coord = make_sm90_bias_coord(params.bias);
  device_params.actual_s_kv = make_tensor1_descriptor(params.actual_s_kv);
  device_params.output_coord = make_sm90_tensor3_coord(params.output);
  device_params.lse = make_sm90_lse_descriptor(params.lse);
  device_params.softmax_scale_log2 = params.softmax_scale_log2;
  device_params.softmax_scale = params.softmax_scale;
  device_params.mult = params.mult;
  device_params.num_heads = params.q.shape[2];
  std::uint64_t const scheduler_m_tiles
    = ceil_div(static_cast<std::uint64_t>(params.q.shape[1]), static_cast<std::uint64_t>(spec.tile_m));
  if (scheduler_m_tiles > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::overflow_error("pairwise-attention SM90 scheduler M tile count overflow");
  device_params.scheduler_m_tiles = static_cast<std::int32_t>(scheduler_m_tiles);
  device_params.scheduler_heads = params.q.shape[2];
  device_params.scheduler_batch_flat = params.q.shape[0];

  void* kernel_params[abi::kSM90ParameterCount];
  abi::pack_sm90_kernel_params(&device_params, kernel_params);
  cubin_launch_config_t const launch_config = make_sm90_launch_config(spec, metadata, params, context, smem_bytes);
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr),
    "launch_cubin_kernel(pairwise_attention_sm90)");
}

embedded::CubinImage const&
find_embedded_cubin(std::int32_t target_sm, std::int32_t head_dim, std::int32_t S, DType dtype, bool packed_output)
{
  if (S < 0)
    throw std::invalid_argument("pairwise-attention S must be non-negative");

  bool const is_bfloat16 = dtype == DType::kBFloat16;
  embedded::CubinImage const* nearest = nullptr;
  std::uint64_t nearest_distance = 0;
  for (std::size_t index = 0; index < embedded::kCubinCount; ++index)
  {
    embedded::CubinImage const& image = embedded::kCubins[index];
    if (
      !cubin_supports_sm(image.cubin, target_sm) || image.head_dim != head_dim || image.is_bfloat16 != is_bfloat16
      || image.packed_output != packed_output)
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
    "No embedded pairwise-attention CUBIN for SM" + std::to_string(target_sm) + ", D=" + std::to_string(head_dim)
    + ", S=" + std::to_string(S));
}

std::size_t preload_kernels(CUcontext context, std::int32_t device_sm)
{
  std::size_t loaded = 0;
  for (std::size_t index = 0; index < embedded::kCubinCount; ++index)
  {
    EmbeddedCubinImage const& image = embedded::kCubins[index].cubin;
    if (!cubin_supports_sm(image, device_sm))
      continue;
    (void) load_embedded_kernel(context, image, false);
    ++loaded;
  }
  return loaded;
}

CubinPreloadRegistration const kPreloader{
  "pairwise_attention",
  &preload_kernels,
};

} // namespace

KernelConfig
make_kernel_config(std::int32_t target_sm, std::int32_t head_dim, std::int32_t S, DType dtype, bool packed_output)
{
  embedded::CubinImage const& image = find_embedded_cubin(target_sm, head_dim, S, dtype, packed_output);
  KernelSpec const& spec = find_kernel_spec(target_sm, head_dim, image.bucket);
  return KernelConfig{
    spec,
    dtype,
    packed_output,
    image.cubin,
    &image,
  };
}

void launch(KernelConfig const& config, LaunchParams const& params)
{
  validate_launch(config, params);
  CUcontext const context = current_cuda_context();
  std::int32_t const device_sm = cuda_sm_for_context(context);
  std::int32_t const configured_sm = spec_target_sm(config.spec);
  if (device_sm != configured_sm || !cubin_supports_sm(config.cubin, device_sm))
  {
    throw std::invalid_argument(
      "pairwise-attention config selects SM" + std::to_string(configured_sm) + " but current device is SM"
      + std::to_string(device_sm));
  }

  cubin_kernel_t const loaded = load_embedded_kernel(context, config.cubin);
  std::uint32_t const smem_bytes = dynamic_smem_bytes(config);

  if (auto const* spec = std::get_if<KernelSpecSM80>(&config.spec))
  {
    launch_sm80(loaded, *spec, params, context, smem_bytes);
    return;
  }

  if (auto const* spec = std::get_if<KernelSpecSM90>(&config.spec))
  {
    launch_sm90(loaded, *spec, config.embedded_image->sm90, config.dtype, params, context, smem_bytes);
    return;
  }

  throw std::invalid_argument("pairwise-attention config has no registered SM launcher");
}

} // namespace trtbnm::cutedsl::pairwise_attention
