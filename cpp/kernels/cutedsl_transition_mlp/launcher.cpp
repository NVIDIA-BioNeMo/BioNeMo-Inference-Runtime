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

#include "transition_mlp_registry.h"

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace bioir::cutedsl::transition_mlp
{
namespace
{

constexpr char kSM90LaunchAbi[] = "transition_mlp_sm90_v1";
constexpr char kSM80LaunchAbi[] = "transition_mlp_sm80_v1";

/* CuTe DSL 4.5.2 sets the operation tag on every atom, the store included, unless the atom's tensor
 * is static and holds fewer than 2^16 elements. Only the weights are static. The lowering does not
 * document this rule, so a new shape needs its parameter bank compared against the source launch.
 */
bool has_operation_tag(std::int64_t static_elements)
{
  return static_elements >= (std::int64_t{1} << 16);
}

void finalize_tma_atom(CUtensorMap& descriptor, bool operation_tag)
{
  /* CuTe DSL 4.5.2 lowers each by-value non-executable TMA CopyAtom into a 64-byte Hopper atom
   * payload carried in a 128-byte kernel parameter slot. cuTensorMapEncodeTiled returns the
   * standalone tensor-map form, which differs only in the atom and operation tags. Add them and
   * clear the unused upper half; every other field belongs in the encoded descriptor.
   *
   * These offsets are part of launch ABI transition_mlp_sm90_v1. The builder pins the CuTe DSL
   * toolchain, so a compiler encoding change requires a new launch ABI.
   */
  constexpr std::size_t kAtomTagOffset = 8;
  constexpr std::size_t kOperationTagOffset = 10;
  constexpr std::size_t kAtomPayloadBytes = 64;
  constexpr std::uint8_t kNonExecutableAtom = 0x02U;
  constexpr std::uint8_t kOperationTag = 0x20U;

  auto* bytes = reinterpret_cast<std::uint8_t*>(&descriptor);
  bytes[kAtomTagOffset] |= kNonExecutableAtom;
  if (operation_tag)
    bytes[kOperationTagOffset] |= kOperationTag;
  std::fill(bytes + kAtomPayloadBytes, bytes + sizeof(CUtensorMap), 0U);
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

  check(params.x.device, "x");
  check(params.w1.device, "w1");
  check(params.w2.device, "w2");
  if (config.spec.has_residual)
    check(params.residual.device, "residual");
  check(params.output.device, "output");
  if (config.spec.has_bias)
  {
    check(params.b1.device, "b1");
    check(params.b2.device, "b2");
  }
  if (config.spec.has_mask)
    check(params.mask.device, "mask");
}

/* x, the residual and the output are [rows, width] with 16-byte rows. */
void validate_row_operand(Tensor2View const& view, std::int32_t rows, std::int32_t width, char const* name)
{
  validate_tensor(view, name, 16);
  if (view.shape[0] != rows || view.shape[1] != width)
    throw std::invalid_argument(std::string(name) + " must be [rows, " + std::to_string(width) + "]");
  if (view.strides[0] < width || view.strides[0] % 8 != 0)
    throw std::invalid_argument(std::string(name) + " row stride must cover the width and keep 16-byte rows");
}

/* One block of hidden rows for ReLU, two for the SwiGLU's value and gate, three with the 3-way SwiGLU's
 * second value.
 */
std::int32_t w1_row_count(KernelSpec const& spec)
{
  return spec.hidden * (spec.is_three_way ? 3 : spec.is_silu_gate ? 2 : 1);
}

void validate_launch(KernelConfig const& config, LaunchParams const& params)
{
  if (config.cubin.data == nullptr || config.cubin.size == 0)
    throw std::invalid_argument("embedded CUBIN image must not be empty");
  if (config.cubin.variant_id == nullptr || config.cubin.variant_id[0] == '\0')
    throw std::invalid_argument("embedded CUBIN variant_id must not be empty");
  if (config.cubin.kernel_symbol == nullptr || config.cubin.kernel_symbol[0] == '\0')
    throw std::invalid_argument("kernel_symbol must not be empty");
  bool const is_sm80 = config.cubin.kernel_sm == 80;
  char const* expected_abi = is_sm80 ? kSM80LaunchAbi : kSM90LaunchAbi;
  if (
    (config.cubin.kernel_sm != 80 && config.cubin.kernel_sm != 90) || config.cubin.launch_abi == nullptr
    || std::strcmp(config.cubin.launch_abi, expected_abi) != 0)
  {
    throw std::invalid_argument("transition MLP CUBIN has an incompatible launch ABI");
  }
  if (config.embedded_image == nullptr || config.embedded_image->sm90.enabled == is_sm80)
    throw std::invalid_argument("transition MLP CUBIN has incompatible architecture launch metadata");
  if (!cubin_supports_sm(config.cubin, config.spec.target_sm))
    throw std::invalid_argument(
      "embedded CUBIN does not support configured device SM" + std::to_string(config.spec.target_sm));

  KernelSpec const& spec = config.spec;
  if (spec.width <= 0 || spec.hidden <= 0 || spec.tile_m == 0 || spec.num_threads == 0)
    throw std::invalid_argument("transition MLP CUBIN has invalid launch geometry");
  std::int32_t const rows = params.x.shape[0];
  std::int32_t const w1_rows = w1_row_count(spec);
  validate_row_operand(params.x, rows, spec.width, "x");
  if (spec.has_residual)
    validate_row_operand(params.residual, rows, spec.width, "residual");
  validate_row_operand(params.output, rows, spec.width, "output");
  /* The weights were compiled with static shapes and row strides. */
  validate_tensor(params.w1, "w1", 16);
  if (params.w1.shape[0] != w1_rows || params.w1.shape[1] != spec.width || params.w1.strides[0] != spec.width)
  {
    throw std::invalid_argument(
      "w1 must be contiguous [" + std::to_string(w1_rows) + ", " + std::to_string(spec.width) + "]");
  }
  validate_tensor(params.w2, "w2", 16);
  if (params.w2.shape[0] != spec.width || params.w2.shape[1] != spec.hidden || params.w2.strides[0] != spec.hidden)
  {
    throw std::invalid_argument(
      "w2 must be contiguous [" + std::to_string(spec.width) + ", " + std::to_string(spec.hidden) + "]");
  }
  if (spec.has_bias)
  {
    validate_tensor(params.b1, "b1", 2);
    validate_tensor(params.b2, "b2", 2);
    if (params.b1.shape[0] != w1_rows || params.b2.shape[0] != spec.width)
      throw std::invalid_argument("b1 needs one entry per w1 row and b2 one per output column");
  }
  if (spec.has_mask)
  {
    validate_tensor(params.mask, "mask", 2);
    if (params.mask.shape[0] != rows)
      throw std::invalid_argument("mask needs one entry per row");
  }
}

void launch_sm80(
  cubin_kernel_t loaded,
  KernelConfig const& config,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  validate_operand_devices(config, params, cuda_device_for_context(context));
  abi::SM80Params device_params{};
  device_params.x = make_tensor2_s1_d1_descriptor(params.x);
  device_params.w1.data = params.w1.data;
  device_params.w2.data = params.w2.data;
  if (config.spec.has_bias)
  {
    device_params.b1.data = params.b1.data;
    device_params.b2.data = params.b2.data;
  }
  if (config.spec.has_residual)
    device_params.residual = make_tensor2_s1_d1_descriptor(params.residual);
  if (config.spec.has_mask)
    device_params.mask = make_tensor1_descriptor(params.mask);
  device_params.output = make_tensor2_s1_d1_descriptor(params.output);
  void* kernel_params[abi::kSM80MaxParameterCount];
  std::size_t const count = abi::pack_sm80_kernel_params(
    &device_params, config.spec.has_residual, config.spec.has_bias, config.spec.has_mask, kernel_params);
  if (count != abi::sm80_parameter_count(config.spec.has_residual, config.spec.has_bias, config.spec.has_mask))
    throw std::invalid_argument("transition MLP packed an unexpected number of kernel parameters");

  cubin_launch_config_t launch_config{};
  launch_config.grid_x = checked_u32(
    (static_cast<std::uint64_t>(params.x.shape[0]) + config.spec.tile_m - 1) / config.spec.tile_m,
    "transition MLP grid.x");
  launch_config.grid_y = launch_config.grid_z = 1;
  launch_config.block_x = config.spec.num_threads;
  launch_config.block_y = launch_config.block_z = 1;
  launch_config.dynamic_smem_bytes = smem_bytes;
  launch_config.stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream));
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(transition_mlp_sm80)");
}

cubin_launch_config_t make_sm90_launch_config(
  embedded::CubinImage const& image,
  std::uint64_t num_tiles,
  CUcontext context,
  std::uint32_t smem_bytes,
  std::uint64_t stream)
{
  embedded::SM90LaunchInfo const& metadata = image.sm90;
  for (std::uint32_t dimension : metadata.cluster_dims)
  {
    if (dimension != 1)
      throw std::invalid_argument("transition MLP SM90 CUBIN must launch unit clusters");
  }
  std::int32_t const multiprocessor_count = cuda_multiprocessor_count_for_context(context);
  if (multiprocessor_count <= 0)
    throw std::invalid_argument("current CUDA device has no active multiprocessors");

  /* Persistent CTAs stride over the row tiles. */
  cubin_launch_config_t launch_config{};
  launch_config.grid_x = checked_u32(
    std::min<std::uint64_t>(num_tiles, static_cast<std::uint64_t>(multiprocessor_count)), "transition MLP grid.x");
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
  launch_config.stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(stream));
  return launch_config;
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
  CUtensorMapDataType const expected_dtype = tma_data_type(true);

  abi::SM90Params device_params{};
  encode_tma_descriptor(device_params.x_tma, metadata.x, expected_dtype, make_tma_tensor2_source(params.x, false), "x");
  encode_tma_descriptor(
    device_params.w1_tma, metadata.w1, expected_dtype, make_tma_tensor2_source(params.w1, false), "w1");
  encode_tma_descriptor(
    device_params.w2_tma, metadata.w2, expected_dtype, make_tma_tensor2_source(params.w2, false), "w2");
  encode_tma_descriptor(
    device_params.output_tma, metadata.output, expected_dtype, make_tma_tensor2_source(params.output, false), "output");
  std::int64_t const w1_elements = std::int64_t{w1_row_count(config.spec)} * config.spec.width;
  std::int64_t const w2_elements = std::int64_t{config.spec.width} * config.spec.hidden;
  finalize_tma_atom(device_params.x_tma, true);
  finalize_tma_atom(device_params.w1_tma, has_operation_tag(w1_elements));
  finalize_tma_atom(device_params.w2_tma, has_operation_tag(w2_elements));
  finalize_tma_atom(device_params.output_tma, true);
  if (config.spec.has_residual)
  {
    encode_tma_descriptor(
      device_params.residual_tma,
      metadata.residual,
      expected_dtype,
      make_tma_tensor2_source(params.residual, false),
      "residual");
    finalize_tma_atom(device_params.residual_tma, true);
    device_params.residual_coord = make_tensor2_s1_coord(params.residual);
  }

  device_params.x_coord = make_tensor2_s1_coord(params.x);
  device_params.output_coord = make_tensor2_s1_coord(params.output);
  if (config.spec.has_bias)
  {
    device_params.b1.data = static_cast<CUdeviceptr>(params.b1.data);
    device_params.b2.data = static_cast<CUdeviceptr>(params.b2.data);
  }
  if (config.spec.has_mask)
    device_params.mask = make_tensor1_descriptor(params.mask);
  std::uint64_t const num_tiles
    = ceil_div(static_cast<std::uint64_t>(params.x.shape[0]), static_cast<std::uint64_t>(config.spec.tile_m));
  device_params.num_tiles = static_cast<std::int32_t>(checked_u32(num_tiles, "transition MLP tile count"));

  KernelSpec const& spec = config.spec;
  void* kernel_params[abi::kSM90MaxParameterCount]{};
  std::size_t const parameter_count
    = abi::pack_sm90_kernel_params(&device_params, spec.has_residual, spec.has_bias, spec.has_mask, kernel_params);
  if (parameter_count != abi::sm90_parameter_count(spec.has_residual, spec.has_bias, spec.has_mask))
    throw std::logic_error("transition MLP SM90 parameter packer produced the wrong ABI count");

  cubin_launch_config_t const launch_config
    = make_sm90_launch_config(image, num_tiles, context, smem_bytes, params.stream);
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(transition_mlp_sm90)");
}

KernelSpec make_kernel_spec(embedded::CubinImage const& image)
{
  return KernelSpec{
    image.cubin.target_sm,
    image.cubin.kernel_sm,
    image.width,
    image.hidden,
    image.bucket,
    image.is_silu_gate,
    image.is_three_way,
    image.has_bias,
    image.has_mask,
    image.has_residual,
    image.tile_m,
    image.num_threads,
  };
}

embedded::CubinImage const& find_embedded_cubin(
  std::int32_t target_sm,
  DType dtype,
  bool is_silu_gate,
  bool is_three_way,
  bool has_bias,
  bool has_mask,
  bool has_residual,
  std::int32_t width,
  std::int32_t hidden,
  std::int32_t bucket)
{
  bool const is_bfloat16 = dtype == DType::kBFloat16;
  embedded::RegistryView const registry = embedded::registry();
  for (std::size_t index = 0; index < registry.count; ++index)
  {
    embedded::CubinImage const& image = registry.images[index];
    if (
      cubin_supports_sm(image.cubin, target_sm) && image.is_bfloat16 == is_bfloat16
      && image.is_silu_gate == is_silu_gate && image.is_three_way == is_three_way && image.has_bias == has_bias
      && image.has_mask == has_mask && image.has_residual == has_residual && image.width == width
      && image.hidden == hidden && image.bucket == bucket)
      return image;
  }
  throw std::invalid_argument(
    "No embedded transition MLP CUBIN for SM" + std::to_string(target_sm) + ", width=" + std::to_string(width)
    + ", hidden=" + std::to_string(hidden) + ", bucket=" + std::to_string(bucket)
    + ", silu_gate=" + (is_silu_gate ? "true" : "false") + ", three_way=" + (is_three_way ? "true" : "false")
    + ", has_bias=" + (has_bias ? "true" : "false") + ", has_mask=" + (has_mask ? "true" : "false")
    + ", has_residual=" + (has_residual ? "true" : "false"));
}

std::size_t preload_kernels(CUcontext context, std::int32_t device_sm)
{
  return preload_registry_kernels(context, device_sm, embedded::registry());
}

CubinPreloadRegistration const kPreloader{
  "transition_mlp",
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
  bool is_silu_gate,
  bool is_three_way,
  bool has_bias,
  bool has_mask,
  bool has_residual,
  std::int32_t width,
  std::int32_t hidden,
  std::int32_t bucket)
{
  embedded::CubinImage const& image = find_embedded_cubin(
    target_sm, dtype, is_silu_gate, is_three_way, has_bias, has_mask, has_residual, width, hidden, bucket);
  return KernelConfig{make_kernel_spec(image), dtype, image.cubin, &image};
}

void launch(KernelConfig const& config, LaunchParams const& params)
{
  validate_launch(config, params);
  CUcontext const context = current_cuda_context();
  std::int32_t const device_sm = cuda_sm_for_context(context);
  if (device_sm != config.spec.target_sm || !cubin_supports_sm(config.cubin, device_sm))
  {
    throw std::invalid_argument(
      "transition MLP config selects SM" + std::to_string(config.spec.target_sm) + " but current device is SM"
      + std::to_string(device_sm));
  }
  cubin_kernel_t const loaded = load_embedded_kernel(context, config.cubin);
  if (config.cubin.kernel_sm == 80)
    launch_sm80(loaded, config, params, context, dynamic_smem_bytes(config));
  else
    launch_sm90(loaded, config, params, context, dynamic_smem_bytes(config));
}

} // namespace bioir::cutedsl::transition_mlp
