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

#include "attn_epilogue_registry.h"

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace bioir::cutedsl::attn_epilogue
{
namespace
{

constexpr char kSM80LaunchAbi[] = "attn_epilogue_sm80_v1";
constexpr char kSM80TiledLaunchAbi[] = "attn_epilogue_sm80_tiled_v1";
constexpr char kSM90LaunchAbi[] = "attn_epilogue_sm90_v1";
constexpr char kSM90StreamedLaunchAbi[] = "attn_epilogue_sm90_streamed_v1";
/* Output channels of the resident-weight kernels; wider layers take the
 * streamed SM90 or channel-tiled SM80 kernel. */
constexpr std::int32_t kResidentChannels = 128;

/* Bytes from the first to one past the last element the tensor can address. */
std::uint64_t span_bytes(TmaTensorSource const& source, std::uint32_t rank)
{
  std::uint64_t last = 0;
  for (std::uint32_t index = 0; index < rank; ++index)
    last += (source.dimensions[index] - 1U) * source.strides[index];
  return (last + 1U) * 2U;
}

void finalize_tma_atom(CUtensorMap& descriptor, TmaTensorSource const& source, std::uint32_t rank)
{
  /* CuTe DSL 4.5.2 lowers each by-value non-executable TMA CopyAtom into a
   * 64-byte Hopper atom payload carried in a 128-byte kernel parameter slot.
   * cuTensorMapEncodeTiled returns the standalone tensor-map form, which
   * differs only in two tags; add them and clear the unused upper half.
   *
   * Byte 8 marks every non-executable atom. Byte 10 carries a second tag
   * whose meaning CuTe does not publish; its host wrapper sets it exactly
   * when the tensor spans at least 128 KiB, whatever the operand or copy
   * direction. Both are part of launch ABI attn_epilogue_sm90_v1; a
   * compiler encoding change needs a new one.
   */
  constexpr std::size_t kAtomTagOffset = 8;
  constexpr std::size_t kSpanTagOffset = 10;
  constexpr std::size_t kAtomPayloadBytes = 64;
  constexpr std::uint8_t kNonExecutableAtom = 0x02U;
  constexpr std::uint8_t kSpanTag = 0x20U;
  constexpr std::uint64_t kSpanTagBytes = std::uint64_t{1} << 17;

  auto* bytes = reinterpret_cast<std::uint8_t*>(&descriptor);
  bytes[kAtomTagOffset] |= kNonExecutableAtom;
  if (span_bytes(source, rank) >= kSpanTagBytes)
    bytes[kSpanTagOffset] |= kSpanTag;
  std::fill(bytes + kAtomPayloadBytes, bytes + sizeof(CUtensorMap), 0U);
}

/* Bare device pointers require explicit cross-device validation. */
void validate_operand_devices(LaunchParams const& params, KernelSpec const& spec, std::int32_t device)
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
  check(params.o.device, "o");
  check(params.g.device, "g");
  check(params.w.device, "w");
  check(params.d.device, "d");
  check(params.z.device, "z");
  if (spec.has_bias)
    check(params.b.device, "b");
  if (spec.has_output_gate)
    check(params.y.device, "y");
}

void validate_launch(KernelConfig const& config, LaunchParams const& params)
{
  if (config.cubin.data == nullptr || config.cubin.size == 0)
    throw std::invalid_argument("embedded CUBIN image must not be empty");
  if (config.cubin.kernel_symbol == nullptr || config.cubin.kernel_symbol[0] == '\0')
    throw std::invalid_argument("kernel_symbol must not be empty");
  if (
    config.spec.tile_j == 0 || config.spec.num_threads == 0 || config.spec.tile_n == 0 || config.spec.channels <= 0
    || static_cast<std::uint32_t>(config.spec.channels) % config.spec.tile_n != 0)
  {
    throw std::invalid_argument("attention epilogue CUBIN has invalid launch geometry");
  }
  /* Only kResidentChannels leave room for a resident Wo. Wider layers stream
   * Wo on SM90 and tile the channels over grid.z on SM80, even when one tile
   * covers every channel. */
  bool const wide = config.spec.channels > kResidentChannels;
  char const* const expected_abi = config.cubin.kernel_sm == 90
    ? (wide ? kSM90StreamedLaunchAbi : kSM90LaunchAbi)
    : (config.cubin.kernel_sm == 80 ? (wide ? kSM80TiledLaunchAbi : kSM80LaunchAbi) : nullptr);
  if (
    expected_abi == nullptr || config.cubin.launch_abi == nullptr
    || std::strcmp(config.cubin.launch_abi, expected_abi) != 0 || config.spec.kernel_sm != config.cubin.kernel_sm)
  {
    throw std::invalid_argument("attention epilogue CUBIN has an incompatible launch ABI");
  }

  validate_tensor(params.o, "o", 16);
  validate_tensor(params.g, "g", 16);
  validate_tensor(params.w, "w", 16);
  validate_tensor(params.d, "d", 16);
  validate_tensor(params.z, "z", 16);

  std::int32_t const folded = params.o.shape[0];
  std::int32_t const columns = params.o.shape[1];
  std::int32_t const width = config.spec.heads * config.spec.head_dim;
  std::int32_t const channels = config.spec.channels;
  if (params.o.shape[2] != config.spec.heads || params.o.shape[3] != config.spec.head_dim)
    throw std::invalid_argument("o must be [B*I, J, heads, head_dim] for the selected CUBIN");
  if (
    (columns > 1 && params.o.strides[1] != width)
    || (config.spec.heads > 1 && params.o.strides[2] != config.spec.head_dim))
  {
    throw std::invalid_argument("o must be heads-inner with contiguous [J, heads, head_dim] rows");
  }
  if (params.g.shape[0] != folded || params.g.shape[1] != columns || params.g.shape[2] != width)
    throw std::invalid_argument("g must be [B*I, J, heads*head_dim] matching o");
  if (params.w.shape[0] != channels || params.w.shape[1] != width || params.w.strides[0] != width)
    throw std::invalid_argument("w must be a row-major [channels, heads*head_dim] weight");
  for (Tensor3View const* pair : {&params.d, &params.z})
  {
    if (pair->shape[0] != folded || pair->shape[1] != columns || pair->shape[2] != channels)
      throw std::invalid_argument("d and z must be [B*I, J, channels] matching o");
  }
  if (config.spec.has_bias)
  {
    validate_tensor(params.b, "b", 16);
    if (params.b.shape[0] != channels)
      throw std::invalid_argument("b must be the [channels] output-projection bias");
  }
  if (config.spec.has_output_gate)
  {
    validate_tensor(params.y, "y", 16);
    std::int32_t const gate_rows = params.y.shape[0];
    if (gate_rows <= 0 || folded % gate_rows != 0 || params.y.shape[1] != columns || params.y.shape[2] != channels)
      throw std::invalid_argument("y must be [B*I / mult, J, channels] matching o");
  }
  /* One pair row leaves the J and B*I modes both at extent 1, which builds a
   * TMA descriptor the kernel traps on.
   */
  if (static_cast<std::int64_t>(folded) * columns <= 1)
    throw std::invalid_argument("attention epilogue needs more than one pair row");
}

/* Each source lists the kernel's modes: o (J, D, H, B*I), g (J, H*D, B*I),
 * w (C, H*D), and d/z (J, C, B*I); the recorded dimension order maps them to
 * tensor-map dimensions.
 *
 * o is compiled heads-inner with static J and H strides, so those come from
 * the spec: an extent-1 mode may carry any stride in the view.
 */
TmaTensorSource attention_source(Tensor4View const& view, KernelSpec const& spec)
{
  return TmaTensorSource{
    view.data,
    {static_cast<std::uint64_t>(view.shape[1]),
     static_cast<std::uint64_t>(view.shape[3]),
     static_cast<std::uint64_t>(view.shape[2]),
     static_cast<std::uint64_t>(view.shape[0])},
    {static_cast<std::uint64_t>(spec.heads * spec.head_dim),
     1,
     static_cast<std::uint64_t>(spec.head_dim),
     static_cast<std::uint64_t>(view.strides[0])},
  };
}

TmaTensorSource pair_source(Tensor3View const& view)
{
  return TmaTensorSource{
    view.data,
    {static_cast<std::uint64_t>(view.shape[1]),
     static_cast<std::uint64_t>(view.shape[2]),
     static_cast<std::uint64_t>(view.shape[0]),
     0},
    {static_cast<std::uint64_t>(view.strides[1]), 1, static_cast<std::uint64_t>(view.strides[0]), 0},
  };
}

void launch_sm90(cubin_kernel_t loaded, KernelConfig const& config, LaunchParams const& params, CUcontext context)
{
  validate_operand_devices(params, config.spec, cuda_device_for_context(context));

  embedded::CubinImage const& image = *config.embedded_image;
  embedded::SM90LaunchInfo const& metadata = image.sm90;
  if (!metadata.is_native)
    throw std::invalid_argument("attention epilogue CUBIN is missing its Hopper launch metadata");
  CUtensorMapDataType const dtype = tma_data_type(true);

  abi::SM90Params device_params{};
  auto const encode
    = [dtype](CUtensorMap& atom, TmaDescriptorInfo const& info, TmaTensorSource const& source, char const* name)
  {
    encode_tma_descriptor(atom, info, dtype, source, name);
    finalize_tma_atom(atom, source, info.rank);
  };
  encode(device_params.o_tma, metadata.o, attention_source(params.o, config.spec), "o");
  encode(device_params.g_tma, metadata.g, pair_source(params.g), "g");
  encode(device_params.w_tma, metadata.w, make_tma_tensor2_source(params.w, false), "w");
  encode(device_params.z_tma, metadata.z, pair_source(params.z), "z");
  encode(device_params.d_tma, metadata.d, pair_source(params.d), "d");

  std::int32_t const folded = params.o.shape[0];
  std::int32_t const columns = params.o.shape[1];
  CoordTensorS2 const coord{{columns, folded}};
  device_params.o_coord = coord;
  device_params.g_coord = coord;
  device_params.z_coord = coord;
  device_params.d_coord = coord;
  if (config.spec.has_output_gate)
  {
    encode(device_params.y_tma, metadata.y, pair_source(params.y), "y");
    device_params.y_coord = CoordTensorS2{{columns, params.y.shape[0]}};
  }

  std::uint64_t const gj = ceil_div(static_cast<std::uint64_t>(columns), image.tile_j);
  std::uint64_t const n_tiles = static_cast<std::uint64_t>(config.spec.channels) / config.spec.tile_n;
  std::uint64_t const total_tiles = gj * static_cast<std::uint64_t>(folded) * n_tiles;
  if (total_tiles > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::overflow_error("attention epilogue tile count does not fit int32");
  device_params.gj = static_cast<std::int32_t>(gj);
  device_params.total_tiles = static_cast<std::int32_t>(total_tiles);
  device_params.tiled_mma = 0;
  if (config.spec.has_bias)
    device_params.bias.data = static_cast<CUdeviceptr>(params.b.data);

  void* kernel_params[abi::kSM90MaxParameterCount]{};
  if (
    abi::pack_sm90_kernel_params(&device_params, config.spec.has_bias, config.spec.has_output_gate, kernel_params)
    != abi::sm90_parameter_count(config.spec.has_bias, config.spec.has_output_gate))
    throw std::logic_error("attention epilogue SM90 parameter packer produced the wrong ABI count");

  std::int32_t const multiprocessor_count = cuda_multiprocessor_count_for_context(context);
  if (multiprocessor_count <= 0)
    throw std::invalid_argument("current CUDA device has no active multiprocessors");
  /* One persistent CTA per SM; the kernel strides tiles by the grid size. */
  cubin_launch_config_t launch_config{};
  launch_config.grid_x = checked_u32(
    std::min<std::uint64_t>(total_tiles, static_cast<std::uint64_t>(multiprocessor_count)),
    "attention epilogue grid.x");
  launch_config.grid_y = 1;
  launch_config.grid_z = 1;
  launch_config.block_x = metadata.block_dims[0];
  launch_config.block_y = metadata.block_dims[1];
  launch_config.block_z = metadata.block_dims[2];
  launch_config.dynamic_smem_bytes = config.cubin.dynamic_smem_bytes;
  launch_config.stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream));
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(attn_epilogue_sm90)");
}

cute_tensor_s2_d2_t pair_descriptor(Tensor3View const& view)
{
  cute_tensor_s2_d2_t descriptor{};
  descriptor.data = static_cast<CUdeviceptr>(view.data);
  descriptor.dynamic_shapes[0] = view.shape[1];
  descriptor.dynamic_shapes[1] = view.shape[0];
  descriptor.dynamic_strides[0] = view.strides[1];
  descriptor.dynamic_strides[1] = view.strides[0];
  return descriptor;
}

void launch_sm80(cubin_kernel_t loaded, KernelConfig const& config, LaunchParams const& params, CUcontext context)
{
  validate_operand_devices(params, config.spec, cuda_device_for_context(context));

  abi::SM80Params device_params{};
  device_params.o.data = static_cast<CUdeviceptr>(params.o.data);
  device_params.o.dynamic_shapes[0] = params.o.shape[1];
  device_params.o.dynamic_shapes[1] = params.o.shape[0];
  device_params.o.dynamic_strides[0] = params.o.strides[0];
  device_params.g = pair_descriptor(params.g);
  device_params.w.data = static_cast<CUdeviceptr>(params.w.data);
  if (config.spec.has_bias)
    device_params.bias.data = static_cast<CUdeviceptr>(params.b.data);
  device_params.d = pair_descriptor(params.d);
  device_params.z = pair_descriptor(params.z);
  if (config.spec.has_output_gate)
    device_params.y = pair_descriptor(params.y);

  void* kernel_params[abi::kSM80MaxParameterCount]{};
  if (
    abi::pack_sm80_kernel_params(&device_params, config.spec.has_bias, config.spec.has_output_gate, kernel_params)
    != abi::sm80_parameter_count(config.spec.has_bias, config.spec.has_output_gate))
    throw std::logic_error("attention epilogue SM80 parameter packer produced the wrong ABI count");

  std::uint32_t const folded = static_cast<std::uint32_t>(params.o.shape[0]);
  if (folded > 65535U)
    throw std::invalid_argument("attention epilogue SM80 grid.y cannot hold more than 65535 pair rows");
  cubin_launch_config_t launch_config{};
  launch_config.grid_x = checked_u32(
    ceil_div(static_cast<std::uint64_t>(params.o.shape[1]), config.spec.tile_j), "attention epilogue grid.x");
  launch_config.grid_y = folded;
  launch_config.grid_z = static_cast<std::uint32_t>(config.spec.channels) / config.spec.tile_n;
  launch_config.block_x = config.spec.num_threads;
  launch_config.block_y = 1;
  launch_config.block_z = 1;
  launch_config.dynamic_smem_bytes = config.cubin.dynamic_smem_bytes;
  launch_config.stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream));
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(attn_epilogue_sm80)");
}

KernelSpec make_kernel_spec(embedded::CubinImage const& image)
{
  return KernelSpec{
    image.cubin.target_sm,
    image.cubin.kernel_sm,
    image.heads,
    image.head_dim,
    image.channels,
    image.has_bias,
    image.has_output_gate,
    image.tile_j,
    image.tile_n,
    image.num_threads,
    image.bucket,
  };
}

embedded::CubinImage const& find_embedded_cubin(
  std::int32_t target_sm,
  std::int32_t heads,
  std::int32_t head_dim,
  std::int32_t channels,
  bool has_bias,
  bool has_output_gate,
  std::int32_t rows)
{
  if (rows < 0)
    throw std::invalid_argument("attention epilogue rows must be non-negative");

  embedded::RegistryView const registry = embedded::registry();
  embedded::CubinImage const* nearest = nullptr;
  std::uint64_t nearest_distance = 0;
  for (std::size_t index = 0; index < registry.count; ++index)
  {
    embedded::CubinImage const& image = registry.images[index];
    if (
      !cubin_supports_sm(image.cubin, target_sm) || image.heads != heads || image.head_dim != head_dim
      || image.channels != channels || !image.is_bfloat16 || image.has_bias != has_bias
      || image.has_output_gate != has_output_gate)
      continue;

    std::int64_t const delta = static_cast<std::int64_t>(image.bucket) - static_cast<std::int64_t>(rows);
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
    "No embedded attention epilogue CUBIN for SM" + std::to_string(target_sm) + ", heads=" + std::to_string(heads)
    + ", head_dim=" + std::to_string(head_dim) + ", channels=" + std::to_string(channels)
    + ", has_bias=" + (has_bias ? "true" : "false") + ", has_output_gate=" + (has_output_gate ? "true" : "false")
    + ", rows=" + std::to_string(rows));
}

std::size_t preload_kernels(CUcontext context, std::int32_t device_sm)
{
  return preload_registry_kernels(context, device_sm, embedded::registry());
}

CubinPreloadRegistration const kPreloader{
  "attn_epilogue",
  &preload_kernels,
};

} // namespace

std::vector<KernelSpec> kernel_specs()
{
  return map_registry(embedded::registry(), make_kernel_spec);
}

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  std::int32_t heads,
  std::int32_t head_dim,
  std::int32_t channels,
  bool has_bias,
  bool has_output_gate,
  std::int32_t rows)
{
  embedded::CubinImage const& image
    = find_embedded_cubin(target_sm, heads, head_dim, channels, has_bias, has_output_gate, rows);
  return KernelConfig{make_kernel_spec(image), image.cubin, &image};
}

void launch(KernelConfig const& config, LaunchParams const& params)
{
  validate_launch(config, params);
  CUcontext const context = current_cuda_context();
  std::int32_t const device_sm = cuda_sm_for_context(context);
  if (device_sm != config.spec.target_sm || !cubin_supports_sm(config.cubin, device_sm))
  {
    throw std::invalid_argument(
      "attention epilogue config selects SM" + std::to_string(config.spec.target_sm) + " but current device is SM"
      + std::to_string(device_sm));
  }
  cubin_kernel_t const loaded = load_embedded_kernel(context, config.cubin);
  if (config.cubin.kernel_sm == 90)
    launch_sm90(loaded, config, params, context);
  else
    launch_sm80(loaded, config, params, context);
}

} // namespace bioir::cutedsl::attn_epilogue
