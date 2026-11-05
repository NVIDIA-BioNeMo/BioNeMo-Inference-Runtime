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
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace bioir::cutedsl::attn_epilogue
{
namespace
{

constexpr char kSM80LaunchAbi[] = "attn_epilogue_sm80_v1";
constexpr char kSM80TiledLaunchAbi[] = "attn_epilogue_sm80_tiled_v1";
constexpr char kSM90LaunchAbi[] = "attn_epilogue_sm90_v1";
constexpr char kSM90StreamedLaunchAbi[] = "attn_epilogue_sm90_streamed_v1";
constexpr char kSM100LaunchAbi[] = "attn_epilogue_sm100_v1";
constexpr char kSM100TokenLaunchAbi[] = "attn_epilogue_sm100_token_v1";
/* Rows of one Blackwell CTA's tile, and CTAs per cluster of each kernel. */
constexpr std::uint64_t kSM100TileRows = 128;
constexpr std::uint32_t kSM100PairCtas = 2;
constexpr std::uint32_t kSM100TokenCtas = 6;
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
  if (spec.has_residual)
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
  char const* expected_abi = nullptr;
  if (config.cubin.kernel_sm == 90)
    expected_abi = wide ? kSM90StreamedLaunchAbi : kSM90LaunchAbi;
  else if (config.cubin.kernel_sm == 80)
    expected_abi = wide ? kSM80TiledLaunchAbi : kSM80LaunchAbi;
  else if (config.cubin.kernel_sm == 100)
    expected_abi = kSM100LaunchAbi;
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
  if (config.spec.has_residual)
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
  auto const matches_rows = [&](Tensor3View const& pair)
  { return pair.shape[0] == folded && pair.shape[1] == columns && pair.shape[2] == channels; };
  if (!matches_rows(params.d))
    throw std::invalid_argument("d must be [B*I, J, channels] matching o");
  if (config.spec.has_residual && !matches_rows(params.z))
    throw std::invalid_argument("z must be [B*I, J, channels] matching o");
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
  std::int32_t const folded = params.o.shape[0];
  std::int32_t const columns = params.o.shape[1];
  CoordTensorS2 const coord{{columns, folded}};
  encode(device_params.o_tma, metadata.o, attention_source(params.o, config.spec), "o");
  encode(device_params.g_tma, metadata.g, pair_source(params.g), "g");
  encode(device_params.w_tma, metadata.w, make_tma_tensor2_source(params.w, false), "w");
  encode(device_params.d_tma, metadata.d, pair_source(params.d), "d");
  device_params.o_coord = coord;
  device_params.g_coord = coord;
  device_params.d_coord = coord;
  if (config.spec.has_residual)
  {
    encode(device_params.z_tma, metadata.z, pair_source(params.z), "z");
    device_params.z_coord = coord;
  }
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

  KernelSpec const& spec = config.spec;
  void* kernel_params[abi::kSM90MaxParameterCount]{};
  if (
    abi::pack_sm90_kernel_params(&device_params, spec.has_bias, spec.has_output_gate, spec.has_residual, kernel_params)
    != abi::sm90_parameter_count(spec.has_bias, spec.has_output_gate, spec.has_residual))
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

void validate_sm100_metadata(embedded::CubinImage const& image, std::uint32_t cluster_ctas)
{
  embedded::SM100LaunchInfo const& metadata = image.sm100;
  if (!metadata.enabled)
    throw std::invalid_argument("attention epilogue CUBIN is missing its Blackwell launch metadata");
  if (metadata.block_dims[0] != image.num_threads || metadata.block_dims[1] != 1 || metadata.block_dims[2] != 1)
    throw std::invalid_argument("attention epilogue Blackwell block metadata disagrees with the launch geometry");
  if (metadata.cluster_dims[0] != cluster_ctas || metadata.cluster_dims[1] != 1 || metadata.cluster_dims[2] != 1)
    throw std::invalid_argument("attention epilogue CUBIN has an unsupported Blackwell cluster shape");
  if (!image.cubin.non_portable_cluster_size_allowed)
    throw std::invalid_argument("attention epilogue Blackwell CUBIN has inconsistent cluster function metadata");
}

cubin_launch_config_t make_sm100_launch_config(
  embedded::SM100LaunchInfo const& metadata, KernelConfig const& config, std::uint64_t grid_x, std::uint64_t stream)
{
  cubin_launch_config_t launch_config{};
  launch_config.grid_x = checked_u32(grid_x, "attention epilogue grid.x");
  launch_config.grid_y = 1;
  launch_config.grid_z = 1;
  launch_config.block_x = metadata.block_dims[0];
  launch_config.block_y = metadata.block_dims[1];
  launch_config.block_z = metadata.block_dims[2];
  launch_config.cluster_x = metadata.cluster_dims[0];
  launch_config.cluster_y = metadata.cluster_dims[1];
  launch_config.cluster_z = metadata.cluster_dims[2];
  launch_config.cluster_scheduling_policy = metadata.cluster_scheduling_policy;
  launch_config.dynamic_smem_bytes = config.cubin.dynamic_smem_bytes;
  launch_config.stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(stream));
  return launch_config;
}

/* o as the Blackwell triangle kernel reads it, (J, H*D, B*I): each pair's
 * heads-inner output is one contiguous H*D row, at the spec's static J stride.
 */
TmaTensorSource sm100_attention_source(Tensor4View const& view, KernelSpec const& spec)
{
  std::uint64_t const width = static_cast<std::uint64_t>(spec.heads * spec.head_dim);
  return TmaTensorSource{
    view.data,
    {static_cast<std::uint64_t>(view.shape[1]), width, static_cast<std::uint64_t>(view.shape[0]), 0},
    {width, 1, static_cast<std::uint64_t>(view.strides[0]), 0},
  };
}

void launch_sm100(cubin_kernel_t loaded, KernelConfig const& config, LaunchParams const& params, CUcontext context)
{
  validate_operand_devices(params, config.spec, cuda_device_for_context(context));
  embedded::CubinImage const& image = *config.embedded_image;
  validate_sm100_metadata(image, kSM100PairCtas);
  embedded::SM100LaunchInfo const& metadata = image.sm100;
  KernelSpec const& spec = config.spec;
  if (!spec.has_residual)
    throw std::invalid_argument("the Blackwell triangle epilogue takes a residual");
  if (spec.has_output_gate && metadata.y.rank == 0)
    throw std::invalid_argument("output-gated Blackwell attention epilogue CUBIN is missing its y tensor map");
  CUtensorMapDataType const dtype = tma_data_type(true);

  abi::SM100Params device_params{};
  auto const encode
    = [dtype](CUtensorMap& atom, TmaDescriptorInfo const& info, TmaTensorSource const& source, char const* name)
  {
    encode_tma_descriptor(atom, info, dtype, source, name);
    finalize_sm100_tma_atom(atom);
  };
  std::int32_t const folded = params.o.shape[0];
  std::int32_t const columns = params.o.shape[1];
  CoordTensorS2 const coord{{columns, folded}};
  encode(device_params.o_tma, metadata.o, sm100_attention_source(params.o, config.spec), "o");
  encode(device_params.g_tma, metadata.g, pair_source(params.g), "g");
  encode(device_params.z_tma, metadata.z, pair_source(params.z), "z");
  encode(device_params.d_tma, metadata.d, pair_source(params.d), "d");
  encode(device_params.w_tma, metadata.w, make_tma_tensor2_source(params.w, false), "w");
  device_params.o_coord = coord;
  device_params.g_coord = coord;
  device_params.z_coord = coord;
  device_params.d_coord = coord;
  if (spec.has_output_gate)
  {
    encode(device_params.y_tma, metadata.y, pair_source(params.y), "y");
    device_params.y_coord = CoordTensorS2{{columns, params.y.shape[0]}};
  }
  if (spec.has_bias)
    device_params.bias.data = static_cast<CUdeviceptr>(params.b.data);

  std::uint64_t const tiles
    = static_cast<std::uint64_t>(folded) * ceil_div(static_cast<std::uint64_t>(columns), kSM100TileRows);
  if (tiles > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::overflow_error("attention epilogue tile count does not fit int32");
  std::int32_t const multiprocessor_count = cuda_multiprocessor_count_for_context(context);
  if (multiprocessor_count <= 0)
    throw std::invalid_argument("current CUDA device has no active multiprocessors");
  /* One persistent pair per two SMs; the kernel strides pair tiles by the grid size. */
  std::uint64_t const clusters = std::max<std::uint64_t>(
    1,
    std::min<std::uint64_t>(
      ceil_div(tiles, kSM100PairCtas), static_cast<std::uint64_t>(multiprocessor_count) / kSM100PairCtas));

  void* kernel_params[abi::kSM100MaxParameterCount]{};
  if (
    abi::pack_sm100_kernel_params(&device_params, spec.has_bias, spec.has_output_gate, kernel_params)
    != abi::sm100_parameter_count(spec.has_bias, spec.has_output_gate))
    throw std::logic_error("attention epilogue SM100 parameter packer produced the wrong ABI count");
  cubin_launch_config_t const launch_config
    = make_sm100_launch_config(metadata, config, kSM100PairCtas * clusters, params.stream);
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(attn_epilogue_sm100)");
}

std::array<std::pair<Tensor2View const*, char const*>, 6> token_operands(TokenLaunchParams const& params)
{
  return {{{&params.o, "o"}, {&params.g, "g"}, {&params.w, "w"}, {&params.d, "d"}, {&params.z, "z"}, {&params.y, "y"}}};
}

void require_token_image(KernelConfig const& config)
{
  if (
    config.cubin.kernel_sm != 100 || config.spec.kernel_sm != 100 || config.cubin.launch_abi == nullptr
    || std::strcmp(config.cubin.launch_abi, kSM100TokenLaunchAbi) != 0)
  {
    throw std::invalid_argument("attention epilogue CUBIN is not a Blackwell token kernel");
  }
}

void validate_token_launch(KernelConfig const& config, TokenLaunchParams const& params)
{
  if (config.cubin.data == nullptr || config.cubin.size == 0)
    throw std::invalid_argument("embedded CUBIN image must not be empty");
  if (config.cubin.kernel_symbol == nullptr || config.cubin.kernel_symbol[0] == '\0')
    throw std::invalid_argument("kernel_symbol must not be empty");
  require_token_image(config);
  KernelSpec const& spec = config.spec;
  if (!spec.has_output_gate || !spec.has_residual)
    throw std::invalid_argument("the Blackwell token epilogue takes a residual and an output gate");

  for (auto const& [view, name] : token_operands(params))
    validate_tensor(*view, name, 16);
  std::int32_t const rows = params.g.shape[0];
  std::int32_t const width = spec.heads * spec.head_dim;
  std::int32_t const channels = spec.channels;
  if (rows <= 0 || params.o.shape[0] != rows || params.o.shape[1] != width || params.g.shape[1] != width)
    throw std::invalid_argument("o and g must be [M, heads*head_dim] row views");
  if (params.w.shape[0] != channels || params.w.shape[1] != width || params.w.strides[0] != width)
    throw std::invalid_argument("w must be a row-major [channels, heads*head_dim] weight");
  if (spec.has_bias)
  {
    validate_tensor(params.b, "b", 16);
    if (params.b.shape[0] != channels)
      throw std::invalid_argument("b must be the [channels] output-projection bias");
  }
  for (Tensor2View const* view : {&params.d, &params.z})
  {
    if (view->shape[0] != rows || view->shape[1] != channels)
      throw std::invalid_argument("d and z must be [M, channels] row views matching g");
  }
  std::int32_t const gate_rows = params.y.shape[0];
  if (
    params.columns <= 0 || params.gate_repeat <= 0 || gate_rows <= 0 || params.y.shape[1] != channels
    || gate_rows % params.columns != 0
    || static_cast<std::int64_t>(gate_rows) * params.gate_repeat != static_cast<std::int64_t>(rows))
  {
    throw std::invalid_argument("y must be [M / gate_repeat, channels] in whole samples of columns rows");
  }
  if (params.max_clusters <= 0)
    throw std::invalid_argument("the Blackwell token epilogue needs a positive co-resident cluster count");
}

void launch_sm100_token(
  cubin_kernel_t loaded, KernelConfig const& config, TokenLaunchParams const& params, CUcontext context)
{
  std::int32_t const device = cuda_device_for_context(context);
  auto const check_device = [device](std::int32_t operand_device, char const* name)
  {
    if (operand_device == kUnknownDevice)
      throw std::invalid_argument(std::string(name) + " has no CUDA device: it is a host tensor");
    if (operand_device != device)
    {
      throw std::invalid_argument(
        std::string(name) + " is on CUDA device " + std::to_string(operand_device) + " but the launch targets device "
        + std::to_string(device));
    }
  };
  for (auto const& [view, name] : token_operands(params))
    check_device(view->device, name);
  if (config.spec.has_bias)
    check_device(params.b.device, "b");
  embedded::CubinImage const& image = *config.embedded_image;
  validate_sm100_metadata(image, kSM100TokenCtas);
  embedded::SM100LaunchInfo const& metadata = image.sm100;
  CUtensorMapDataType const dtype = tma_data_type(true);

  abi::SM100TokenParams device_params{};
  auto const encode
    = [dtype](CUtensorMap& atom, TmaDescriptorInfo const& info, Tensor2View const& view, char const* name)
  {
    encode_tma_descriptor(atom, info, dtype, make_tma_tensor2_source(view, false), name);
    finalize_sm100_tma_atom(atom);
  };
  encode(device_params.o_tma, metadata.o, params.o, "o");
  encode(device_params.g_tma, metadata.g, params.g, "g");
  encode(device_params.w_tma, metadata.w, params.w, "w");
  encode(device_params.z_tma, metadata.z, params.z, "z");
  encode(device_params.d_tma, metadata.d, params.d, "d");
  std::int32_t const rows = params.g.shape[0];
  CoordTensorS1 const coord{{rows}};
  device_params.o_coord = coord;
  device_params.g_coord = coord;
  device_params.z_coord = coord;
  device_params.d_coord = coord;
  device_params.y = make_tensor2_s1_d1_descriptor(params.y);
  if (config.spec.has_bias)
    device_params.bias.data = static_cast<CUdeviceptr>(params.b.data);
  device_params.columns = params.columns;
  device_params.gate_repeat = params.gate_repeat;

  /* A step is one row tile for the triples and two for the pairs; the kernel strides steps by its cluster count. */
  std::uint64_t const steps
    = ceil_div(ceil_div(static_cast<std::uint64_t>(rows), kSM100TileRows), config.spec.pairs ? kSM100PairCtas : 1U);
  std::uint64_t const clusters
    = std::max<std::uint64_t>(1, std::min<std::uint64_t>(steps, static_cast<std::uint64_t>(params.max_clusters)));

  void* kernel_params[abi::kSM100TokenMaxParameterCount]{};
  if (
    abi::pack_sm100_token_kernel_params(&device_params, config.spec.has_bias, kernel_params)
    != abi::sm100_token_parameter_count(config.spec.has_bias))
    throw std::logic_error("attention epilogue token parameter packer produced the wrong ABI count");
  cubin_launch_config_t const launch_config
    = make_sm100_launch_config(metadata, config, kSM100TokenCtas * clusters, params.stream);
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr),
    "launch_cubin_kernel(attn_epilogue_sm100_token)");
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
  if (config.spec.has_residual)
    device_params.z = pair_descriptor(params.z);
  if (config.spec.has_output_gate)
    device_params.y = pair_descriptor(params.y);

  KernelSpec const& spec = config.spec;
  void* kernel_params[abi::kSM80MaxParameterCount]{};
  if (
    abi::pack_sm80_kernel_params(&device_params, spec.has_bias, spec.has_output_gate, spec.has_residual, kernel_params)
    != abi::sm80_parameter_count(spec.has_bias, spec.has_output_gate, spec.has_residual))
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
    image.has_residual,
    image.tile_j,
    image.tile_n,
    image.num_threads,
    image.bucket,
    image.pairs,
  };
}

embedded::CubinImage const& find_embedded_cubin(
  std::int32_t target_sm,
  std::int32_t heads,
  std::int32_t head_dim,
  std::int32_t channels,
  bool has_bias,
  bool has_output_gate,
  bool has_residual,
  std::int32_t rows,
  bool pairs)
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
      || image.has_output_gate != has_output_gate || image.has_residual != has_residual || image.pairs != pairs)
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
    + ", head_dim=" + std::to_string(head_dim) + ", channels=" + std::to_string(channels) + ", has_bias="
    + (has_bias ? "true" : "false") + ", has_output_gate=" + (has_output_gate ? "true" : "false") + ", has_residual="
    + (has_residual ? "true" : "false") + ", rows=" + std::to_string(rows) + ", pairs=" + (pairs ? "true" : "false"));
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
  std::int32_t rows,
  bool has_residual,
  bool pairs)
{
  embedded::CubinImage const& image
    = find_embedded_cubin(target_sm, heads, head_dim, channels, has_bias, has_output_gate, has_residual, rows, pairs);
  return KernelConfig{make_kernel_spec(image), image.cubin, &image};
}

namespace
{

CUcontext context_for(KernelConfig const& config)
{
  CUcontext const context = current_cuda_context();
  std::int32_t const device_sm = cuda_sm_for_context(context);
  if (device_sm != config.spec.target_sm || !cubin_supports_sm(config.cubin, device_sm))
  {
    throw std::invalid_argument(
      "attention epilogue config selects SM" + std::to_string(config.spec.target_sm) + " but current device is SM"
      + std::to_string(device_sm));
  }
  return context;
}

} // namespace

void launch(KernelConfig const& config, LaunchParams const& params)
{
  validate_launch(config, params);
  CUcontext const context = context_for(config);
  cubin_kernel_t const loaded = load_embedded_kernel(context, config.cubin);
  if (config.cubin.kernel_sm == 100)
    launch_sm100(loaded, config, params, context);
  else if (config.cubin.kernel_sm == 90)
    launch_sm90(loaded, config, params, context);
  else
    launch_sm80(loaded, config, params, context);
}

std::int32_t max_active_clusters(KernelConfig const& config)
{
  require_token_image(config);
  CUcontext const context = context_for(config);
  embedded::CubinImage const& image = *config.embedded_image;
  validate_sm100_metadata(image, kSM100TokenCtas);
  cubin_kernel_t const loaded = load_embedded_kernel(context, config.cubin);
  embedded::SM100LaunchInfo const& metadata = image.sm100;
  CUlaunchAttribute attribute{};
  attribute.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attribute.value.clusterDim.x = metadata.cluster_dims[0];
  attribute.value.clusterDim.y = metadata.cluster_dims[1];
  attribute.value.clusterDim.z = metadata.cluster_dims[2];
  CUlaunchConfig launch_config{};
  launch_config.gridDimX = metadata.cluster_dims[0];
  launch_config.gridDimY = metadata.cluster_dims[1];
  launch_config.gridDimZ = metadata.cluster_dims[2];
  launch_config.blockDimX = metadata.block_dims[0];
  launch_config.blockDimY = metadata.block_dims[1];
  launch_config.blockDimZ = metadata.block_dims[2];
  launch_config.sharedMemBytes = config.cubin.dynamic_smem_bytes;
  launch_config.attrs = &attribute;
  launch_config.numAttrs = 1;
  int clusters = 0;
  check_cuda_driver(
    cuOccupancyMaxActiveClusters(&clusters, loaded.function, &launch_config),
    "cuOccupancyMaxActiveClusters(attn_epilogue_sm100_token)");
  return clusters;
}

void launch_token(KernelConfig const& config, TokenLaunchParams const& params)
{
  validate_token_launch(config, params);
  CUcontext const context = context_for(config);
  launch_sm100_token(load_embedded_kernel(context, config.cubin), config, params, context);
}

} // namespace bioir::cutedsl::attn_epilogue
