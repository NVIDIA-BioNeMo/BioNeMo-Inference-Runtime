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

/* Shared host-side launch helpers for CuTeDSL CUBIN families.
 *
 * Validation, descriptor packing, TMA encoding, and integer helpers used by
 * more than one family launcher. Family-specific device ABIs stay in each
 * family's launcher.h.
 */

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_LAUNCH_UTILS_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_LAUNCH_UTILS_H_

#include "cubin_runtime.h"
#include "cutedsl_tensor_abi.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace bioir::cutedsl
{

inline void validate_pointer(std::uint64_t pointer, std::uint64_t alignment, char const* name)
{
  if (pointer == 0)
    throw std::invalid_argument(std::string(name) + " pointer is null");
  if (pointer % alignment != 0)
  {
    throw std::invalid_argument(
      std::string(name) + " pointer does not satisfy " + std::to_string(alignment) + "-byte alignment");
  }
}

template <typename ViewT>
void validate_tensor(ViewT const& tensor, char const* name, std::uint64_t alignment)
{
  validate_pointer(tensor.data, alignment, name);
  for (std::int32_t extent : tensor.shape)
  {
    if (extent <= 0)
      throw std::invalid_argument(std::string(name) + " has a non-positive extent");
  }
  for (std::int64_t stride : tensor.strides)
  {
    if (stride <= 0)
      throw std::invalid_argument(std::string(name) + " has a non-positive stride");
  }
}

inline void validate_static_tail(Tensor4View const& tensor, std::int32_t expected, char const* name)
{
  if (tensor.shape[3] != expected)
  {
    throw std::invalid_argument(
      std::string(name) + " static tail must be " + std::to_string(expected) + ", got "
      + std::to_string(tensor.shape[3]));
  }
}

inline bool leading_shape_matches(Tensor3View const& lhs, Tensor4View const& rhs)
{
  for (std::size_t index = 0; index < lhs.shape.size(); ++index)
  {
    if (lhs.shape[index] != rhs.shape[index])
      return false;
  }
  return true;
}

template <std::size_t N>
inline cute_tensor_s3_d2_t make_tensor3_descriptor(TensorView<N> const& view)
{
  static_assert(N >= 3, "an s3_d2 descriptor needs three host extents");
  cute_tensor_s3_d2_t descriptor{};
  descriptor.data = static_cast<CUdeviceptr>(view.data);
  for (std::size_t index = 0; index < 3; ++index)
    descriptor.dynamic_shapes[index] = view.shape[index];
  for (std::size_t index = 0; index < 2; ++index)
    descriptor.dynamic_strides[index] = view.strides[index];
  return descriptor;
}

inline cute_tensor_s4_d3_t make_tensor4_descriptor(Tensor4View const& view)
{
  cute_tensor_s4_d3_t descriptor{};
  descriptor.data = static_cast<CUdeviceptr>(view.data);
  for (std::size_t index = 0; index < view.shape.size(); ++index)
    descriptor.dynamic_shapes[index] = view.shape[index];
  for (std::size_t index = 0; index < view.strides.size(); ++index)
    descriptor.dynamic_strides[index] = view.strides[index];
  return descriptor;
}

inline cute_tensor_s3_d3_t make_tensor4_s3_d3_descriptor(Tensor4View const& view)
{
  cute_tensor_s3_d3_t descriptor{};
  descriptor.data = static_cast<CUdeviceptr>(view.data);
  for (std::size_t index = 0; index < view.strides.size(); ++index)
  {
    descriptor.dynamic_shapes[index] = view.shape[index];
    descriptor.dynamic_strides[index] = view.strides[index];
  }
  return descriptor;
}

inline cute_tensor_s1_d0_t make_tensor1_descriptor(Tensor1View const& view)
{
  cute_tensor_s1_d0_t descriptor{};
  descriptor.data = static_cast<CUdeviceptr>(view.data);
  descriptor.dynamic_shapes[0] = view.shape[0];
  return descriptor;
}

/* Rank-2 operand whose inner extent the kernel compiled as a constant, so only
 * the row count survives into the descriptor. The caller still supplies both
 * extents; the launcher checks the static one against its own spec first.
 */
inline cute_tensor_s1_d1_t make_tensor2_s1_d1_descriptor(Tensor2View const& view)
{
  cute_tensor_s1_d1_t descriptor{};
  descriptor.data = static_cast<CUdeviceptr>(view.data);
  descriptor.dynamic_shapes[0] = view.shape[0];
  descriptor.dynamic_strides[0] = view.strides[0];
  return descriptor;
}

/* Rank-2 operand with both extents dynamic. */
inline cute_tensor_s2_d1_t make_tensor2_s2_d1_descriptor(Tensor2View const& view)
{
  cute_tensor_s2_d1_t descriptor{};
  descriptor.data = static_cast<CUdeviceptr>(view.data);
  for (std::size_t index = 0; index < view.shape.size(); ++index)
    descriptor.dynamic_shapes[index] = view.shape[index];
  descriptor.dynamic_strides[0] = view.strides[0];
  return descriptor;
}

inline cute_tensor_s3_d2_t make_sm90_lse_descriptor(Tensor3View const& view)
{
  cute_tensor_s3_d2_t descriptor{};
  descriptor.data = static_cast<CUdeviceptr>(view.data);
  descriptor.dynamic_shapes[0] = view.shape[1];
  descriptor.dynamic_shapes[1] = view.shape[2];
  descriptor.dynamic_shapes[2] = view.shape[0];
  descriptor.dynamic_strides[0] = view.strides[1];
  descriptor.dynamic_strides[1] = view.strides[0];
  return descriptor;
}

inline cute_tensor_s3_d2_t make_sm100_lse_descriptor(Tensor3View const& view)
{
  cute_tensor_s3_d2_t descriptor{};
  descriptor.data = static_cast<CUdeviceptr>(view.data);
  descriptor.dynamic_shapes[0] = view.shape[1];
  descriptor.dynamic_shapes[1] = view.shape[2];
  descriptor.dynamic_shapes[2] = view.shape[0];
  descriptor.dynamic_strides[0] = view.strides[1];
  descriptor.dynamic_strides[1] = view.strides[0];
  return descriptor;
}

struct CoordTensorS1
{
  std::int32_t dynamic_shapes[1];
};

struct CoordTensorS2
{
  std::int32_t dynamic_shapes[2];
};

struct CoordTensorS3
{
  std::int32_t dynamic_shapes[3];
};

struct CoordTensorS4
{
  std::int32_t dynamic_shapes[4];
};

static_assert(sizeof(CoordTensorS1) == 4);
static_assert(alignof(CoordTensorS1) == alignof(std::int32_t));
static_assert(offsetof(CoordTensorS1, dynamic_shapes) == 0);
static_assert(sizeof(CoordTensorS2) == 8);
static_assert(alignof(CoordTensorS2) == alignof(std::int32_t));
static_assert(offsetof(CoordTensorS2, dynamic_shapes) == 0);
static_assert(sizeof(CoordTensorS3) == 12);
static_assert(alignof(CoordTensorS3) == alignof(std::int32_t));
static_assert(offsetof(CoordTensorS3, dynamic_shapes) == 0);
static_assert(sizeof(CoordTensorS4) == 16);
static_assert(alignof(CoordTensorS4) == alignof(std::int32_t));
static_assert(offsetof(CoordTensorS4, dynamic_shapes) == 0);

inline CoordTensorS1 make_tensor2_s1_coord(Tensor2View const& view)
{
  return CoordTensorS1{{view.shape[0]}};
}

inline CoordTensorS2 make_tensor2_s2_coord(Tensor2View const& view)
{
  return CoordTensorS2{{
    view.shape[0],
    view.shape[1],
  }};
}

template <std::size_t N>
inline CoordTensorS3 make_sm90_tensor3_coord(TensorView<N> const& view)
{
  static_assert(N >= 3, "an s3 coordinate needs three host extents");
  return CoordTensorS3{{
    view.shape[1],
    view.shape[2],
    view.shape[0],
  }};
}

inline CoordTensorS4 make_sm90_bias_coord(Tensor4View const& view)
{
  return CoordTensorS4{{
    view.shape[2],
    view.shape[3],
    view.shape[1],
    view.shape[0],
  }};
}

struct TmaTensorSource
{
  std::uint64_t data;
  std::array<std::uint64_t, 4> dimensions;
  std::array<std::uint64_t, 4> strides;
};

inline TmaTensorSource make_tma_tensor2_source(Tensor2View const& view, bool is_column_major)
{
  std::uint64_t const dynamic_stride = static_cast<std::uint64_t>(view.strides[0]);
  return TmaTensorSource{
    view.data,
    {
      static_cast<std::uint64_t>(view.shape[0]),
      static_cast<std::uint64_t>(view.shape[1]),
      0,
      0,
    },
    is_column_major ? std::array<std::uint64_t, 4>{1, dynamic_stride, 0, 0}
                    : std::array<std::uint64_t, 4>{dynamic_stride, 1, 0, 0},
  };
}

template <std::size_t N>
inline TmaTensorSource make_tma_tensor3_source(TensorView<N> const& view, std::int32_t head_dim)
{
  static_assert(N >= 3, "an s3 TMA source needs three host extents");
  return TmaTensorSource{
    view.data,
    {
      static_cast<std::uint64_t>(view.shape[0]),
      static_cast<std::uint64_t>(view.shape[1]),
      static_cast<std::uint64_t>(view.shape[2]),
      static_cast<std::uint64_t>(head_dim),
    },
    {
      static_cast<std::uint64_t>(view.strides[0]),
      static_cast<std::uint64_t>(view.strides[1]),
      static_cast<std::uint64_t>(head_dim),
      1,
    },
  };
}

inline TmaTensorSource make_tma_bias_source(Tensor4View const& view)
{
  return TmaTensorSource{
    view.data,
    {
      static_cast<std::uint64_t>(view.shape[0]),
      static_cast<std::uint64_t>(view.shape[1]),
      static_cast<std::uint64_t>(view.shape[2]),
      static_cast<std::uint64_t>(view.shape[3]),
    },
    {
      static_cast<std::uint64_t>(view.strides[0]),
      static_cast<std::uint64_t>(view.strides[1]),
      static_cast<std::uint64_t>(view.strides[2]),
      1,
    },
  };
}

inline CUtensorMapDataType tma_data_type(bool is_bfloat16)
{
  return is_bfloat16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
}

inline void encode_tma_descriptor(
  CUtensorMap& descriptor,
  TmaDescriptorInfo const& info,
  CUtensorMapDataType expected_dtype,
  TmaTensorSource const& source,
  char const* name)
{
  constexpr std::uint32_t kSourceRank = 4;
  constexpr std::uint64_t kTmaAddressAlignment = 16;
  constexpr std::uint64_t kTmaStrideAlignment = 16;
  constexpr std::uint64_t kMaxTmaGlobalDimension = std::uint64_t{1} << 32;
  constexpr std::uint64_t kMaxTmaGlobalStride = std::uint64_t{1} << 40;
  static_assert(kSourceRank <= kTmaMaxRank);
  if (info.rank == 0 || info.rank > kSourceRank)
    throw std::invalid_argument("CUBIN has an unsupported TMA rank");
  if (info.data_type != expected_dtype)
    throw std::invalid_argument("CUBIN has a mismatched TMA data type");
  validate_pointer(source.data, kTmaAddressAlignment, name);

  std::array<std::uint64_t, kTmaMaxRank> global_dimensions{};
  std::array<std::uint64_t, kTmaMaxRank - 1> global_strides{};
  std::array<bool, kSourceRank> seen_dimensions{};
  std::uint64_t const item_size = 2;
  for (std::uint32_t index = 0; index < info.rank; ++index)
  {
    std::uint32_t const source_index = info.global_dim_order[index];
    if (source_index >= info.rank || seen_dimensions[source_index])
      throw std::invalid_argument("CUBIN has an invalid TMA dimension order");
    seen_dimensions[source_index] = true;
    global_dimensions[index] = source.dimensions[source_index];
    if (global_dimensions[index] == 0 || info.box_dims[index] == 0 || info.element_strides[index] == 0)
      throw std::invalid_argument("CUBIN has an invalid TMA extent");
    if (global_dimensions[index] > kMaxTmaGlobalDimension)
      throw std::invalid_argument(std::string(name) + " TMA dimension exceeds the CUDA limit");
    if (info.box_dims[index] > 256 || info.element_strides[index] > 8)
      throw std::invalid_argument("CUBIN has out-of-range TMA traversal metadata");

    std::uint64_t const element_stride = source.strides[source_index];
    if (index == 0)
    {
      if (element_stride != 1)
        throw std::invalid_argument("TMA innermost dimension must be contiguous");
      continue;
    }
    if (element_stride > std::numeric_limits<std::uint64_t>::max() / item_size)
      throw std::overflow_error("TMA byte stride overflow");
    std::uint64_t const byte_stride = element_stride * item_size;
    if (byte_stride % kTmaStrideAlignment != 0 || byte_stride >= kMaxTmaGlobalStride)
    {
      throw std::invalid_argument(
        std::string(name) + " TMA byte strides must be 16-byte aligned and smaller than 2^40");
    }
    global_strides[index - 1] = byte_stride;
  }

  check_cuda_driver(
    cuTensorMapEncodeTiled(
      &descriptor,
      info.data_type,
      info.rank,
      reinterpret_cast<void*>(static_cast<std::uintptr_t>(source.data)),
      global_dimensions.data(),
      global_strides.data(),
      info.box_dims,
      info.element_strides,
      info.interleave,
      info.swizzle,
      info.l2_promotion,
      info.oob_fill),
    "cuTensorMapEncodeTiled");
}

/* A multicast load splits the tile across the cluster: every CTA fetches an
 * equal share of the outer box extent and multicasts it to its peers, so the
 * descriptor box is that much shorter than the tile the metadata records.
 */
inline TmaDescriptorInfo multicast_tma_descriptor(TmaDescriptorInfo info, std::uint32_t num_multicast, char const* name)
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

/* CuTe DSL 4.5.2 passes each Hopper TMA operand of a non-executable atom as the
 * encoded tensor map in the low 64 bytes of the 128-byte parameter slot. Its
 * host wrapper tags byte 8 of every atom, and byte 10 of every atom, the store
 * included, unless the atom's tensor is static and holds fewer than 2^16
 * elements. The upper half is never read. The lowering does not document this
 * rule, so a new shape needs its parameter bank compared against the source
 * launch. Captured against the CuTe host launcher for transition_mlp_sm90_v1
 * and dual_gemm_x_x_sm90_resident_mask_ptr_v1, so a compiler encoding change
 * requires new launch ABIs.
 */
inline bool sm90_static_tma_operation_tag(std::int64_t static_elements)
{
  return static_elements >= (std::int64_t{1} << 16);
}

/* Elements from a TMA source's first to one past its last addressed element. */
inline std::uint64_t tma_source_span(TmaTensorSource const& source)
{
  std::uint64_t span = 1;
  for (std::size_t index = 0; index < source.dimensions.size(); ++index)
  {
    if (source.dimensions[index] > 0)
      span += (source.dimensions[index] - 1) * source.strides[index];
  }
  return span;
}

/* The byte-10 rule above applies at launch to dynamic tensors too: the host wrapper tags an atom
 * once its tensor spans 2^16 elements, a strided view counting its span.
 */
inline bool sm90_tma_operation_tag(TmaTensorSource const& source)
{
  return sm90_static_tma_operation_tag(static_cast<std::int64_t>(tma_source_span(source)));
}

inline void finalize_sm90_tma_atom(CUtensorMap& descriptor, bool operation_tag)
{
  constexpr std::size_t kAtomTagOffset = 8;
  constexpr std::size_t kOperationTagOffset = 10;
  constexpr std::size_t kAtomPayloadBytes = 64;
  constexpr std::uint8_t kNonExecutableAtom = 0x02U;
  constexpr std::uint8_t kOperationTag = 0x20U;
  static_assert(sizeof(CUtensorMap) == 128);

  auto* bytes = reinterpret_cast<std::uint8_t*>(&descriptor);
  bytes[kAtomTagOffset] |= kNonExecutableAtom;
  if (operation_tag)
    bytes[kOperationTagOffset] |= kOperationTag;
  std::fill(bytes + kAtomPayloadBytes, bytes + sizeof(CUtensorMap), 0U);
}

/* CuTe DSL 4.5.2 passes each Blackwell TMA operand as a by-value
 * non-executable atom: the encoded tensor map in the low 64 bytes of the
 * 128-byte parameter slot, tagged at byte 8. Plain, multicast, and 2-SM loads
 * and the store all carry that one tag; unlike Hopper's asymmetric ABI there
 * is no multicast tag. The upper half is never read. Captured against the
 * CuTe host launcher for the dual-GEMM Blackwell launch ABIs, so a compiler
 * encoding change requires new launch ABIs.
 */
inline void finalize_sm100_tma_atom(CUtensorMap& descriptor)
{
  constexpr std::size_t kAtomTagOffset = 8;
  constexpr std::size_t kAtomPayloadBytes = 64;
  constexpr std::uint8_t kNonExecutableAtom = 0x02U;
  static_assert(sizeof(CUtensorMap) == 128);

  auto* bytes = reinterpret_cast<std::uint8_t*>(&descriptor);
  bytes[kAtomTagOffset] |= kNonExecutableAtom;
  std::fill(bytes + kAtomPayloadBytes, bytes + sizeof(CUtensorMap), 0U);
}

inline std::uint64_t ceil_div(std::uint64_t value, std::uint64_t divisor)
{
  if (divisor == 0)
    throw std::invalid_argument("launch divisor must be positive");
  return value / divisor + (value % divisor != 0 ? 1U : 0U);
}

inline std::uint32_t checked_u32(std::uint64_t value, char const* name)
{
  if (value == 0 || value > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error(std::string(name) + " does not fit a positive uint32");
  return static_cast<std::uint32_t>(value);
}

/* Products and sums of non-negative element counts, rejected instead of wrapping past int64. */
inline std::int64_t checked_mul(std::int64_t left, std::int64_t right, char const* name)
{
  if (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left)
    throw std::invalid_argument(std::string(name) + " overflows int64");
  return left * right;
}

inline std::int64_t checked_add(std::int64_t left, std::int64_t right, char const* name)
{
  if (right > std::numeric_limits<std::int64_t>::max() - left)
    throw std::invalid_argument(std::string(name) + " overflows int64");
  return left + right;
}

inline void validate_flat_tensor(FlatTensorView const& view, char const* name, std::uint64_t alignment)
{
  validate_pointer(view.data, alignment, name);
  if (view.extent <= 0)
    throw std::invalid_argument(std::string(name) + " has a non-positive extent");
}

} // namespace bioir::cutedsl

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_LAUNCH_UTILS_H_ */
