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

/* Stable C representations of CuTeDSL dynamic tensor descriptors, plus
 * reusable C++ host views for constructing them.
 *
 * A raw CuTeDSL CUBIN receives these descriptors by value.  The descriptor
 * shape is determined by the symbolic fake tensor used during compilation:
 * ``sN`` is the number of dynamic shape values and ``dN`` is the number of
 * dynamic stride values.  Select the exact descriptor declared in the CUBIN
 * manifest; descriptors with the same byte size are not interchangeable.
 */

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_TENSOR_ABI_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_TENSOR_ABI_H_

#include <cuda.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <array>
#include <cstddef>
#include <cstdint>

extern "C"
{
#define CUTE_ABI_ASSERT(condition, message) static_assert(condition, message)
#define CUTE_ABI_ALIGNOF(type) alignof(type)
#else
#define CUTE_ABI_ASSERT(condition, message) _Static_assert(condition, message)
#define CUTE_ABI_ALIGNOF(type) _Alignof(type)
#endif

  /* Fully static tensor: only the global-memory pointer is dynamic. */
  typedef struct
  {
    CUdeviceptr data;
  } cute_tensor_s0_d0_t;

  /* Static shape with one dynamic outer stride. */
  typedef struct
  {
    CUdeviceptr data;
    int64_t dynamic_strides[1];
  } cute_tensor_s0_d1_t;

  /* One dynamic extent and fully static strides. */
  typedef struct
  {
    CUdeviceptr data;
    int32_t dynamic_shapes[1];
  } cute_tensor_s1_d0_t;

  /* One dynamic extent and one dynamic outer stride. */
  typedef struct
  {
    CUdeviceptr data;
    int32_t dynamic_shapes[1];
    int64_t dynamic_strides[1];
  } cute_tensor_s1_d1_t;

  /* Two dynamic extents and one dynamic outer stride. */
  typedef struct
  {
    CUdeviceptr data;
    int32_t dynamic_shapes[2];
    int64_t dynamic_strides[1];
  } cute_tensor_s2_d1_t;

  /* Two dynamic extents and three independently dynamic outer strides. */
  typedef struct
  {
    CUdeviceptr data;
    int32_t dynamic_shapes[2];
    int64_t dynamic_strides[3];
  } cute_tensor_s2_d3_t;

  /* Three dynamic extents and two dynamic outer strides. */
  typedef struct
  {
    CUdeviceptr data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
  } cute_tensor_s3_d2_t;

  /* Three dynamic extents and three independently dynamic outer strides. */
  typedef struct
  {
    CUdeviceptr data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[3];
  } cute_tensor_s3_d3_t;

  /* Four dynamic extents and three dynamic outer strides. */
  typedef struct
  {
    CUdeviceptr data;
    int32_t dynamic_shapes[4];
    int64_t dynamic_strides[3];
  } cute_tensor_s4_d3_t;

  CUTE_ABI_ASSERT(sizeof(CUdeviceptr) == 8, "CuTeDSL ABI requires a 64-bit CUdeviceptr");
  CUTE_ABI_ASSERT(sizeof(cute_tensor_s0_d0_t) == 8, "unexpected CuTe tensor s0_d0 ABI");
  CUTE_ABI_ASSERT(sizeof(cute_tensor_s0_d1_t) == 16, "unexpected CuTe tensor s0_d1 ABI");
  CUTE_ABI_ASSERT(sizeof(cute_tensor_s1_d0_t) == 16, "unexpected CuTe tensor s1_d0 ABI");
  CUTE_ABI_ASSERT(sizeof(cute_tensor_s1_d1_t) == 24, "unexpected CuTe tensor s1_d1 ABI");
  CUTE_ABI_ASSERT(offsetof(cute_tensor_s1_d1_t, dynamic_strides) == 16, "unexpected CuTe tensor s1_d1 stride offset");
  CUTE_ABI_ASSERT(sizeof(cute_tensor_s2_d1_t) == 24, "unexpected CuTe tensor s2_d1 ABI");
  CUTE_ABI_ASSERT(offsetof(cute_tensor_s2_d1_t, dynamic_strides) == 16, "unexpected CuTe tensor s2_d1 stride offset");
  CUTE_ABI_ASSERT(sizeof(cute_tensor_s2_d3_t) == 40, "unexpected CuTe tensor s2_d3 ABI");
  CUTE_ABI_ASSERT(offsetof(cute_tensor_s2_d3_t, dynamic_strides) == 16, "unexpected CuTe tensor s2_d3 stride offset");
  CUTE_ABI_ASSERT(sizeof(cute_tensor_s3_d2_t) == 40, "unexpected CuTe tensor s3_d2 ABI");
  CUTE_ABI_ASSERT(offsetof(cute_tensor_s3_d2_t, dynamic_strides) == 24, "unexpected CuTe tensor s3_d2 stride offset");
  CUTE_ABI_ASSERT(sizeof(cute_tensor_s3_d3_t) == 48, "unexpected CuTe tensor s3_d3 ABI");
  CUTE_ABI_ASSERT(CUTE_ABI_ALIGNOF(cute_tensor_s3_d3_t) == 8, "unexpected CuTe tensor s3_d3 alignment");
  CUTE_ABI_ASSERT(offsetof(cute_tensor_s3_d3_t, dynamic_strides) == 24, "unexpected CuTe tensor s3_d3 stride offset");
  CUTE_ABI_ASSERT(sizeof(cute_tensor_s4_d3_t) == 48, "unexpected CuTe tensor s4_d3 ABI");
  CUTE_ABI_ASSERT(offsetof(cute_tensor_s4_d3_t, dynamic_strides) == 24, "unexpected CuTe tensor s4_d3 stride offset");

#undef CUTE_ABI_ASSERT
#undef CUTE_ABI_ALIGNOF

#ifdef __cplusplus
} /* extern "C" */

namespace bioir::cutedsl
{

inline constexpr std::int32_t kUnknownDevice = -1;

/* Host-side view of one rank-``N`` operand.
 *
 * Holds ``N - 1`` strides in the compiled descriptor's dynamic-stride order.
 * Rank-3/4 views omit their contiguous final stride, while a rank-2 view's
 * single value may represent dimension 0 or 1. The ``sN_dM`` descriptors above
 * vary independently: their ``M`` counts the strides the compiled kernel left
 * dynamic, not the operand's rank.
 *
 * ``device`` records the owning CUDA device ordinal, or ``kUnknownDevice`` when
 * the tensor is not on a GPU, for a cross-device check the launchers do not
 * perform yet: a launch targets the current context, where a pointer from
 * another device faults or corrupts silently. It stays out of the C descriptors
 * above, whose layout the compiled kernel signature fixes.
 */
template <std::size_t N>
struct TensorView
{
  static_assert(N >= 1, "a tensor view needs at least one dimension");

  std::uint64_t data{};
  std::array<std::int32_t, N> shape{};
  std::array<std::int64_t, N - 1> strides{};
  std::int32_t device{kUnknownDevice};

  TensorView() = default;

  TensorView(
    std::uint64_t data_,
    std::array<std::int32_t, N> shape_,
    std::array<std::int64_t, N - 1> strides_,
    std::int32_t device_ = kUnknownDevice)
    : data(data_)
    , shape(shape_)
    , strides(strides_)
    , device(device_)
  {
  }
};

using Tensor1View = TensorView<1>;
using Tensor2View = TensorView<2>;
using Tensor3View = TensorView<3>;
using Tensor4View = TensorView<4>;

} // namespace bioir::cutedsl
#endif

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_TENSOR_ABI_H_ */
