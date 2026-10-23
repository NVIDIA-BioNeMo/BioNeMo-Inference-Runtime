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

/* Windowed-attention CUBIN configuration, device ABIs, and launcher interface.
 */

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_WINDOW_ATTENTION_LAUNCHER_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_WINDOW_ATTENTION_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace bioir::cutedsl::window_attention::embedded
{
struct CubinImage;
}

/* Both parameter banks were read from EIATTR_KPARAM_INFO. They are the
 * device-kernel ABIs, not the high-level CuTeDSL callable signature. Every
 * stride is a 64-bit element count, so no offset wraps past 2^31 elements.
 */
namespace bioir::cutedsl::window_attention::abi
{

/* The warp-MMA kernel takes q, k, v and the output as descriptors with dynamic
 * (S, N, H) and sample and row strides, the bias with dynamic (B, T, H) and
 * batch, window, head and query strides, then the softmax scale in log2 and
 * natural units.
 */
inline constexpr std::size_t kSM80ParameterCount = 7;

struct SM80Params
{
  cute_tensor_s3_d2_t q;
  cute_tensor_s3_d2_t k;
  cute_tensor_s3_d2_t v;
  cute_tensor_s3_d4_t bias;
  cute_tensor_s3_d2_t output;
  float softmax_scale_log2;
  float softmax_scale;
};

inline void pack_sm80_kernel_params(SM80Params* params, void* kernel_params[kSM80ParameterCount])
{
  kernel_params[0] = &params->q;
  kernel_params[1] = &params->k;
  kernel_params[2] = &params->v;
  kernel_params[3] = &params->bias;
  kernel_params[4] = &params->output;
  kernel_params[5] = &params->softmax_scale_log2;
  kernel_params[6] = &params->softmax_scale;
}

static_assert(kSM80ParameterCount == 7);
static_assert(std::is_standard_layout_v<SM80Params>);
static_assert(offsetof(SM80Params, q) == 0);
static_assert(offsetof(SM80Params, k) == 40);
static_assert(offsetof(SM80Params, v) == 80);
static_assert(offsetof(SM80Params, bias) == 120);
static_assert(offsetof(SM80Params, output) == 176);
static_assert(offsetof(SM80Params, softmax_scale_log2) == 216);
static_assert(offsetof(SM80Params, softmax_scale) == 220);
static_assert(sizeof(SM80Params) == 224);

/* The cp.async + WGMMA kernel takes only the pointers and strides its loads
 * read, in that order: q and k with their sample and row strides, the row
 * count N, the samples-per-batch-item divisor, the bias with its batch,
 * window, head and query strides, v and the output with their sample and row
 * strides, and the softmax scale. Its two tiled-MMA arguments lower to one-byte
 * accumulate flags the kernel never reads.
 */
inline constexpr std::size_t kSM90ParameterCount = 22;

/* CuTe's FastDivmodDivisor: the quotient is
 * (umulhi(n, multiplier) + ((n - umulhi(n, multiplier)) >> shift_right_1)) >> shift_right_2.
 */
struct FastDivmod
{
  std::int32_t divisor;
  std::uint32_t multiplier;
  std::uint8_t shift_right_1;
  std::uint8_t shift_right_2;
  std::uint8_t reserved[2]{};
};

struct SM90Params
{
  CUdeviceptr q;
  CUdeviceptr k;
  std::int64_t q_sample_stride;
  std::int64_t q_row_stride;
  std::int64_t k_sample_stride;
  std::int64_t k_row_stride;
  std::int32_t rows;
  FastDivmod samples_per_item;
  CUdeviceptr bias;
  std::int64_t bias_batch_stride;
  std::int64_t bias_window_stride;
  std::int64_t bias_head_stride;
  std::int64_t bias_query_stride;
  CUdeviceptr v;
  std::int64_t v_sample_stride;
  std::int64_t v_row_stride;
  CUdeviceptr output;
  std::int64_t output_sample_stride;
  std::int64_t output_row_stride;
  float softmax_scale;
  std::uint8_t qk_tiled_mma;
  std::uint8_t pv_tiled_mma;
};

inline void pack_sm90_kernel_params(SM90Params* params, void* kernel_params[kSM90ParameterCount])
{
  kernel_params[0] = &params->q;
  kernel_params[1] = &params->k;
  kernel_params[2] = &params->q_sample_stride;
  kernel_params[3] = &params->q_row_stride;
  kernel_params[4] = &params->k_sample_stride;
  kernel_params[5] = &params->k_row_stride;
  kernel_params[6] = &params->rows;
  kernel_params[7] = &params->samples_per_item;
  kernel_params[8] = &params->bias;
  kernel_params[9] = &params->bias_batch_stride;
  kernel_params[10] = &params->bias_window_stride;
  kernel_params[11] = &params->bias_head_stride;
  kernel_params[12] = &params->bias_query_stride;
  kernel_params[13] = &params->v;
  kernel_params[14] = &params->v_sample_stride;
  kernel_params[15] = &params->v_row_stride;
  kernel_params[16] = &params->output;
  kernel_params[17] = &params->output_sample_stride;
  kernel_params[18] = &params->output_row_stride;
  kernel_params[19] = &params->softmax_scale;
  kernel_params[20] = &params->qk_tiled_mma;
  kernel_params[21] = &params->pv_tiled_mma;
}

static_assert(kSM90ParameterCount == 22);
static_assert(sizeof(FastDivmod) == 12);
static_assert(alignof(FastDivmod) == 4);
static_assert(offsetof(FastDivmod, multiplier) == 4);
static_assert(offsetof(FastDivmod, shift_right_1) == 8);
static_assert(offsetof(FastDivmod, shift_right_2) == 9);
static_assert(std::is_standard_layout_v<SM90Params>);
static_assert(offsetof(SM90Params, k) == 0x08);
static_assert(offsetof(SM90Params, q_sample_stride) == 0x10);
static_assert(offsetof(SM90Params, k_row_stride) == 0x28);
static_assert(offsetof(SM90Params, rows) == 0x30);
static_assert(offsetof(SM90Params, samples_per_item) == 0x34);
static_assert(offsetof(SM90Params, bias) == 0x40);
static_assert(offsetof(SM90Params, bias_query_stride) == 0x60);
static_assert(offsetof(SM90Params, v) == 0x68);
static_assert(offsetof(SM90Params, output) == 0x80);
static_assert(offsetof(SM90Params, output_row_stride) == 0x90);
static_assert(offsetof(SM90Params, softmax_scale) == 0x98);
static_assert(offsetof(SM90Params, qk_tiled_mma) == 0x9c);
static_assert(offsetof(SM90Params, pv_tiled_mma) == 0x9d);

} // namespace bioir::cutedsl::window_attention::abi

namespace bioir::cutedsl::window_attention
{

enum class DType : std::uint8_t
{
  kFloat16,
  kBFloat16,
};

struct KernelConfig
{
  std::int32_t target_sm;
  std::int32_t head_dim;
  std::int32_t n_queries;
  std::int32_t n_keys;
  DType dtype;
  EmbeddedCubinImage cubin;
  embedded::CubinImage const* embedded_image;
};

/* q, k, v and the output are [S, N, H, D] with sample, row and head strides;
 * the bias is [B, T, H, n_queries, n_keys] with batch, window, head and query
 * strides. Every last axis is contiguous.
 */
struct LaunchParams
{
  Tensor4View q;
  Tensor4View k;
  Tensor4View v;
  Tensor5View bias;
  Tensor4View output;
  float softmax_scale_log2{};
  float softmax_scale{};
  std::uint64_t stream{};
};

KernelConfig make_kernel_config(
  std::int32_t target_sm, std::int32_t head_dim, std::int32_t n_queries, std::int32_t n_keys, DType dtype);

std::uint32_t dynamic_smem_bytes(KernelConfig const& config);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace bioir::cutedsl::window_attention

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_WINDOW_ATTENTION_LAUNCHER_H_ */
