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

/* Pairwise-attention CUBIN configuration, device ABI, and launcher interface.
 */

#ifndef TENSORRT_BIONEMO_CPP_KERNELS_CUTEDSL_PAIRWISE_ATTENTION_LAUNCHER_H_
#define TENSORRT_BIONEMO_CPP_KERNELS_CUTEDSL_PAIRWISE_ATTENTION_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_tensor_abi.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>

namespace trtbnm::cutedsl::pairwise_attention::embedded
{
struct CubinImage;
}

/* Direct CUDA Driver launch ABIs for Ampere and Hopper pairwise attention.
 * These are device-kernel ABIs, not the high-level CuTeDSL __call__ ABI, and
 * were read back from the compiled CUBINs' EIATTR_KPARAM_INFO.
 */
namespace trtbnm::cutedsl::pairwise_attention::abi
{

inline constexpr std::size_t kSM80ParameterCount = 12;
inline constexpr std::size_t kSM90ParameterCount = 21;

struct SM80Params
{
  cute_tensor_s3_d2_t q;
  cute_tensor_s3_d2_t k;
  cute_tensor_s3_d2_t v;
  cute_tensor_s1_d0_t actual_s_kv;
  cute_tensor_s4_d3_t bias;
  cute_tensor_s3_d2_t output;
  cute_tensor_s3_d2_t lse;
  float softmax_scale_log2;
  float softmax_scale;
  std::int32_t mult;
  std::int32_t seqlen_q;
  std::int32_t seqlen_k;
};

/* Both params and kernel_params must remain alive until the CUDA launch
 * call returns.
 */
inline void pack_sm80_kernel_params(SM80Params* params, void* kernel_params[kSM80ParameterCount])
{
  kernel_params[0] = &params->q;
  kernel_params[1] = &params->k;
  kernel_params[2] = &params->v;
  kernel_params[3] = &params->actual_s_kv;
  kernel_params[4] = &params->bias;
  kernel_params[5] = &params->output;
  kernel_params[6] = &params->lse;
  kernel_params[7] = &params->softmax_scale_log2;
  kernel_params[8] = &params->softmax_scale;
  kernel_params[9] = &params->mult;
  kernel_params[10] = &params->seqlen_q;
  kernel_params[11] = &params->seqlen_k;
}

inline cubin_launch_config_t sm80_launch_config(
  std::uint32_t seqlen_q,
  std::uint32_t tile_m,
  std::uint32_t batch_flat,
  std::uint32_t num_heads,
  std::uint32_t num_threads,
  std::uint32_t smem_bytes,
  CUstream stream)
{
  cubin_launch_config_t config = {0};
  config.grid_x = (seqlen_q + tile_m - 1U) / tile_m;
  config.grid_y = batch_flat;
  config.grid_z = num_heads;
  config.block_x = num_threads;
  config.block_y = 1;
  config.block_z = 1;
  config.dynamic_smem_bytes = smem_bytes;
  config.stream = stream;
  return config;
}

struct CoordTensorS3
{
  std::int32_t dynamic_shapes[3];
};

struct CoordTensorS4
{
  std::int32_t dynamic_shapes[4];
};

/* Hopper's five TMA descriptors are kernel parameters passed by value.
 * The descriptor bytes are encoded on the host for each runtime tensor.
 */
struct SM90Params
{
  std::uint8_t qk_tiled_mma;
  std::uint8_t pv_tiled_mma;
  CUtensorMap q_tma;
  CoordTensorS3 q_coord;
  CUtensorMap k_tma;
  CoordTensorS3 k_coord;
  CUtensorMap v_tma;
  CoordTensorS3 v_coord;
  CUtensorMap bias_tma;
  CoordTensorS4 bias_coord;
  cute_tensor_s1_d0_t actual_s_kv;
  CUtensorMap output_tma;
  CoordTensorS3 output_coord;
  cute_tensor_s3_d2_t lse;
  float softmax_scale_log2;
  float softmax_scale;
  std::int32_t mult;
  std::int32_t num_heads;
  std::int32_t scheduler_m_tiles;
  std::int32_t scheduler_heads;
  std::int32_t scheduler_batch_flat;
};

inline void pack_sm90_kernel_params(SM90Params* params, void* kernel_params[kSM90ParameterCount])
{
  kernel_params[0] = &params->qk_tiled_mma;
  kernel_params[1] = &params->pv_tiled_mma;
  kernel_params[2] = &params->q_tma;
  kernel_params[3] = &params->q_coord;
  kernel_params[4] = &params->k_tma;
  kernel_params[5] = &params->k_coord;
  kernel_params[6] = &params->v_tma;
  kernel_params[7] = &params->v_coord;
  kernel_params[8] = &params->bias_tma;
  kernel_params[9] = &params->bias_coord;
  kernel_params[10] = &params->actual_s_kv;
  kernel_params[11] = &params->output_tma;
  kernel_params[12] = &params->output_coord;
  kernel_params[13] = &params->lse;
  kernel_params[14] = &params->softmax_scale_log2;
  kernel_params[15] = &params->softmax_scale;
  kernel_params[16] = &params->mult;
  kernel_params[17] = &params->num_heads;
  kernel_params[18] = &params->scheduler_m_tiles;
  kernel_params[19] = &params->scheduler_heads;
  kernel_params[20] = &params->scheduler_batch_flat;
}

static_assert(sizeof(CUtensorMap) == 128);
static_assert(alignof(CUtensorMap) >= 64);
static_assert(sizeof(CoordTensorS3) == 12);
static_assert(sizeof(CoordTensorS4) == 16);

} // namespace trtbnm::cutedsl::pairwise_attention::abi

namespace trtbnm::cutedsl::pairwise_attention
{

enum class DType : std::uint8_t
{
  kFloat16,
  kBFloat16,
};

struct KernelSpecSM80
{
  std::int32_t target_sm;
  std::int32_t head_dim;
  std::int32_t bucket;

  std::uint32_t tile_m;
  std::uint32_t tile_n;
  std::uint32_t num_threads;
  std::uint32_t swizzle_b;
  bool load_bias_before_gemm;
};

struct KernelSpecSM90
{
  std::int32_t target_sm;
  std::int32_t head_dim;
  std::int32_t bucket;

  std::uint32_t tile_m;
  std::uint32_t tile_n;
  std::uint32_t num_threads;
  std::uint32_t mma_m;
  std::uint32_t mma_n;
  std::uint32_t kv_stage;
  std::uint32_t raster_factor;
  bool persistent;
};

using KernelSpec = std::variant<KernelSpecSM80, KernelSpecSM90>;

struct KernelConfig
{
  KernelSpec spec;
  DType dtype;
  bool packed_output;
  EmbeddedCubinImage cubin;
  embedded::CubinImage const* embedded_image;
};

constexpr KernelSpecSM80 make_sm80_spec(
  std::int32_t target_sm,
  std::int32_t head_dim,
  std::int32_t bucket,
  std::uint32_t tile_m,
  std::uint32_t tile_n,
  std::uint32_t num_threads,
  std::uint32_t swizzle_b = 3)
{
  return KernelSpecSM80{
    target_sm,
    head_dim,
    bucket,
    tile_m,
    tile_n,
    num_threads,
    swizzle_b,
    true,
  };
}

constexpr KernelSpecSM90 make_sm90_spec(
  std::int32_t head_dim,
  std::int32_t bucket,
  std::uint32_t mma_m,
  std::uint32_t mma_n,
  std::uint32_t kv_stage,
  std::uint32_t raster_factor)
{
  return KernelSpecSM90{
    90,
    head_dim,
    bucket,
    mma_m * 2U,
    mma_n,
    384,
    mma_m,
    mma_n,
    kv_stage,
    raster_factor,
    true,
  };
}

/* One entry for every (target SM, head dimension, S anchor) production
 * configuration in configs/pairwise_attention. Dtype and packed output select
 * distinct CUBINs but do not change launch geometry.
 */
inline std::array<KernelSpec, 24> const kKernelSpecs = {
  /* SM80 */
  make_sm80_spec(80, 32, 0, 128, 128, 128),
  make_sm80_spec(80, 32, 512, 128, 128, 128),
  make_sm80_spec(80, 48, 0, 64, 64, 128),
  make_sm80_spec(80, 48, 512, 64, 64, 128),
  make_sm80_spec(80, 64, 0, 64, 128, 128),
  make_sm80_spec(80, 64, 512, 64, 128, 128),

  /* SM86 */
  make_sm80_spec(86, 32, 0, 128, 128, 128, 4),
  make_sm80_spec(86, 32, 512, 64, 128, 128, 4),
  make_sm80_spec(86, 48, 0, 64, 128, 128, 4),
  make_sm80_spec(86, 48, 512, 128, 128, 256),
  make_sm80_spec(86, 64, 0, 64, 64, 128),
  make_sm80_spec(86, 64, 512, 128, 128, 256),

  /* SM89 */
  make_sm80_spec(89, 32, 0, 128, 64, 128),
  make_sm80_spec(89, 32, 512, 256, 128, 256),
  make_sm80_spec(89, 48, 0, 64, 256, 128, 4),
  make_sm80_spec(89, 48, 512, 128, 64, 128),
  make_sm80_spec(89, 64, 0, 128, 64, 128),
  make_sm80_spec(89, 64, 512, 128, 64, 128, 4),

  /* Native Hopper entries use host-encoded TMA descriptors. */
  make_sm90_spec(32, 0, 64, 64, 7, 1),
  make_sm90_spec(32, 512, 64, 128, 5, 1),
  make_sm90_spec(48, 0, 64, 64, 7, 1),
  make_sm90_spec(48, 512, 64, 192, 5, 2),
  make_sm90_spec(64, 0, 64, 64, 7, 1),
  make_sm90_spec(64, 512, 64, 128, 7, 0),
};

inline std::int32_t spec_target_sm(KernelSpec const& spec)
{
  return std::visit([](auto const& value) { return value.target_sm; }, spec);
}

inline std::int32_t spec_head_dim(KernelSpec const& spec)
{
  return std::visit([](auto const& value) { return value.head_dim; }, spec);
}

inline std::int32_t spec_bucket(KernelSpec const& spec)
{
  return std::visit([](auto const& value) { return value.bucket; }, spec);
}

inline KernelSpec const& find_kernel_spec(std::int32_t target_sm, std::int32_t head_dim, std::int32_t bucket)
{
  for (auto const& spec : kKernelSpecs)
  {
    if (spec_target_sm(spec) == target_sm && spec_head_dim(spec) == head_dim && spec_bucket(spec) == bucket)
    {
      return spec;
    }
  }
  throw std::invalid_argument(
    "No pairwise-attention kernel config for SM" + std::to_string(target_sm) + ", D=" + std::to_string(head_dim)
    + ", bucket=" + std::to_string(bucket));
}

inline bool supports_direct_launch(KernelSpec const& spec)
{
  return std::holds_alternative<KernelSpecSM80>(spec) || std::holds_alternative<KernelSpecSM90>(spec);
}

inline KernelSpecSM80 const& sm80_spec(KernelSpec const& spec)
{
  auto const* typed_spec = std::get_if<KernelSpecSM80>(&spec);
  if (typed_spec == nullptr)
    throw std::invalid_argument("pairwise-attention kernel does not use the SM80 launch ABI");
  return *typed_spec;
}

inline KernelSpecSM90 const& sm90_spec(KernelSpec const& spec)
{
  auto const* typed_spec = std::get_if<KernelSpecSM90>(&spec);
  if (typed_spec == nullptr)
    throw std::invalid_argument("pairwise-attention kernel does not use the SM90 launch ABI");
  return *typed_spec;
}

inline std::uint32_t dynamic_smem_bytes(KernelConfig const& config)
{
  if (config.cubin.dynamic_smem_bytes == 0)
    throw std::invalid_argument("Pairwise-attention CUBIN has no dynamic shared-memory metadata");
  return config.cubin.dynamic_smem_bytes;
}

struct LaunchParams
{
  Tensor3View q;
  Tensor3View k;
  Tensor3View v;
  Tensor1View actual_s_kv;
  Tensor4View bias;
  Tensor3View output;
  Tensor3View lse;
  /* Both scales come from the caller. Recomputing the log2 form here would
   * round differently from the source path's double-precision product for
   * head dimensions whose inverse square root is not exactly representable
   * (D=48), so the two implementations would stop matching bitwise.
   */
  float softmax_scale{};
  float softmax_scale_log2{};
  std::int32_t mult{};
  std::uint64_t stream{};
};

KernelConfig
make_kernel_config(std::int32_t target_sm, std::int32_t head_dim, std::int32_t S, DType dtype, bool packed_output);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace trtbnm::cutedsl::pairwise_attention

#endif /* TENSORRT_BIONEMO_CPP_KERNELS_CUTEDSL_PAIRWISE_ATTENTION_LAUNCHER_H_ */
