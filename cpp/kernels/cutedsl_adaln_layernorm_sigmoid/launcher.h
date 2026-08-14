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

/* AdaLN layernorm-sigmoid CUBIN configuration, device ABI, and launcher. */

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_ADALN_LAYERNORM_SIGMOID_LAUNCHER_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_ADALN_LAYERNORM_SIGMOID_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace bioir::cutedsl::adaln_layernorm_sigmoid::embedded
{
struct CubinImage;
}

/* Direct CUDA Driver launch ABI for the AdaLN fused LayerNorm + sigmoid gate.
 *
 * This is the device-kernel ABI, not the high-level CuTeDSL __call__ ABI, and
 * was read back from the compiled CUBINs' EIATTR_KPARAM_INFO: seven parameters
 * over 108 bytes, IDENTICAL for sm_80 and sm_90a and for every dtype and N.
 *
 * The four operands are `s1_d1`, not `s2_d1`, because N is a compile-time
 * constant: only the row count and the outer stride stay dynamic. The launcher
 * still receives both extents so it can check N against the payload's compiled
 * value -- a mismatch means the interface selected the wrong CUBIN.
 *
 * The tiler/tiled-copy/threads-per-row/vecsize arguments in the @cute.kernel
 * signature are compile-time objects and are traced away.
 */
namespace bioir::cutedsl::adaln_layernorm_sigmoid::abi
{

inline constexpr std::size_t kParameterCount = 7;

struct Params
{
  cute_tensor_s1_d1_t x;
  cute_tensor_s1_d1_t s_scale;
  cute_tensor_s1_d1_t s_bias;
  cute_tensor_s1_d1_t output;
  float eps;
  std::int32_t mult;
  std::int32_t inner;
};

/* Both params and kernel_params must remain alive until the CUDA launch call
 * returns.
 */
inline void pack_kernel_params(Params* params, void* kernel_params[kParameterCount])
{
  kernel_params[0] = &params->x;
  kernel_params[1] = &params->s_scale;
  kernel_params[2] = &params->s_bias;
  kernel_params[3] = &params->output;
  kernel_params[4] = &params->eps;
  kernel_params[5] = &params->mult;
  kernel_params[6] = &params->inner;
}

static_assert(sizeof(cute_tensor_s1_d1_t) == 24, "AdaLN s1_d1 operand size changed");
static_assert(offsetof(Params, x) == 0, "unexpected AdaLN x offset");
static_assert(offsetof(Params, s_scale) == 24, "unexpected AdaLN s_scale offset");
static_assert(offsetof(Params, s_bias) == 48, "unexpected AdaLN s_bias offset");
static_assert(offsetof(Params, output) == 72, "unexpected AdaLN output offset");
static_assert(offsetof(Params, eps) == 96, "unexpected AdaLN eps offset");
static_assert(offsetof(Params, mult) == 100, "unexpected AdaLN mult offset");
static_assert(offsetof(Params, inner) == 104, "unexpected AdaLN inner offset");
/* The lowered bank ends at 108 bytes. The backing struct pads to 112 because
 * the int64 strides force 8-byte alignment; harmless, since each parameter is
 * handed to the driver through its own pointer rather than as one blob.
 */
static_assert(offsetof(Params, inner) + sizeof(Params::inner) == 108, "AdaLN parameter bank changed");
static_assert(sizeof(Params) == 112, "AdaLN backing struct size changed");

} // namespace bioir::cutedsl::adaln_layernorm_sigmoid::abi

namespace bioir::cutedsl::adaln_layernorm_sigmoid
{

enum class DType : std::uint8_t
{
  kFloat16 = 0,
  kBFloat16 = 1,
  kFloat32 = 2,
};

/* One compiled variant.
 *
 * N is baked into the machine code, so it is part of the key rather than a
 * runtime extent. `threads_per_row` and `num_threads` are the resolved outputs
 * of the kernel's two override-able geometry functions and together determine
 * the tiled copy -- hence the rows-per-block divisor the grid needs.
 */
struct KernelSpec
{
  std::int32_t target_sm;
  std::int32_t feature_dim;
  std::int32_t threads_per_row;
  std::int32_t num_threads;
  std::int32_t cluster_n;
};

struct KernelConfig
{
  KernelSpec spec;
  DType dtype;
  EmbeddedCubinImage cubin;
  embedded::CubinImage const* embedded_image;
};

inline std::uint32_t dynamic_smem_bytes(KernelConfig const& config)
{
  if (config.cubin.dynamic_smem_bytes == 0)
    throw std::invalid_argument("AdaLN CUBIN has no dynamic shared-memory metadata");
  return config.cubin.dynamic_smem_bytes;
}

/* Rows each CTA covers: tiler_mn[0] == num_threads / threads_per_row. */
inline std::int32_t rows_per_block(KernelSpec const& spec)
{
  if (spec.threads_per_row <= 0 || spec.num_threads % spec.threads_per_row != 0)
    throw std::invalid_argument("AdaLN spec has an invalid thread geometry");
  return spec.num_threads / spec.threads_per_row;
}

struct LaunchParams
{
  /* All four are rank-2 [M, N] row-major. N must equal the payload's compiled
   * feature_dim; only M and the outer stride reach the device.
   */
  Tensor2View x;
  Tensor2View s_scale;
  Tensor2View s_bias;
  Tensor2View output;
  float eps{};
  /* Broadcast multiplicity: x has `mult` times as many rows as s_scale. */
  std::int32_t mult{1};
  std::int32_t inner{1};
  std::uint64_t stream{};
};

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  DType dtype,
  std::int32_t feature_dim,
  std::int32_t threads_per_row,
  std::int32_t num_threads);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace bioir::cutedsl::adaln_layernorm_sigmoid

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_ADALN_LAYERNORM_SIGMOID_LAUNCHER_H_ */
