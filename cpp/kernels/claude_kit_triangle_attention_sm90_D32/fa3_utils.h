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

/******************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024 Tri Dao
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Portions of FlashAttention-3 (hopper/utils.h), BSD-3-Clause:
 *   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 *   following conditions are met: 1. Redistributions of source code must retain the above copyright notice, this list
 * of conditions and the following disclaimer. 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided
 * with the distribution. 3. Neither the name of the copyright holder nor the names of its contributors may be used to
 * endorse or promote products derived from this software without specific prior written permission. THIS SOFTWARE IS
 * PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. (trimmed to the pieces the triangle-attention kernel uses: the
 * accumulator layout views)
 ******************************************************************************/
#ifndef BIOIR_CPP_KERNELS_CLAUDE_KIT_TRIANGLE_ATTENTION_SM90_D32_FA3_UTILS_H_
#define BIOIR_CPP_KERNELS_CLAUDE_KIT_TRIANGLE_ATTENTION_SM90_D32_FA3_UTILS_H_

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

namespace flash
{

using namespace cute;

// SM90: convert acc_layout from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
template <typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_rowcol(Layout0 acc_layout)
{
  static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
  static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
  static_assert(decltype(rank(acc_layout))::value == 3);
  auto l = acc_layout;
  return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)));
};

// SM90, FP16/BF16: convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16, MMA_N))
template <typename MMA_Traits, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_Aregs(Layout0 acc_layout)
{
  static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
  static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
  static_assert(decltype(rank(acc_layout))::value == 3);
  static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
  static_assert(sizeof(typename MMA_Traits::ValTypeA) == 2);
  auto l = logical_divide(get<0, 2>(acc_layout), Tile<_2>{}); // ((2, N / 16))
  return make_layout(
    make_layout(get<0, 0>(acc_layout), get<0, 1>(acc_layout), get<0, 0>(l)),
    get<1>(acc_layout),
    coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
};

} // namespace flash

#endif // BIOIR_CPP_KERNELS_CLAUDE_KIT_TRIANGLE_ATTENTION_SM90_D32_FA3_UTILS_H_
