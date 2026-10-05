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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Anthropic, PBC
 * Modified by NVIDIA Corporation and affiliates.
 *
 * Adapted from the Uplifting Biomolecular Modeling M1 triangle-attention kernel (triattn_m1_sm90.cuh). NOTICE.md lists
 * the changes.
 */

// Triangle attention forward for SM90 (sm_90a), BF16, head dimension 32, over the keys k < actual_s_kv[b, i]:
//   out[b, i, h, q, :] = softmax_k(scale * q[b, i, h, q, :] . k[b, i, h, k, :] + bias[b, h, q, k]) @ v[b, i, h, k, :]
// Persistent CTAs take (b, h, 128 queries, 3 pair rows) work tiles. Producer warp 1 loads Q and K/V, warp 0 the bias,
// warp 2 stores O, all by TMA; consumer warpgroup w computes pair row i0 + w.

#ifndef BIOIR_CPP_KERNELS_CLAUDE_KIT_TRIANGLE_ATTENTION_SM90_D32_KERNEL_CUH_
#define BIOIR_CPP_KERNELS_CLAUDE_KIT_TRIANGLE_ATTENTION_SM90_D32_KERNEL_CUH_

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

#include <cfloat>
#include <climits>
#include <cuda_bf16.h>

#include "fa3_utils.h"

namespace bioir::claude_kit_triangle_attention_sm90_d32::detail
{

using namespace cute;

// TMA-filled shared-memory stages; use u of a stage has phase u & 1. ClusterBarrier::wait() has no "memory" clobber,
// so every wait here ends in a compiler barrier.
template <int Stages>
struct Pipe
{
  struct SharedStorage
  {
    cutlass::arch::ClusterTransactionBarrier full[Stages];
    cutlass::arch::ClusterBarrier empty[Stages];
  };

  SharedStorage& storage;

  CUTLASS_DEVICE explicit Pipe(SharedStorage& s)
    : storage(s)
  {
  }

  CUTLASS_DEVICE static void init(SharedStorage& s, int empty_arrivals, int full_arrivals = 1)
  {
    for (int i = 0; i < Stages; ++i)
    {
      s.full[i].init(full_arrivals);
      s.empty[i].init(empty_arrivals);
    }
  }

  CUTLASS_DEVICE void producer_wait_empty(int stage, uint32_t phase)
  {
    storage.empty[stage].wait(phase ^ 1);
    asm volatile("" ::: "memory");
  }

  CUTLASS_DEVICE void producer_expect(int stage, uint32_t bytes)
  {
    storage.full[stage].arrive_and_expect_tx(bytes);
  }

  CUTLASS_DEVICE void producer_arrive(int stage)
  {
    storage.full[stage].arrive();
  }

  CUTLASS_DEVICE uint64_t* full_barrier(int stage)
  {
    return reinterpret_cast<uint64_t*>(&storage.full[stage]);
  }

  CUTLASS_DEVICE void wait_full(int stage, uint32_t phase)
  {
    storage.full[stage].wait(phase);
    asm volatile("" ::: "memory");
  }

  CUTLASS_DEVICE void release(int stage, bool elected)
  {
    if (elected)
    {
      storage.empty[stage].arrive();
    }
  }
};

struct Traits
{
  using Element = cutlass::bfloat16_t;
  static constexpr int kHeadDim = 32;
  // A work tile is kBlockM queries x kRows pair rows; keys stream in kBlockN-key tiles of kChunkN-key columns.
  static constexpr int kBlockM = 128;
  static constexpr int kBlockN = 128;
  static constexpr int kChunkN = 32;
  static constexpr int kRows = 3;
  static constexpr int kBiasSlots = kBlockN / kChunkN;
  static constexpr int kKVRingTiles = 2;
  static constexpr int kStagesKV = kKVRingTiles * kRows;
  static constexpr int kNumMmaWG = kRows;
  static constexpr int kNumMmaThreads = kNumMmaWG * 128;
  static constexpr int kNumThreads = kNumMmaThreads + 128;
  static_assert(kBiasSlots == 4);

  using AtomLayout = Layout<Shape<_1, _1, _1>>;
  using TiledMmaQK = decltype(make_tiled_mma(
    GMMA::ss_op_selector<Element, Element, float, Shape<_64, Int<kChunkN>, Int<kHeadDim>>>(), AtomLayout{}));
  using TiledMmaPV
    = decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}, AtomLayout{}));
  using TiledMmaL = decltype(make_tiled_mma(
    SM90_64x8x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{}, AtomLayout{})); // l += P 1 (B = all-ones 8x32 tile)

  using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::
                                     ss_smem_selector<GMMA::Major::K, Element, Int<kBlockM>, Int<kHeadDim>>());
  using SmemLayoutQ
    = decltype(tile_to_shape(SmemLayoutAtomQ{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, Int<kRows>{})));
  using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::
                                     ss_smem_selector<GMMA::Major::K, Element, Int<kBlockN>, Int<kHeadDim>>());
  using SmemLayoutK
    = decltype(tile_to_shape(SmemLayoutAtomK{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, Int<kStagesKV>{})));
  static constexpr int kStageElemsK = kBlockN * kHeadDim;
  static_assert(cosize(SmemLayoutK{}) == kStageElemsK * kStagesKV);
  using SmemLayoutV = SmemLayoutK;
  using SmemLayoutVt = decltype(cute::composition(
    SmemLayoutV{},
    make_ordered_layout(make_shape(Int<kHeadDim>{}, Int<kBlockN>{}, Int<kStagesKV>{}), Step<_2, _1, _3>{})));
  static constexpr int kStageElemsV = kStageElemsK;
  using SmemLayoutOnes
    = decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<Element>{}, make_shape(_8{}, Int<kChunkN>{})));

  // Bias slot c (keys 32c .. 32c + 31, 128 queries) holds bias / scale in fragment order [half][u][thread][4]: float4 u
  // of thread t is its score elements 4u .. 4u + 3. The pipeline unit is the half slot m = 2c + h.
  static constexpr int kBiasSlotElems = 2 * 4 * 128 * 4;
  static constexpr int kBiasHalfElems = kBiasSlotElems / 2;
  static constexpr int kBiasHalves = 2 * kBiasSlots;
  using SmemLayoutBiasHalf = Layout<Shape<_256, _8>, Stride<_1, _256>>;
  // staged bias (256, 8, half slot 8 j + m of key tile j, query tile, b * H + h)
  using ShapeBias = Shape<int32_t, int32_t, int32_t, int32_t, int32_t>;
  using StrideBias = Stride<_1, _256, int64_t, int64_t, int64_t>;
  using ShapeQK = Shape<int32_t, int32_t, int32_t, int32_t, int32_t>; // (S, D, H, I, B)
  using StrideQK = Stride<int64_t, _1, int64_t, int64_t, int64_t>;

  using TMA_Q = decltype(make_tma_copy(
    SM90_TMA_LOAD{},
    make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQK{}, StrideQK{}),
    take<0, 2>(SmemLayoutQ{}),
    make_shape(Int<kBlockM>{}, Int<kHeadDim>{}),
    _1{}));
  using TMA_K = decltype(make_tma_copy(
    SM90_TMA_LOAD{},
    make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQK{}, StrideQK{}),
    take<0, 2>(SmemLayoutK{}),
    make_shape(Int<kBlockN>{}, Int<kHeadDim>{}),
    _1{}));
  using TMA_V = TMA_K;
  using TMA_O = decltype(make_tma_copy(
    SM90_TMA_STORE{},
    make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeQK{}, StrideQK{}),
    take<0, 2>(SmemLayoutQ{}),
    make_shape(Int<kBlockM>{}, Int<kHeadDim>{}),
    _1{}));
  using TMA_Bias = decltype(make_tma_copy(
    SM90_TMA_LOAD{},
    make_tensor(make_gmem_ptr(static_cast<float const*>(nullptr)), ShapeBias{}, StrideBias{}),
    SmemLayoutBiasHalf{},
    make_shape(_256{}, _8{}),
    _1{}));

  static constexpr uint32_t kBytesQ = kBlockM * kHeadDim * sizeof(Element);
  static constexpr uint32_t kBytesK = kBlockN * kHeadDim * sizeof(Element);
  static constexpr uint32_t kBytesV = kBytesK;
  static constexpr uint32_t kBytesBiasHalf = kBiasHalfElems * sizeof(float);

  // full: the K and the V transaction of a stage (two arrivals); empty: V free, after the key tile's last PV
  using PipeKV = Pipe<kStagesKV>;
  // empty only: K free, after the key tile's last QK (its full barriers are unused)
  using PipeK = Pipe<kStagesKV>;
  // empty[m]: half slot m = 2c + h free; full[c] (c < 4): both halves of slot c landed (two transactions)
  using PipeBias = Pipe<kBiasHalves>;
  // wgmma.wait_group retires only the calling warp's wgmmas, so every consumer warp arrives on an empty barrier.
  static constexpr int kArrivalsKV = 4;
  static constexpr int kArrivalsBias = kNumMmaWG * 4;

  static constexpr int kQBuffers = 2;
  static constexpr int kQBufferElems = int(cute::cosize_v<SmemLayoutQ>);
  static constexpr int kArrivalsO = kNumMmaWG * 4;
  static constexpr int kArrivalsQ = 1;

  struct SharedStorage
  {
    cute::array_aligned<Element, kQBufferElems * kQBuffers, 1024> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, 1024> smem_k;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, 1024> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutOnes>, 128> smem_ones;
    cute::array_aligned<float, kBiasSlotElems * kBiasSlots, 1024> smem_bias;
    typename PipeKV::SharedStorage pipe_kv;
    typename PipeK::SharedStorage pipe_k;
    typename PipeBias::SharedStorage pipe_bias;
    cutlass::arch::ClusterTransactionBarrier q_full[kQBuffers];
    cutlass::arch::ClusterBarrier q_empty[kQBuffers];
    cutlass::arch::ClusterBarrier o_full[kQBuffers];
    // Published with q_full: the tile (b < 0: no work left), its row lengths and its batch's staged length.
    alignas(16) int tile_coord[kQBuffers][4];
    int tile_length[kQBuffers][kRows];
    int tile_staged_length[kQBuffers];
  };

  struct Params
  {
    TMA_Q tma_q;
    TMA_K tma_k;
    TMA_V tma_v;
    TMA_Bias tma_bias;
    TMA_O tma_o;
    ShapeQK shape_qk;
    ShapeBias shape_bias;
    int seqlen, i_dim, num_heads;
    int n_qtiles, n_row_groups, head_group;
    int n_work;
    int* tile_counter; // tickets for the work tiles after the first wave; 0 at launch
    float softmax_scale;
    int const* staged_length; // [B]: the longest row of each batch; the staged bias is -inf at or past it
    int const* actual_s_kv;   // [B * I]: live leading keys of each row
    float* lse;               // [B * I, S, H] with H stride 1; null to skip
    int64_t lse_stride_bi, lse_stride_s;
  };
};

__device__ __forceinline__ uint32_t max_u16x2(uint32_t a, uint32_t b)
{
  uint32_t d;
  asm("max.u16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ float ex2_approx(float x)
{
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

template <class T>
__global__ void __launch_bounds__(T::kNumThreads, 1)
  triangle_attention_kernel(CUTE_GRID_CONSTANT typename T::Params const params)
{
  using Element = typename T::Element;
  constexpr int kRows = T::kRows, kBlockM = T::kBlockM, kBlockN = T::kBlockN, kHeadDim = T::kHeadDim;
  constexpr int kChunkN = T::kChunkN;
  using SharedStorage = typename T::SharedStorage;
  extern __shared__ char smem_buf[];
  SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_buf);

  int const warp_idx = cutlass::canonical_warp_idx_sync();
  int const lane_predicate = cute::elect_one_sync();
  int const wg_idx = cutlass::canonical_warp_group_idx();
  int const tid = threadIdx.x;
  int const lane = tid % 32;
  int const seqlen = params.seqlen;

  struct WorkTile
  {
    int b, h, qtile, i0; // batch, head, query tile, first pair row
  };

  auto decode_tile = [&](int linear) __attribute__((always_inline))
  {
    WorkTile tile;
    int const h_lo = linear % params.head_group;
    linear /= params.head_group;
    tile.qtile = linear % params.n_qtiles;
    linear /= params.n_qtiles;
    tile.i0 = (linear % params.n_row_groups) * kRows; // rows >= I load clamped and are never stored
    linear /= params.n_row_groups;
    int const n_head_groups = params.num_heads / params.head_group;
    tile.h = (linear % n_head_groups) * params.head_group + h_lo;
    tile.b = linear / n_head_groups;
    return tile;
  };

  auto published_tile = [&](int q_buf) __attribute__((always_inline))
  {
    int4 const c = *reinterpret_cast<int4 const*>(shared.tile_coord[q_buf]);
    return WorkTile{c.x, c.y, c.z, c.w};
  };

  if (warp_idx == 0 && lane_predicate)
  {
    cute::prefetch_tma_descriptor(params.tma_q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_k.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_v.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_bias.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_o.get_tma_descriptor());
#pragma unroll
    for (int s = 0; s < T::kQBuffers; ++s)
    {
      shared.q_full[s].init(2); // the Q transaction and the tile's publication
      shared.q_empty[s].init(T::kArrivalsQ);
      shared.o_full[s].init(T::kArrivalsO);
    }
    T::PipeKV::init(shared.pipe_kv, T::kArrivalsKV, 2);
    T::PipeK::init(shared.pipe_k, T::kArrivalsKV);
    T::PipeBias::init(shared.pipe_bias, T::kArrivalsBias, 2);
    cutlass::arch::fence_barrier_init();
  }
  typename T::PipeKV pipe_kv(shared.pipe_kv);
  typename T::PipeK pipe_k(shared.pipe_k);
  typename T::PipeBias pipe_bias(shared.pipe_bias);

  if (wg_idx != 0)
  {
    for (int idx = tid - 128; idx < int(cute::cosize_v<typename T::SmemLayoutOnes>); idx += T::kNumMmaThreads)
    {
      shared.smem_ones[idx] = Element(1.f);
    }
    cutlass::arch::fence_view_async_shared(); // make the generic-proxy writes visible to wgmma (async proxy)
  }
  __syncthreads();

  // All rows of a work tile stream the key tiles of its longest row; a shorter row masks its own dead keys.
  auto load_lengths = [&](WorkTile const& tile, int (&length)[kRows]) __attribute__((always_inline))
  {
#pragma unroll
    for (int r = 0; r < kRows; ++r)
    {
      int64_t const row = int64_t(tile.b) * params.i_dim + min(tile.i0 + r, params.i_dim - 1);
      length[r] = min(max(params.actual_s_kv[row], 0), seqlen);
    }
  };
  auto count_key_tiles = [&](int const(&length)[kRows]) __attribute__((always_inline))
  {
    int const longest = max(max(length[0], length[1]), length[2]);
    return (longest + kBlockN - 1) / kBlockN;
  };
  auto count_key_columns = [&](int const(&length)[kRows]) __attribute__((always_inline))
  {
    int const longest = max(max(length[0], length[1]), length[2]);
    return (longest + kChunkN - 1) / kChunkN;
  };
  auto second_half_empty
    = [&](WorkTile const& tile) __attribute__((always_inline)) { return seqlen - tile.qtile * kBlockM <= 64; };

  // Producers and consumers count the key tiles streamed so far (the ring position p, carried across work tiles): key
  // tile p uses K/V stage set p & 1 for the (p >> 1)-th time and every bias half slot for the p-th time.
  if (wg_idx == 0)
  {
    // ============================================== PRODUCERS =====================================================
    cutlass::arch::warpgroup_reg_dealloc<32>(); // 128 * 32 + 384 * 160 = 512 * 128, the registers the CTA launched with
    int const warp_idx_in_wg = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if (warp_idx_in_wg == 1 && lane_predicate)
    {
      Tensor sK = make_tensor(make_smem_ptr(shared.smem_k.data()), typename T::SmemLayoutK{});
      Tensor sV = make_tensor(make_smem_ptr(shared.smem_v.data()), typename T::SmemLayoutV{});
      auto block_tma_q = params.tma_q.get_slice(_0{});
      auto block_tma_k = params.tma_k.get_slice(_0{});
      auto block_tma_v = params.tma_v.get_slice(_0{});
      Tensor tKsK = group_modes<0, 3>(block_tma_k.partition_D(sK));
      Tensor tVsV = group_modes<0, 3>(block_tma_v.partition_D(sV));
      int ring_pos = 0;
      int linear = int(blockIdx.x);
#pragma unroll 1
      for (int n = 0;; ++n)
      {
        int const q_buf = n & 1;
        shared.q_empty[q_buf].wait(((n >> 1) & 1) ^ 1);
        asm volatile("" ::: "memory");
        if (linear >= params.n_work)
        {
          shared.tile_coord[q_buf][0] = -1;
          asm volatile("" ::: "memory"); // stored before the arrivals publish it
          shared.q_full[q_buf].arrive();
          shared.q_full[q_buf].arrive();
          break;
        }
        WorkTile const tile = decode_tile(linear);
        Tensor sQ
          = make_tensor(make_smem_ptr(shared.smem_q.data() + q_buf * T::kQBufferElems), typename T::SmemLayoutQ{});
        Tensor mQ = params.tma_q.get_tma_tensor(params.shape_qk)(_, _, tile.h, _, tile.b);
        Tensor gQ = local_tile(mQ, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(tile.qtile, _0{}, _));
        Tensor tQgQ = group_modes<0, 3>(block_tma_q.partition_S(gQ));
        Tensor tQsQ = group_modes<0, 3>(block_tma_q.partition_D(sQ));
        shared.q_full[q_buf].arrive_and_expect_tx(T::kBytesQ * kRows);
#pragma unroll
        for (int r = 0; r < kRows; ++r)
        {
          copy(
            params.tma_q.with(reinterpret_cast<uint64_t&>(shared.q_full[q_buf]), 0),
            tQgQ(_, min(tile.i0 + r, params.i_dim - 1)),
            tQsQ(_, r));
        }
        int length[kRows];
        load_lengths(tile, length);
        int const staged_length = params.staged_length[tile.b];
        int const n_key_tiles = count_key_tiles(length);
        shared.tile_coord[q_buf][0] = tile.b;
        shared.tile_coord[q_buf][1] = tile.h;
        shared.tile_coord[q_buf][2] = tile.qtile;
        shared.tile_coord[q_buf][3] = tile.i0;
#pragma unroll
        for (int r = 0; r < kRows; ++r)
        {
          shared.tile_length[q_buf][r] = length[r];
        }
        shared.tile_staged_length[q_buf] = staged_length;
        asm volatile("" ::: "memory"); // the tile is stored before the arrival publishes it
        shared.q_full[q_buf].arrive();
        linear = int(gridDim.x) + atomicAdd(params.tile_counter, 1);
        if (n_key_tiles == 0)
        {
          continue;
        }
        Tensor mK = params.tma_k.get_tma_tensor(params.shape_qk)(_, _, tile.h, _, tile.b);
        Tensor mV = params.tma_v.get_tma_tensor(params.shape_qk)(_, _, tile.h, _, tile.b);
        // TMA zero-fills keys >= S
        Tensor gK = local_tile(mK, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(_, _0{}, _));
        Tensor gV = local_tile(mV, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(_, _0{}, _));
        Tensor tKgK = group_modes<0, 3>(block_tma_k.partition_S(gK));
        Tensor tVgV = group_modes<0, 3>(block_tma_v.partition_S(gV));
#pragma unroll 1
        for (int j = 0; j < n_key_tiles; ++j, ++ring_pos)
        {
          int const set = (ring_pos & 1) * kRows;
          uint32_t const phase = (ring_pos >> 1) & 1;
#pragma unroll 1
          for (int r = 0; r < kRows; ++r)
          {
            pipe_k.producer_wait_empty(set + r, phase);
            pipe_kv.producer_expect(set + r, T::kBytesK);
            copy(
              params.tma_k.with(*pipe_kv.full_barrier(set + r), 0),
              tKgK(_, j, min(tile.i0 + r, params.i_dim - 1)),
              tKsK(_, set + r));
          }
#pragma unroll 1
          for (int r = 0; r < kRows; ++r)
          {
            pipe_kv.producer_wait_empty(set + r, phase);
            pipe_kv.producer_expect(set + r, T::kBytesV);
            copy(
              params.tma_v.with(*pipe_kv.full_barrier(set + r), 0),
              tVgV(_, j, min(tile.i0 + r, params.i_dim - 1)),
              tVsV(_, set + r));
          }
        }
      }
    }
    else if (warp_idx_in_wg == 0 && lane_predicate)
    {
      // Warp 0: the bias. A half slot with nothing to load completes its phase with a data-less arrival.
      auto sBias_half = [&](int m)
      {
        return make_tensor(
          make_smem_ptr(shared.smem_bias.data() + m * T::kBiasHalfElems), typename T::SmemLayoutBiasHalf{});
      };
      auto block_tma_bias = params.tma_bias.get_slice(_0{});
      auto tBsB = [&](int m) { return group_modes<0, 3>(block_tma_bias.partition_D(sBias_half(m))); };
      uint32_t ring_pos = 0;
#pragma unroll 1
      for (int n = 0;; ++n)
      {
        int const q_buf = n & 1;
        shared.q_full[q_buf].wait((n >> 1) & 1);
        asm volatile("" ::: "memory");
        WorkTile const tile = published_tile(q_buf);
        if (tile.b < 0)
        {
          break;
        }
        int length[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r)
        {
          length[r] = shared.tile_length[q_buf][r];
        }
        int const n_key_tiles = count_key_tiles(length);
        int const n_key_cols = count_key_columns(length);
        bool const first_half_only = second_half_empty(tile);
        Tensor gB
          = params.tma_bias.get_tma_tensor(params.shape_bias)(_, _, _, tile.qtile, tile.b * params.num_heads + tile.h);
        Tensor tBgB = group_modes<0, 3>(block_tma_bias.partition_S(gB));
#pragma unroll 1
        for (int j = 0; j < n_key_tiles; ++j, ++ring_pos)
        {
          uint32_t const phase = ring_pos & 1;
#pragma unroll
          for (int m = 0; m < T::kBiasHalves; ++m)
          {
            pipe_bias.producer_wait_empty(m, phase);
            if (4 * j + (m >> 1) < n_key_cols && ((m & 1) == 0 || !first_half_only))
            {
              pipe_bias.producer_expect(m >> 1, T::kBytesBiasHalf);
              copy(params.tma_bias.with(*pipe_bias.full_barrier(m >> 1), 0), tBgB(_, 8 * j + m), tBsB(m));
            }
            else
            {
              pipe_bias.producer_arrive(m >> 1);
            }
          }
        }
      }
    }
    else if (warp_idx_in_wg == 2 && lane_predicate)
    {
      auto block_tma_o = params.tma_o.get_slice(_0{});
#pragma unroll 1
      for (int n = 0;; ++n)
      {
        int const q_buf = n & 1;
        uint32_t const phase = (n >> 1) & 1;
        shared.q_full[q_buf].wait(phase);
        asm volatile("" ::: "memory");
        WorkTile const tile = published_tile(q_buf);
        if (tile.b < 0)
        {
          break;
        }
        shared.o_full[q_buf].wait(phase);
        asm volatile("" ::: "memory");
        Tensor sO
          = make_tensor(make_smem_ptr(shared.smem_q.data() + q_buf * T::kQBufferElems), typename T::SmemLayoutQ{});
        Tensor mO = params.tma_o.get_tma_tensor(params.shape_qk)(_, _, tile.h, _, tile.b);
        Tensor gO = local_tile(mO, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(tile.qtile, _0{}, _));
        Tensor tOsO = group_modes<0, 3>(block_tma_o.partition_S(sO));
        Tensor tOgO = group_modes<0, 3>(block_tma_o.partition_D(gO));
#pragma unroll
        for (int r = 0; r < kRows; ++r)
        {
          if (tile.i0 + r < params.i_dim)
          {
            copy(params.tma_o, tOsO(_, r), tOgO(_, tile.i0 + r));
          }
        }
        tma_store_arrive();
        tma_store_wait<0>(); // the stores have read the buffer
        shared.q_empty[q_buf].arrive();
      }
    }
    return;
  }

  // ================================================ CONSUMERS =======================================================
  cutlass::arch::warpgroup_reg_alloc<160>();
  int const mma_thread = tid - 128;
  int const mma_wg = int(__reduce_max_sync(0xffffffffu, unsigned(mma_thread) / 128u)); // in a uniform register
  int const wg_thread = mma_thread % 128;
  bool const warp_leader = lane == 0;
  float const softmax_scale = params.softmax_scale;

  WorkTile tile{};
  int i = 0;
  int row_length = 0;
  int mask_from = INT_MAX; // keys at or past it are masked (INT_MAX: the row needs no mask)
  int n_key_tiles = 0;
  int n_key_cols = 0;
  int ring_base = 0;

  constexpr float kLog2e = 1.4426950408889634f;
  float const scale_log2 = softmax_scale * kLog2e;

  typename T::TiledMmaQK tiled_mma_qk;
  typename T::TiledMmaPV tiled_mma_pv;
  typename T::TiledMmaL tiled_mma_l;
  auto wg_mma_qk = tiled_mma_qk.get_slice(0);
  auto wg_mma_pv = tiled_mma_pv.get_slice(0);
  auto wg_mma_l = tiled_mma_l.get_slice(0);
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(wg_thread);
  Tensor sOnes = make_tensor(make_smem_ptr(shared.smem_ones.data()), typename T::SmemLayoutOnes{});
  Tensor tOnes = wg_mma_l.partition_fragment_B(sOnes);
  int const kv_stage0 = mma_wg; // this warpgroup's K/V stage in set 0 (set s: + s * kRows)

  // Q operand descriptors of Q buffer q_buf: (frag, 1, k-block, query half, row)
  auto make_q_operand = [&](int q_buf)
  {
    Tensor sQ = make_tensor(make_smem_ptr(shared.smem_q.data() + q_buf * T::kQBufferElems), typename T::SmemLayoutQ{});
    return wg_mma_qk.partition_fragment_A(local_tile(sQ, make_shape(_64{}, Int<kHeadDim>{}), make_coord(_, _0{}, _)));
  };
  auto tQ = make_q_operand(0);
  static_assert(
    decltype(size<2>(tQ))::value == 2 && decltype(size<3>(tQ))::value == 2 && decltype(size<4>(tQ))::value == kRows);
  // K and V operand descriptors (frag, 1, k-block, key column, stage) at this warpgroup's set-0 stage plus offset
  auto make_k_operand = [&](int offset)
  {
    Tensor sK
      = make_tensor(make_smem_ptr(shared.smem_k.data() + mma_wg * T::kStageElemsK + offset), typename T::SmemLayoutK{});
    return wg_mma_qk.partition_fragment_B(
      local_tile(sK, make_shape(Int<kChunkN>{}, Int<kHeadDim>{}), make_coord(_, _0{})));
  };
  auto make_v_operand = [&](int offset)
  {
    Tensor sVt = make_tensor(
      make_smem_ptr(shared.smem_v.data() + mma_wg * T::kStageElemsV + offset), typename T::SmemLayoutVt{});
    return wg_mma_pv.partition_fragment_B(
      local_tile(sVt, make_shape(Int<kHeadDim>{}, Int<kChunkN>{}), make_coord(_0{}, _)));
  };
  using KOperand = decltype(make_k_operand(0));
  using VOperand = decltype(make_v_operand(0));
  constexpr int kKVSetElems = kRows * T::kStageElemsK;
  static_assert(T::kStageElemsV == T::kStageElemsK);

  Tensor cO = make_identity_tensor(make_shape(_64{}, Int<kHeadDim>{}));
  Tensor tOcO = thr_mma_pv.partition_C(cO);
  Tensor tOcO_rc = make_tensor(tOcO.data(), flash::convert_layout_acc_rowcol(tOcO.layout()));
  using AccS = decltype(partition_fragment_C(tiled_mma_qk, make_shape(_64{}, Int<kChunkN>{})));
  using AccO = decltype(partition_fragment_C(tiled_mma_pv, make_shape(_64{}, Int<kHeadDim>{})));
  using AccL = decltype(partition_fragment_C(tiled_mma_l, make_shape(_64{}, _8{})));
  static_assert(
    decltype(size(AccS{}))::value == 16 && decltype(size(AccO{}))::value == 16 && decltype(size(AccL{}))::value == 4);
  constexpr int kAccRows = 2;
  constexpr int kPVKBlocks = kChunkN / 16;
  constexpr int kAccColsO = kHeadDim / 4;
  AccO acc_o[2];
  AccL acc_l[2];
  AccS acc_s[4]; // the scores of chunk k in acc_s[k & 3]
  auto p_proto = make_tensor_like<Element>(
    make_tensor(acc_s[0].data(), flash::convert_layout_acc_Aregs<typename T::TiledMmaPV>(acc_s[0].layout())));
  using AccP = decltype(p_proto);
  AccP p_regs[2]; // bf16 P of chunk k in p_regs[k & 1], the A operand of PV(k)
  static_assert(decltype(size<2>(p_proto))::value == kPVKBlocks);

  // Lazy online softmax per query half and row: p = 2^(S * scale_log2 + neg_max) with neg_max = -m * scale_log2. The
  // reference m moves (rescaling O and l) only past a kTau margin, from -FLT_MAX so masked rows avoid inf - inf.
  constexpr float kTau = 8.f;
  constexpr uint32_t kPBoundPacked = 0x43804380u; // 2^kTau in both bf16 halves
  float neg_max[2][kAccRows];
  float const neg_max_init = FLT_MAX * fminf(scale_log2, 1.f);

  // Epilogue: O / l to this row's Q-buffer slot, the LSE to global memory; zero-length rows write zeros and LSE -1e9.
  auto smem_tiled_copy_o = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, Element>{}, tiled_mma_pv);
  auto smem_thr_copy_o = smem_tiled_copy_o.get_thread_slice(wg_thread);
  auto stage_output = [&](int q_buf) __attribute__((always_inline))
  {
    bool const live = row_length > 0;
    Tensor sO = make_tensor(make_smem_ptr(shared.smem_q.data() + q_buf * T::kQBufferElems), typename T::SmemLayoutQ{})(
      _, _, mma_wg);
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
    {
      Tensor o_rc = make_tensor(acc_o[hh].data(), flash::convert_layout_acc_rowcol(acc_o[hh].layout()));
      Tensor l_rc = make_tensor(acc_l[hh].data(), flash::convert_layout_acc_rowcol(acc_l[hh].layout()));
      Tensor r_o = make_tensor_like<Element>(acc_o[hh]);
      Tensor r_rc = make_tensor(r_o.data(), flash::convert_layout_acc_rowcol(r_o.layout()));
#pragma unroll
      for (int mi = 0; mi < kAccRows; ++mi)
      {
        float const l = l_rc(mi, 0);
        float const inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
        for (int ni = 0; ni < kAccColsO; ++ni)
        {
          r_rc(mi, ni) = Element(live ? o_rc(mi, ni) * inv : 0.f);
        }
        int const q = tile.qtile * kBlockM + 64 * hh + get<0>(tOcO_rc(mi, _0{}));
        if (params.lse != nullptr && get<1>(tOcO_rc(mi, _0{})) == 0 && i < params.i_dim && q < seqlen)
        {
          int64_t const bi = int64_t(tile.b) * params.i_dim + i;
          constexpr float kLn2 = 0.6931471805599453f;
          params.lse[bi * params.lse_stride_bi + int64_t(q) * params.lse_stride_s + tile.h]
            = !live ? -1.0e9F * softmax_scale : (l > 0.f ? logf(l) - neg_max[hh][mi] * kLn2 : INFINITY);
        }
      }
      Tensor sOh = local_tile(sO, make_shape(_64{}, Int<kHeadDim>{}), make_coord(hh, _0{}));
      cute::copy(smem_tiled_copy_o, smem_thr_copy_o.retile_S(r_o), smem_thr_copy_o.partition_D(sOh));
    }
    cutlass::arch::fence_view_async_shared(); // make the generic-proxy writes visible to the TMA store (async proxy)
    __syncwarp();
    if (warp_leader)
    {
      shared.o_full[q_buf].arrive();
    }
  };

  // ---- per-chunk primitives; the query half, key column and stage arguments are compile-time constants ----

  float const* bias_frag = shared.smem_bias.data() + wg_thread * 4;
  // Scores start at the bias: float4 u holds keys 8u + 2 (t % 4) + {0, 1} (elements {0, 2} and {1, 3}); masked keys
  // start at -inf. No uniform branch around the selects: ptxas would insert wgmma waits (C7517).
  auto init_scores
    = [&](AccS& acc, auto column_c, auto half_c, int key_tile, auto masked_c) __attribute__((always_inline))
  {
    constexpr int c = decltype(column_c)::value, hh = decltype(half_c)::value;
    int const limit = mask_from - kBlockN * key_tile - kChunkN * c - 2 * (wg_thread % 4);
#pragma unroll
    for (int u = 0; u < 4; ++u)
    {
      float4 const f4 = *reinterpret_cast<float4 const*>(bias_frag + c * T::kBiasSlotElems + (hh * 4 + u) * 512);
      if constexpr (decltype(masked_c)::value)
      {
        bool const keep0 = 8 * u < limit, keep1 = 8 * u + 1 < limit;
        acc(4 * u + 0) = keep0 ? f4.x : -INFINITY;
        acc(4 * u + 1) = keep1 ? f4.y : -INFINITY;
        acc(4 * u + 2) = keep0 ? f4.z : -INFINITY;
        acc(4 * u + 3) = keep1 ? f4.w : -INFINITY;
      }
      else
      {
        acc(4 * u + 0) = f4.x;
        acc(4 * u + 1) = f4.y;
        acc(4 * u + 2) = f4.z;
        acc(4 * u + 3) = f4.w;
      }
    }
  };
  auto init_scores_prologue = [&](AccS& acc, auto column_c, auto half_c, int key_tile) __attribute__((always_inline))
  {
    if (mask_from < kBlockN * (key_tile + 1))
    {
      init_scores(acc, column_c, half_c, key_tile, cute::true_type{});
    }
    else
    {
      init_scores(acc, column_c, half_c, key_tile, cute::false_type{});
    }
  };
  // S += Q K^T onto the bias init; no fence after the commit, so the scores stay in flight until a later wait.
  auto issue_qk
    = [&](AccS& acc, KOperand const& tK, auto half_c, auto column_c, auto stage_c) __attribute__((always_inline))
  {
    constexpr int hh = decltype(half_c)::value, c = decltype(column_c)::value, st = decltype(stage_c)::value;
    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
#pragma unroll
    for (int kb = 0; kb < 2; ++kb)
    {
      cute::gemm(tiled_mma_qk, tQ(_, _, kb, hh, mma_wg), tK(_, _, kb, c, st), acc);
    }
    warpgroup_commit_batch();
  };
  // O += P V and l += P 1 over the same bf16 P, in one commit group. acc_o is not fenced: the PVs of a half chain.
  auto issue_pv
    = [&](AccP& tP, VOperand const& tV, auto half_c, auto column_c, auto stage_c) __attribute__((always_inline))
  {
    constexpr int hh = decltype(half_c)::value, c = decltype(column_c)::value, st = decltype(stage_c)::value;
    warpgroup_fence_operand(tP);
    warpgroup_arrive();
    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
    tiled_mma_l.accumulate_ = GMMA::ScaleOut::One;
#pragma unroll
    for (int kb = 0; kb < kPVKBlocks; ++kb)
    {
      cute::gemm(tiled_mma_pv, tP(_, _, kb), tV(_, _, kb, c, st), acc_o[hh]);
    }
#pragma unroll
    for (int kb = 0; kb < kPVKBlocks; ++kb)
    {
      cute::gemm(tiled_mma_l, tP(_, _, kb), tOnes(_, _, kb), acc_l[hh]);
    }
    warpgroup_commit_batch();
  };
  // PV(k - 1), then QK(k + 2), under one warpgroup fence and as two commit groups in that order: the next body's
  // wait<1> then retires PV(k - 1) before E(k + 1) overwrites its P buffer, while QK(k + 2) may stay in flight.
  auto issue_pv_qk = [&](
                       AccP& tP,
                       VOperand const& tV,
                       auto pv_half_c,
                       auto pv_column_c,
                       AccS& acc,
                       KOperand const& tK,
                       auto qk_half_c,
                       auto qk_column_c) __attribute__((always_inline))
  {
    constexpr int hh = decltype(pv_half_c)::value, c = decltype(pv_column_c)::value;
    constexpr int hq = decltype(qk_half_c)::value, cq = decltype(qk_column_c)::value;
    warpgroup_fence_operand(tP);
    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
    tiled_mma_l.accumulate_ = GMMA::ScaleOut::One;
#pragma unroll
    for (int kb = 0; kb < kPVKBlocks; ++kb)
    {
      cute::gemm(tiled_mma_pv, tP(_, _, kb), tV(_, _, kb, c, _0{}), acc_o[hh]);
    }
#pragma unroll
    for (int kb = 0; kb < kPVKBlocks; ++kb)
    {
      cute::gemm(tiled_mma_l, tP(_, _, kb), tOnes(_, _, kb), acc_l[hh]);
    }
    warpgroup_commit_batch();
    tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
#pragma unroll
    for (int kb = 0; kb < 2; ++kb)
    {
      cute::gemm(tiled_mma_qk, tQ(_, _, kb, hq, mma_wg), tK(_, _, kb, cq, _0{}), acc);
    }
    warpgroup_commit_batch();
  };

  // p = 2^(S * scale_log2 + neg_max) packed to bf16. P >= 0, so the u16 max of the packed halves, floored at 2^kTau's
  // bit pattern, equals kPBoundPacked iff every P <= 2^kTau; otherwise the caller redoes the chunk.
  auto exp_pack = [&](AccS& acc, AccP& tP, auto half_c) __attribute__((always_inline)) -> uint32_t
  {
    constexpr int hh = decltype(half_c)::value;
    auto packed = recast<uint32_t>(tP);
#pragma unroll
    for (int pr = 0; pr < kChunkN / 4; ++pr)
    {
      float const row_neg_max = neg_max[hh][pr & 1];
      __nv_bfloat162 const p2 = __floats2bfloat162_rn(
        ex2_approx(fmaf(acc(2 * pr), scale_log2, row_neg_max)),
        ex2_approx(fmaf(acc(2 * pr + 1), scale_log2, row_neg_max)));
      packed(pr) = reinterpret_cast<uint32_t const&>(p2);
    }
    uint32_t const a = max_u16x2(max_u16x2(packed(0), packed(1)), packed(2));
    uint32_t const b = max_u16x2(max_u16x2(packed(3), packed(4)), packed(5));
    uint32_t const c = max_u16x2(max_u16x2(packed(6), packed(7)), kPBoundPacked);
    return max_u16x2(max_u16x2(a, b), c);
  };
  // Move each row's reference to the chunk row max where it exceeds it by more than kTau; rescale = 2^(new - old).
  auto advance_reference = [&](AccS& acc, auto half_c, float (&rescale)[kAccRows]) __attribute__((always_inline))
  {
    constexpr int hh = decltype(half_c)::value;
    Tensor s_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
#pragma unroll
    for (int mi = 0; mi < kAccRows; ++mi)
    {
      float m = fmaxf(
        fmaxf(fmaxf(s_rc(mi, 0), s_rc(mi, 1)), fmaxf(s_rc(mi, 2), s_rc(mi, 3))),
        fmaxf(fmaxf(s_rc(mi, 4), s_rc(mi, 5)), fmaxf(s_rc(mi, 6), s_rc(mi, 7))));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
      float const neg_max_new = fmaf(m, scale_log2, neg_max[hh][mi]) > kTau ? -m * scale_log2 : neg_max[hh][mi];
      rescale[mi] = ex2_approx(neg_max_new - neg_max[hh][mi]);
      neg_max[hh][mi] = neg_max_new;
    }
  };
  auto redo_chunk = [&](AccS& acc, AccP& tP, auto half_c) __attribute__((always_inline))
  {
    constexpr int hh = decltype(half_c)::value;
    float rescale[kAccRows];
    advance_reference(acc, half_c, rescale);
    warpgroup_fence_operand(acc_o[hh]);
    warpgroup_fence_operand(acc_l[hh]);
    Tensor o_rc = make_tensor(acc_o[hh].data(), flash::convert_layout_acc_rowcol(acc_o[hh].layout()));
    Tensor l_rc = make_tensor(acc_l[hh].data(), flash::convert_layout_acc_rowcol(acc_l[hh].layout()));
#pragma unroll
    for (int mi = 0; mi < kAccRows; ++mi)
    {
#pragma unroll
      for (int ni = 0; ni < kAccColsO; ++ni)
      {
        o_rc(mi, ni) *= rescale[mi];
      }
      l_rc(mi, 0) *= rescale[mi]; // column 1 duplicates column 0 and is never read
    }
    warpgroup_fence_operand(acc_o[hh]);
    warpgroup_fence_operand(acc_l[hh]);
    exp_pack(acc, tP, half_c);
  };
  // A half's first chunk advances the reference before exponentiating; O and l are still zero.
  auto exp_pack_first = [&](AccS& acc, AccP& tP, auto half_c) __attribute__((always_inline))
  {
    float rescale[kAccRows];
    advance_reference(acc, half_c, rescale);
    exp_pack(acc, tP, half_c);
  };
  // leader: this warp's elected lane; false in every lane for a chunk past the stream end (no arrival)
  auto release_bias = [&](int m, bool leader) __attribute__((always_inline))
  {
    asm volatile("" ::: "memory");
    __syncwarp();
    pipe_bias.release(m, leader);
  };

  // Retire every wgmma: ptxas tracks them only in straight-line code, so none may be in flight across a branch or join.
  auto drain = [&]() __attribute__((always_inline))
  {
    warpgroup_wait<0>();
    warpgroup_fence_operand(acc_o[0]);
    warpgroup_fence_operand(acc_o[1]);
    warpgroup_fence_operand(acc_l[0]);
    warpgroup_fence_operand(acc_l[1]);
    warpgroup_fence_operand(acc_s[0]);
    warpgroup_fence_operand(acc_s[1]);
    warpgroup_fence_operand(acc_s[2]);
    warpgroup_fence_operand(acc_s[3]);
    warpgroup_fence_operand(p_regs[0]);
    warpgroup_fence_operand(p_regs[1]);
  };

  // Chunk k is query half k % kHalves of key column (k / kHalves) % 4 of key tile k / (4 kHalves). It accumulates into
  // set k & 1, so E(k) only rescales a set whose last PV(k - 2) retired; a one-half stream merges the sets afterwards.
  struct PeriodState
  {
    int j;
    int kv_stage;
    int kv_stage_next;
    uint32_t kv_phase_next;
    uint32_t bias_phase;
  };

  auto stream = [&](auto halves_c) __attribute__((always_inline))
  {
    constexpr int kHalves = decltype(halves_c)::value, kChunksPerTile = 4 * kHalves;
    auto half_of = [](int e) constexpr { return e % kHalves; };
    auto column_of = [](int e) constexpr { return (e / kHalves) & 3; };
    auto tile_of = [](int e) constexpr { return e / kChunksPerTile; };
    int const n_chunks = kHalves * n_key_cols;
    // Free chunk e's bias half slot; a one-half stream frees both halves (the producer fills the second with no data).
    auto release_chunk = [&](auto e_c, bool leader) __attribute__((always_inline))
    {
      constexpr int e = decltype(e_c)::value;
      if constexpr (kHalves == 2)
      {
        release_bias(e & 7, leader);
      }
      else
      {
        release_bias(2 * column_of(e), leader);
        pipe_bias.release(2 * column_of(e) + 1, leader);
      }
    };

    // Body of chunk k; its wait<1> retires QK(k) and PV(k - 2). Past the stream end, QK and the score init read stale
    // shared memory; their results are never used.
    auto body = [&](
                  auto dd_c,
                  auto masked_c,
                  PeriodState const& state,
                  KOperand const& tKc,
                  VOperand const& tVc,
                  KOperand const& tKn,
                  VOperand const& tVn) __attribute__((always_inline))
    {
      constexpr int dd = decltype(dd_c)::value, e = dd + 2;
      constexpr int set_k = e & 1;
      constexpr int e_qk = e + 2, half_qk = half_of(e_qk), column_qk = column_of(e_qk), tile_qk = tile_of(e_qk);
      constexpr int e_init = e + 3, half_init = half_of(e_init), column_init = column_of(e_init),
                    tile_init = tile_of(e_init);
      constexpr int e_pv = e - 1, set_pv = e_pv & 1, column_pv = column_of(e_pv), tile_pv = tile_of(e_pv);
      constexpr int s_k = e & 3, s_qk = e_qk & 3, s_init = e_init & 3;
      constexpr int p_k = e & 1, p_pv = e_pv & 1;
      if constexpr (dd != 0)
      {
        warpgroup_wait<1>();
      }
      warpgroup_fence_operand(acc_s[s_k]);
      warpgroup_fence_operand(p_regs[p_k]);
      int const key_tile_qk = state.j + tile_qk, key_tile_init = state.j + tile_init;
      if constexpr (dd == kChunksPerTile - 3)
      {
        pipe_k.release(state.kv_stage, warp_leader); // key tile j's last QK retired
      }
      if constexpr (dd == kChunksPerTile - 1)
      {
        pipe_kv.release(state.kv_stage, warp_leader); // key tile j's last PV retired
      }
      if constexpr (e_qk == kChunksPerTile)
      {
        if (key_tile_qk < n_key_tiles)
        {
          pipe_kv.wait_full(state.kv_stage_next, state.kv_phase_next);
        }
      }
      {
        VOperand const& tV_pv = (tile_pv == 0) ? tVc : tVn;
        KOperand const& tK_qk = (tile_qk == 0) ? tKc : tKn;
        issue_pv_qk(
          p_regs[p_pv], tV_pv, Int<set_pv>{}, Int<column_pv>{}, acc_s[s_qk], tK_qk, Int<half_qk>{}, Int<column_qk>{});
      }
      release_chunk(Int<e_qk>{}, warp_leader && key_tile_qk < n_key_tiles);
      uint32_t const p_max = exp_pack(acc_s[s_k], p_regs[p_k], Int<set_k>{});
      if constexpr (half_init == 0)
      {
        if (key_tile_init < n_key_tiles)
        {
          pipe_bias.wait_full(column_init, state.bias_phase ^ uint32_t(tile_init & 1));
        }
      }
      init_scores(acc_s[s_init], Int<column_init>{}, Int<half_init>{}, key_tile_init, masked_c);
      if (__any_sync(0xffffffffu, p_max != kPBoundPacked))
      {
        redo_chunk(acc_s[s_k], p_regs[p_k], Int<set_k>{});
      }
    };

    // ---- prologue: chunks 0 and 1 of key tile 0, ending drained in the state that body 0 of period 0 expects
    {
      uint32_t const kv_phase = uint32_t(ring_base >> 1) & 1u, bias_phase = uint32_t(ring_base) & 1u;
      KOperand const tK0 = make_k_operand((ring_base & 1) * kKVSetElems);
      VOperand const tV0 = make_v_operand((ring_base & 1) * kKVSetElems);
      pipe_kv.wait_full(kv_stage0 + (ring_base & 1) * kRows, kv_phase);
      pipe_bias.wait_full(0, bias_phase);
      init_scores_prologue(acc_s[0], Int<column_of(0)>{}, Int<half_of(0)>{}, 0);
      if constexpr (kHalves == 1)
      {
        pipe_bias.wait_full(1, bias_phase);
        // One-column stream: chunk 1 has no bias data, so all -inf keeps its set empty.
        if (n_key_cols < 2)
        {
#pragma unroll
          for (int v = 0; v < int(size(acc_s[1])); ++v)
          {
            acc_s[1](v) = -INFINITY;
          }
        }
        else
        {
          init_scores_prologue(acc_s[1], Int<column_of(1)>{}, Int<half_of(1)>{}, 0);
        }
      }
      else
      {
        init_scores_prologue(acc_s[1], Int<column_of(1)>{}, Int<half_of(1)>{}, 0);
      }
      issue_qk(acc_s[0], tK0, Int<half_of(0)>{}, Int<column_of(0)>{}, _0{});
      issue_qk(acc_s[1], tK0, Int<half_of(1)>{}, Int<column_of(1)>{}, _0{});
      release_chunk(Int<0>{}, warp_leader);
      release_chunk(Int<1>{}, warp_leader);
      pipe_bias.wait_full(column_of(2), bias_phase);
      init_scores_prologue(acc_s[2], Int<column_of(2)>{}, Int<half_of(2)>{}, 0);
      if constexpr (kHalves == 1)
      {
        pipe_bias.wait_full(3, bias_phase);
      }
      init_scores_prologue(acc_s[3], Int<column_of(3)>{}, Int<half_of(3)>{}, 0);
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc_s[0]);
      warpgroup_fence_operand(acc_s[1]);
      issue_qk(acc_s[2], tK0, Int<half_of(2)>{}, Int<column_of(2)>{}, _0{});
      issue_qk(acc_s[3], tK0, Int<half_of(3)>{}, Int<column_of(3)>{}, _0{});
      release_chunk(Int<2>{}, warp_leader);
      release_chunk(Int<3>{}, warp_leader);
      exp_pack_first(acc_s[0], p_regs[0], _0{});
      issue_pv(p_regs[0], tV0, _0{}, Int<column_of(0)>{}, _0{});
      exp_pack_first(acc_s[1], p_regs[1], _1{});
      if (tile_of(4) < n_key_tiles)
      {
        pipe_bias.wait_full(column_of(4), bias_phase ^ uint32_t(tile_of(4) & 1));
      }
      init_scores_prologue(acc_s[0], Int<column_of(4)>{}, Int<half_of(4)>{}, tile_of(4));
    }
    drain();

    // ---- period j: the last chunks of key tile j and the first two of key tile j + 1, drained at its end
    auto period = [&](auto masked_c, int j) __attribute__((always_inline)) -> bool
    { // -> the stream ended in this period
      int const ring_pos = ring_base + j;
      PeriodState state;
      state.j = j;
      state.kv_stage = kv_stage0 + (ring_pos & 1) * kRows;
      state.kv_stage_next = kv_stage0 + ((ring_pos + 1) & 1) * kRows;
      state.kv_phase_next = uint32_t((ring_pos + 1) >> 1) & 1u;
      state.bias_phase = uint32_t(ring_pos) & 1u;
      KOperand const tKc = make_k_operand((ring_pos & 1) * kKVSetElems);
      VOperand const tVc = make_v_operand((ring_pos & 1) * kKVSetElems);
      KOperand const tKn = make_k_operand(((ring_pos + 1) & 1) * kKVSetElems);
      VOperand const tVn = make_v_operand(((ring_pos + 1) & 1) * kKVSetElems);
      int const last_body = n_chunks - 3 - kChunksPerTile * j;
      // Every path out of a period is drained: the period's last body drains whether the stream ends there or not.
      auto step = [&](auto dd_c) __attribute__((always_inline)) -> bool
      {
        constexpr int dd = decltype(dd_c)::value;
        constexpr bool may_end = half_of(dd + 2) == kHalves - 1;
        body(Int<dd>{}, masked_c, state, tKc, tVc, tKn, tVn);
        if constexpr (dd == kChunksPerTile - 1)
        {
          drain();
          return may_end && last_body == dd;
        }
        else if constexpr (may_end)
        {
          if (last_body == dd)
          {
            drain();
            return true;
          }
        }
        return false;
      };
      if constexpr (kChunksPerTile == 8)
      {
        return step(Int<0>{}) || step(Int<1>{}) || step(Int<2>{}) || step(Int<3>{}) || step(Int<4>{})
          || step(Int<5>{}) || step(Int<6>{}) || step(Int<7>{});
      }
      else
      {
        return step(Int<0>{}) || step(Int<1>{}) || step(Int<2>{}) || step(Int<3>{});
      }
    };
    if (n_chunks > 2)
    {
      // One-half streams run masked periods only (smaller code); two-half streams from the first masked key tile on.
      int const first_masked_period
        = kHalves == 1 ? 0 : (mask_from == INT_MAX ? INT_MAX : max(mask_from / kBlockN - 1, 0));
#pragma unroll 1
      for (int j = 0;; ++j)
      {
        bool ended;
        if constexpr (kHalves == 1)
        {
          ended = period(cute::true_type{}, j);
        }
        else
        {
          ended = (j >= first_masked_period) ? period(cute::true_type{}, j) : period(cute::false_type{}, j);
        }
        if (ended)
        {
          break;
        }
      }
    }

    // ---- tail: PV of the last chunk, then release every K, V and bias slot the stream did not, so every key tile
    // completes the same barrier protocol.
    int const last_chunk = n_chunks - 1, last_tile = n_key_tiles - 1, last_column = (last_chunk / kHalves) & 3;
    int const last_ring_pos = ring_base + last_tile;
    if (last_chunk > 0)
    {
      VOperand const tVl = make_v_operand((last_ring_pos & 1) * kKVSetElems);
      // the set of the last chunk, by its column
      constexpr int s0 = kHalves == 2 ? 1 : 0, s1 = 1, s2 = kHalves == 2 ? 1 : 0, s3 = 1;
      // each case drains before the paths join
      switch (last_column)
      {
      case 0:
        issue_pv(p_regs[s0], tVl, Int<s0>{}, _0{}, _0{});
        drain();
        break;
      case 1:
        issue_pv(p_regs[s1], tVl, Int<s1>{}, _1{}, _0{});
        drain();
        break;
      case 2:
        issue_pv(p_regs[s2], tVl, Int<s2>{}, _2{}, _0{});
        drain();
        break;
      default:
        issue_pv(p_regs[s3], tVl, Int<s3>{}, _3{}, _0{});
        drain();
        break;
      }
    }
    int const last_tile_first_chunk = kChunksPerTile * last_tile;
    __syncwarp();
    if (last_chunk != last_tile_first_chunk + kChunksPerTile - 1)
    {
      pipe_k.release(kv_stage0 + (last_ring_pos & 1) * kRows, warp_leader);
    }
    if (last_tile > 0 && last_chunk < last_tile_first_chunk + 1)
    {
      pipe_kv.release(kv_stage0 + ((last_ring_pos - 1) & 1) * kRows, warp_leader);
    }
    pipe_kv.release(kv_stage0 + (last_ring_pos & 1) * kRows, warp_leader);
#pragma unroll 1
    for (int k = max(last_chunk + 3, 4); k < last_tile_first_chunk + kChunksPerTile; ++k)
    {
      if constexpr (kHalves == 2)
      {
        pipe_bias.release(k & 7, warp_leader);
      }
      else
      {
        pipe_bias.release(2 * (k & 3), warp_leader);
        pipe_bias.release(2 * (k & 3) + 1, warp_leader);
      }
    }
  };

  // ================================================ WORK-TILE LOOP ==================================================
#pragma unroll 1
  for (int n = 0;; ++n)
  {
    int const q_buf = n & 1;
    shared.q_full[q_buf].wait((n >> 1) & 1);
    asm volatile("" ::: "memory");
    tile = published_tile(q_buf);
    if (tile.b < 0)
    {
      break;
    }
    i = tile.i0 + mma_wg;
    int length[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r)
    {
      length[r] = shared.tile_length[q_buf][r];
    }
    n_key_tiles = count_key_tiles(length);
    n_key_cols = count_key_columns(length);
    row_length = (mma_wg == 0) ? length[0] : ((mma_wg == 1) ? length[1] : length[2]);
    // Rows of the staged length are masked by the staged -inf bias, and zero-length rows output zeros.
    mask_from = (row_length == 0 || row_length == shared.tile_staged_length[q_buf]) ? INT_MAX : row_length;
    if (n_key_tiles > 0)
    {
      tQ = make_q_operand(q_buf);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
      {
        clear(acc_o[hh]);
        clear(acc_l[hh]);
        warpgroup_fence_operand(acc_o[hh]);
        warpgroup_fence_operand(acc_l[hh]); // pin the zeroing here
#pragma unroll
        for (int mi = 0; mi < kAccRows; ++mi)
        {
          neg_max[hh][mi] = neg_max_init;
        }
      }
      if (second_half_empty(tile))
      {
        stream(Int<1>{});
        // Merge set 1 into set 0 at the larger reference: both factors are 2^(<= 0), and an empty set adds nothing.
        Tensor o0 = make_tensor(acc_o[0].data(), flash::convert_layout_acc_rowcol(acc_o[0].layout()));
        Tensor o1 = make_tensor(acc_o[1].data(), flash::convert_layout_acc_rowcol(acc_o[1].layout()));
        Tensor l0 = make_tensor(acc_l[0].data(), flash::convert_layout_acc_rowcol(acc_l[0].layout()));
        Tensor l1 = make_tensor(acc_l[1].data(), flash::convert_layout_acc_rowcol(acc_l[1].layout()));
#pragma unroll
        for (int mi = 0; mi < kAccRows; ++mi)
        {
          float const neg_max_ref = fminf(neg_max[0][mi], neg_max[1][mi]);
          float const f0 = ex2_approx(neg_max_ref - neg_max[0][mi]), f1 = ex2_approx(neg_max_ref - neg_max[1][mi]);
#pragma unroll
          for (int ni = 0; ni < kAccColsO; ++ni)
          {
            o0(mi, ni) = o0(mi, ni) * f0 + o1(mi, ni) * f1;
          }
          l0(mi, 0) = l0(mi, 0) * f0 + l1(mi, 0) * f1;
          neg_max[0][mi] = neg_max_ref;
        }
      }
      else
      {
        stream(Int<2>{});
      }
      ring_base += n_key_tiles;
    }
    stage_output(q_buf);
  }
}

} // namespace bioir::claude_kit_triangle_attention_sm90_d32::detail

#endif // BIOIR_CPP_KERNELS_CLAUDE_KIT_TRIANGLE_ATTENTION_SM90_D32_KERNEL_CUH_
