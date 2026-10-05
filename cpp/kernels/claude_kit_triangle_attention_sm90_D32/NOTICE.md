<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ClaudeKit SM90 D=32 triangle attention

This backend derives from the public
[`anthropics/uplifting-biomolecular-modeling`][uplift] repository at commit
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`, licensed under Apache-2.0.

The M1 kernel originated in
`common/opt_core/opt_core/kernels/triattn/triattn_native/pkg/v11/triattn_pkg/cuda_b/csrc/`:

- `m1/triattn_m1_sm90.cuh`, now `kernel.cuh`
- `m1/launch_m1.cuh`, now `kernel.cu`
- the M1 bias staging in `m1/m1_binding.cu`, now `stage_bias.cu`
- `fa3_utils.h`

BioIR modifications:

- replace M1's max-free hot pass (a fixed logit shift with no running max)
  and its SAFE recompute pass over a fix list with a single-pass online
  softmax: each row keeps a reference max that follows the exact running
  max lazily, advancing and rescaling O and the row sum whenever a chunk's
  row max exceeds it by more than 8 log2 units, as FlashAttention-4 does;
  the check runs on the packed bf16 probabilities after the exponentials;
- issue each chunk's PV together with the QK two chunks ahead, and retire
  every wgmma once per key tile instead of once per two;
- make the kernel persistent: one CTA per SM takes work tiles from a ticket
  counter, the K/V and bias rings carry over from one work tile to the next,
  Q is double-buffered, and a store warp writes the output by TMA;
- replace the general bool-mask word tables with left-aligned `actual_s_kv`
  lengths: the bias staging writes -inf past the longest row of each batch,
  so only shorter rows mask keys in the kernel;
- end a work tile's key stream at the last 32-key column holding a live key,
  and stream only the first half of a query tile whose second half lies past
  the sequence end;
- write zero output and the LSE sentinel for a zero-length row inside the
  kernel, and write the BioIR FP32 log-sum-exp output when requested;
- run the work tiles of a group of heads back to back while their staged
  bias stays L2-resident;
- read the packed in_proj Q/K/V views in place through BioIR tensor views,
  replace PyTorch-extension objects with nanobind, and build one SM90a
  instantiation through CMake.

`fa3_utils.h` contains excerpts from FlashAttention-3 by Tri Dao under
BSD-3-Clause. Its complete copyright, conditions, and disclaimer remain in
that file.

These sources need the CUTLASS and CuTe C++ headers, which the
`3rdparty/cutlass` submodule pins to 4.5.2 under BSD-3-Clause. They are a
build-time dependency and are not vendored here.

[uplift]: https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e
