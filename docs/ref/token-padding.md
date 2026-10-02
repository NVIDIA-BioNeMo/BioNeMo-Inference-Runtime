---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# Token Padding

Token padding rounds a model's token count up to the next multiple of 8
before its pair stacks run, then slices the outputs back to the true count.
The added tokens are masked out, so they change only the shapes the kernels
see. Most real token counts are not multiples of 8, and the fast paths of
triangle multiplication need them to be, so padding makes the pair stacks
faster at the same accuracy. BioNeMo Inference Runtime (BioIR) pads by
default.

## Why Padding Is Faster

Triangle multiplication is a large share of every pair stack, and its fused
paths need both pair token axes to be multiples of 8:

- `TriangleMultiplicationNode` fuses the residual into the output-gate GEMM,
  writing `(x + update) * mask` from its epilogue, only on 8-aligned token
  axes. At any other count it runs the gate and the residual add as separate
  passes over the pair tensor.
- Its projection and contraction GEMMs run their fast SM90 kernels on
  8-aligned token axes. At other counts every layer either pads its own
  inputs on every call or runs slower kernels.
- Aligned shapes reuse the kernel variants the trunk already compiled. With
  confidence padding off, Boltz-2 compiled two more Triton kernels on first
  use, and its first confidence call took 6.0 s instead of 0.15 s.

Speedup from padding, measured on H100 in bf16 with CUBIN kernels and one
batch:

| Model                 | Tokens    | Triangle multiplication | 4 pairformer blocks |
| --------------------- | --------- | ----------------------- | ------------------- |
| Boltz-2 (`c_z` 128)   | 635 → 640 | 1.17x                   | 1.03x               |
| Boltz-2 (`c_z` 128)   | 766 → 768 | 1.18x                   | 1.03x               |
| Protenix (`c_z` 256)  | 635 → 640 | 2.06x                   | 1.20x               |
| Protenix (`c_z` 256)  | 766 → 768 | 2.03x                   | 1.19x               |

The confidence pairformer runs once per diffusion sample, so padding it saves
more as `diffusion_samples` grows. With 5 samples at 100 and 199 tokens,
Boltz-1's confidence module takes 708–724 ms padded against 864 ms unpadded,
1.19–1.22x faster.

## What Gets Padded

- **Boltz-1 and Boltz-2**: the trunk's MSA module, pairformer, and Boltz-2
  template module, and the confidence pairformer. Boltz-1's confidence module
  also pads its MSA module.
- **Protenix**: the trunk's template embedder, MSA module, and pairformer,
  and the confidence head's pairformer.
- **OpenFold3**: the trunk's template embedder, MSA module, and pairformer,
  and the confidence pairformer in the auxiliary heads.
- **OpenFold2**: the extra-MSA and Evoformer stacks. The template stack runs
  before padding, at the true residue count.

Each region pads once, before its recycling or layer loop, and slices its
outputs back afterwards; the slices are views, not copies. The diffusion and
structure modules and the confidence heads see the true token count.

## Accuracy

- Padding appends zero tokens after the last real one. Appending keeps every
  mask row left-aligned (`1...1 0...0`), as CuTeDSL's prefix-length masking
  requires. The pad is zeros, never uninitialized memory: epilogues multiply
  by the mask, and `inf * 0` is NaN.
- Padded tokens carry a zero mask, so attention and triangle multiplication
  ignore them. Results for real tokens change only by GEMM rounding over the
  larger shapes. With 5 diffusion samples, turning confidence padding on moved
  pTM by at most 2.2e-4 on Boltz-2 and 1.3e-3 on Boltz-1.

## Enable or Disable

Padding is on by default. Keep it on; turn it off only to compare against the
unpadded path. Each padded region has its own `enable_token_pad` flag:

- Boltz-1, Boltz-2, Protenix, and OpenFold3 read it from the region's
  `PairformerConfig`: the trunk pairformer's for the trunk, the confidence
  pairformer's for the confidence module.
- OpenFold2 reads `trunk.enable_token_pad`.

```python
from bionemo_ir.models.boltz2 import Boltz2

config = Boltz2.get_pretrained_config()
config.trunk.pairformer.enable_token_pad = False  # trunk
config.confidence_module.pairformer.enable_token_pad = False  # confidence module
model = Boltz2(config=config)
```

Boltz-1's confidence module shares the trunk's `PairformerConfig`, so one
flag covers both.

Each model's `config.py` (`bionemo_ir/models/<model>/config.py`) also
declares which tensors a region pads, in a `token_pad_spec` field. A
`TrunkPadSpec` from `bionemo_ir.configs` groups tensor names by layout
(`[..., N]`, `[..., N, C]`, `[..., N, N]`, `[..., N, N, C]`) and lists the
input-feature keys to pad, which may also be `[..., N, R, C]` frames. Change
it only when a region gains a token-dimensioned input. The pad and slice
helpers live in
[`token_padding.py`](../../bionemo_ir/_torch/layers/token_padding.py).

## Cost

Each padded region copies its padded tensors once per call, and the
confidence modules copy the pair representation once per diffusion sample.
