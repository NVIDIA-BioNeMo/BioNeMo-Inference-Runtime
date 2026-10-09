---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# BioIR Support Matrix

What BioNeMo Inference Runtime (BioIR) runs, on which GPUs, and which fused
kernels it uses. How to construct a model or call `build_processor` is in
[Python API](api.md).

Model keys are the strings in
`bionemo_ir.hubs.FoldingSupportMatrix`. Pass them as
`EngineProcessorConfig.model_source` or as `model_name=` on the constructor.

## Models and Data Pipeline

Every model below runs on the **optimized PyTorch backend**.

Distributed execution in `build_processor` is **replica mode** only: each
engine worker holds a full model copy on a single GPU and folds independent
inputs (`ParallelismMode.REPLICA`;
[`api.md` Ray replicas](api.md#ray-multi-gpu-replicas)). Splitting one
forward pass across GPUs (model / context parallelism, as in
[Fold-CP](https://github.com/NVIDIA-Digital-Bio/boltz-cp)) is planned, not
available yet.

| Model key                                                                  | `nn.Module`                   | Pipeline (`build_processor`) | Default `runtime_args`                                                          |
| -------------------------------------------------------------------------- | ----------------------------- | ---------------------------- | ------------------------------------------------------------------------------- |
| `alphafold2_1` … `alphafold2_5`                                            | `OpenFold2`                   | Yes                          | `{}` — recycle count is the feature axis (`max_recycling_iters + 1`, default 4) |
| `alphafold2_multimer_1` … `_5`                                             | `OpenFold2` (multimer config) | Yes                          | `{}`                                                                            |
| `openfold2_finetuning_2` … `_5`, `openfold2_ptm_*`, `openfold2_no_templ_*` | `OpenFold2`                   | Yes                          | `{}`                                                                            |
| `boltz-1`                                                                  | `Boltz1`                      | Yes                          | `{recycling_steps: 3, num_sampling_steps: 200, diffusion_samples: 1}`           |
| `boltz-2`                                                                  | `Boltz2`                      | Yes                          | same as Boltz-1                                                                 |
| `openfold3`                                                                | `OpenFold3`                   | Yes                          | same Boltz-style names ([`api.md` runtime args](api.md#runtime-args))           |
| `protenix-v2`                                                              | `Protenix`                    | **No**                       | — (no pipeline)                                                                 |
| `boltz-2-affinity`                                                         | `Boltz2Affinity`              | **No**                       | —                                                                               |

`protenix-v2` and `boltz-2-affinity` have an `nn.Module` but no tokenizer,
feature factory, or post-processor. Do not pass them to `build_processor`.
Ligand *structure* prediction on Boltz-1/2 and OpenFold3 is supported;
affinity prediction is not.

Input coverage for the bundled data pipeline (`InputRequest` → parser →
writer). "Supported" means the pipeline accepts the input and produces a
structure. "—" means the mode does not apply to that family.

| Model                | Monomer | Homo- / hetero-oligomer | Unpaired MSA        | Paired MSA | Templates (caller-supplied CIF) | RNA / DNA | Ligands (CCD / SMILES) | Ligand affinity |
| -------------------- | ------- | ----------------------- | ------------------- | ---------- | ------------------------------- | --------- | ---------------------- | --------------- |
| AF2 / OF2 monomer    | Yes     | —                       | Required            | —          | Yes (protein)                   | —         | —                      | —               |
| AF2 multimer         | —       | Yes                     | Required            | Optional   | Yes (protein)                   | —         | —                      | —               |
| `boltz-1`, `boltz-2` | Yes     | Yes                     | Required on protein | Optional   | Yes (protein)                   | Yes       | Yes                    | —               |
| `openfold3`          | Yes     | Yes                     | Required on protein | Optional   | Yes (protein)                   | Yes       | Yes                    | —               |

Notes:

- Nucleic acids and ligands are Boltz-1/2 and OpenFold3 only. OpenFold2 /
  AlphaFold2 fold protein chains exclusively.
- Templates are allowed only on `polymer_type="protein"`. BioIR does
  **not** run HHsearch / HMMsearch; pass CIF (or PDB) hits you already have.
- Nucleic-acid and ligand chains carry no MSA (`msas` / `paired_msas` are
  empty).
- AF2 monomer takes a single unpaired a3m per chain. AF2 multimer accepts
  paired a3m (some sample builders require it on every chain). Boltz-1/2 and
  OpenFold3 treat paired MSAs as optional.

`Polymer.polymer_type` is `"protein"`, `"rna"`, `"dna"`, `"ccd_ligand"` (a
CCD code or `_`-joined list), or `"smiles_ligand"` (a SMILES string in
`sequence`). Schema details: [`api.md` input requests](api.md#input-requests).

## GPUs

**This table is backend compatibility, not release qualification.** It says
which fused kernels apply on each architecture; every row runs. The devices
this release is qualified on are H200, H100, A100, L40S, GB200, and GB300, with
measured results for each in [Benchmarks](benchmark.md). The
rest of the table works and is not part of that set.

Optimized CuTeDSL kernels cover Ampere through Hopper, plus SM100 / SM103 for
pair-weighted averaging, AdaLN, and both dual GEMMs; the kernel table below
lists the exact SMs per kernel. Outer-product mean runs CuTeDSL only where it
has a native kernel and Triton on every other SM80+ SKU. On **SM100** and
**SM103**, triangle attention falls back to cuEquivariance; on **SM120** and
**SM121**, triangle attention and both dual GEMMs do. The PyTorch backend
still *runs* on any of these SKUs; rows below describe which fused kernels
apply.

| Architecture       | Compute capability | Datacenter / client GPUs     | Optimized kernels                                                                              |
| ------------------ | ------------------ | ---------------------------- | ---------------------------------------------------------------------------------------------- |
| **Ampere**         | SM80               | A100, A30                    | CuTeDSL CUBIN                                                                                  |
| **Ampere (GA10x)** | SM86               | A10, A16, A40, RTX A6000     | CuTeDSL CUBIN                                                                                  |
| **Ada Lovelace**   | SM89               | L40, L40S                    | CuTeDSL CUBIN                                                                                  |
| **Hopper**         | SM90               | H100, H200, GH200            | CuTeDSL CUBIN                                                                                  |
| **Blackwell**      | SM100              | B100, B200, GB200            | CuTeDSL CUBIN (PWA, AdaLN, both dual GEMMs); Triton (OPM); cuEquivariance (triangle attention) |
| **Blackwell**      | SM103              | B300, GB300                  | CuTeDSL CUBIN (PWA, AdaLN, both dual GEMMs); Triton (OPM); cuEquivariance (triangle attention) |
| **Blackwell**      | SM120              | RTX 5090, RTX 6000 Blackwell | cuEquivariance (triangle attention, dual GEMM); Triton (OPM)                                   |
| **Blackwell**      | SM121              | DGX Spark (GB10)             | cuEquivariance (triangle attention, dual GEMM); Triton (OPM)                                   |

Check the device:

```bash
python -c 'import torch; print(torch.cuda.get_device_capability())'
```

`(8, 0)` is A100 / Ampere, `(8, 9)` is L40 / Ada, `(9, 0)` is H100 / Hopper,
`(10, 0)` is B200 / SM100, `(10, 3)` is B300 / SM103, `(12, 0)` is SM120,
`(12, 1)` is DGX Spark / SM121.

By default, for inputs up to 1024 tokens, Boltz-2, OpenFold3, and Protenix
(`protenix-v2`) capture one trunk recycle, one diffusion step, and the
confidence Pairformer as CUDA graphs; Boltz-1 captures the diffusion step;
OpenFold2 / AlphaFold2 capture one trunk recycle. Replays skip per-kernel
launch overhead (largest win on short sequences). Limits and opt-out:
[`api.md` CUDA graphs](api.md#cuda-graphs).

## Fused Kernels

`get_pretrained_config` selects the **Heuristic** triangle-attention backend and
CuTeDSL pairwise attention on SM80 / SM86 / SM89 / SM90 for fp16 / bf16.
Heuristic sends SM90 BF16 D=32 triangle attention of at least 23M attention
scores (`batch * i_dim * tokens^2 * heads`, for example 184 tokens with four
heads) to the source-visible **ClaudeKit** CUDA kernel and every other shape to
the CuTeDSL CUBIN. On SM100 and SM103, triangle attention uses
**cuEquivariance**; on SM120 and SM121, triangle attention and both dual GEMMs
do (no CuTeDSL CUBIN). Other SKUs or fp32 fall back to cuEquivariance (triangle
attention if installed) or PyTorch SDPA.

Call through the dispatchers below (`bionemo_ir._torch.attention_backend`
and `bionemo_ir._torch.custom_ops`). Layers wrap the same ops:
`TriangleAttention` / `AttentionPairBias`, `TriangleMultiplicationNode`,
`PairWeightedAveraging`, `OuterProductMean`, `AdaLN`, `LNProjMoveaxisPad`,
`Transition`, `ConditionedTransitionBlock`, `PairTransition`, `MSATransition`.

| Kernel                              | Used for                                      | Implementation                                                  | SM (optimized)                                | Calling interface                                                                                                                                                                           |
| ----------------------------------- | --------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Triangle attention                  | Pairformer / Evoformer triangle attn          | Heuristic → ClaudeKit CUDA / CuTeDSL CUBIN; else CUEQUIV / SDPA | 80, 86, 89, 90; CUEQUIV on 100, 103, 120, 121 | [`create_attention`](../../bionemo_ir/_torch/attention_backend/utils.py) (`AttentionType.TRIANGLE`)                                                                                         |
| Pairwise attention                  | Token / atom attention with pair bias         | CuTeDSL → CUBIN; else SDPA                                      | 80, 86, 89, 90                                | [`create_attention`](../../bionemo_ir/_torch/attention_backend/utils.py) (`AttentionType.PAIRWISE`)                                                                                         |
| Dual-GEMM `x_x`                     | Triangle multiplication; SwiGLU projection    | CuTeDSL → CUBIN; else CUEQUIV / PyTorch                         | 80, 86, 89, 90, 100, 103; CUEQUIV on 120, 121 | [`get_dual_gemm_x_x_op`](../../bionemo_ir/_torch/custom_ops/dual_gemm_x_x/ops.py) / [`get_cute_dual_gemm_x_x_op`](../../bionemo_ir/_torch/custom_ops/dual_gemm_x_x/ops.py)                  |
| Dual-GEMM `x0_x1`                   | Triangle multiplication                       | CuTeDSL → CUBIN; else CUEQUIV                                   | 80, 86, 89, 90, 100, 103; CUEQUIV on 120, 121 | [`get_dual_gemm_x0_x1_op`](../../bionemo_ir/_torch/custom_ops/dual_gemm_x0_x1/ops.py)                                                                                                       |
| TriMul KF chain                     | Triangle multiplication on padded tokens      | CuTeDSL → CUBIN (bf16)                                          | 90                                            | [`get_trimul_kf_k1_op`](../../bionemo_ir/_torch/custom_ops/trimul_kf_k1/ops.py) → `get_trimul_kf_k2_op` → `get_trimul_kf_k3_op`                                                             |
| Pair-weighted averaging (PWA)       | Pair → single update                          | CuTeDSL → CUBIN                                                 | 80, 90, 100, 103                              | [`get_pair_weighted_averaging_op`](../../bionemo_ir/_torch/custom_ops/pair_weighted_averaging/ops.py)                                                                                       |
| Outer-product mean (OPM)            | Single → pair update                          | CuTeDSL → CUBIN on native SMs; else Triton / PyTorch            | 80, 86, 89, 90 (bf16); Triton on other 80+    | [`get_outer_product_mean_op`](../../bionemo_ir/_torch/custom_ops/outer_product_mean/ops.py)                                                                                                 |
| Gated sigmoid                       | Attention output gate                         | CuTeDSL → CUBIN                                                 | 80, 86, 89, 90                                | [`get_gated_sigmoid_op`](../../bionemo_ir/_torch/custom_ops/gated_sigmoid/ops.py)                                                                                                           |
| Attention epilogue                  | Attention gate + output projection + residual | CuTeDSL → CUBIN (bf16)                                          | 80, 86, 89, 90                                | [`get_attn_epilogue_op`](../../bionemo_ir/_torch/custom_ops/attn_epilogue/ops.py)                                                                                                           |
| AdaLN (LayerNorm/RMSNorm + sigmoid) | Diffusion adaptive normalization              | CuTeDSL → CUBIN                                                 | 80, 86, 89, 90, 100, 103                      | [`get_adaln_layernorm_sigmoid_op`](../../bionemo_ir/_torch/custom_ops/adaln_layernorm_sigmoid/ops.py)                                                                                       |
| Fused LN + proj + moveaxis/pad      | Pair-bias LayerNorm + linear + layout         | Triton                                                          | all CUDA                                      | [`LNProjMoveaxisPad`](../../bionemo_ir/_torch/custom_ops/fused_ln_proj_moveaxis_pad.py) / [`fused_ln_proj_moveaxis_pad`](../../bionemo_ir/dsl_kernels/triton/fused_ln_proj_moveaxis_pad.py) |
| Transition MLP                      | SwiGLU / ReLU transition, hidden kept on chip | CuTeDSL → CUBIN (bf16)                                          | 80, 86, 89, 90                                | [`get_transition_mlp_op`](../../bionemo_ir/_torch/custom_ops/transition_mlp/ops.py)                                                                                                         |
| Fused SwiGLU                        | Transition / FFN                              | Triton                                                          | all CUDA                                      | [`FusedSwiGLU`](../../bionemo_ir/dsl_kernels/triton/fused_swiglu.py) / [`fused_swiglu`](../../bionemo_ir/dsl_kernels/triton/fused_swiglu.py)                                                |

A transition's SwiGLU MLP takes the first path that ships for its shape: the
transition MLP; then, in `ConditionedTransitionBlock` only, the `x_x` dual
GEMM's silu gate for the gated projection; then the Triton fused SwiGLU. On
SM100 / SM103 the transition MLP does not ship, so `ConditionedTransitionBlock`
projects through the Blackwell `x_x` kernels.

`TriangleMultiplicationNode` runs the TriMul KF chain when its model pads tokens
(`enable_token_pad`, on by default) and the node is bf16 with a
`(dim, hidden_dim)` of 32/32, 64/64, 128/128, 256/256, or 384/256, without
`trimul_high_precision` or `trimul_mean_normalization`. K1 applies the input
LayerNorm and gated projections, K2 the triangle contraction, and K3 the output
LayerNorm, projection, gate, and masked residual. The Boltz, Protenix, and
OpenFold3 trunks run it in their MSA, template, and pairformer stacks; OpenFold2
in its Evoformer and extra-MSA stacks. Confidence heads, OpenFold2 templates,
and calls whose token counts are not multiples of 8 take the dual-GEMM path.

Override backends on a config if you need a reference path, for example
`config.trunk.set_triangle_attention_backend("SDPA")`.

**CuTeDSL source is not open-sourced, and there is no plan to publish it.**
The public tree and wheels ship the Python callables plus precompiled CUBIN
payloads for every CuTeDSL kernel in the table above. They do not ship the
CuTeDSL kernel implementations used to generate those CUBINs.
The Apache-2.0/BSD-3-Clause ClaudeKit D=32 SM90 source is the exception: it
ships in `cpp/kernels/claude_kit_triangle_attention_sm90_D32/` and is compiled
with the wheel's native extension.
Loading a CUBIN skips CuTeDSL JIT (`cute.compile`), so those kernels run at
full speed on the first iterations — no kernel-JIT warmup is required to hide
compile latency. If a CUBIN is missing for the current SM / dtype, those ops
fall back to PyTorch (SDPA or a vanilla reference) rather than JIT-compiling
CuTeDSL. Triton fused ops in `bionemo_ir.dsl_kernels.triton` remain
ordinary Python source and still JIT on first use.
