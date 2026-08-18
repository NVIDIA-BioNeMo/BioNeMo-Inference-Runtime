<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Support matrix

What BioNeMo Inference Runtime (BioIR) runs, on which GPUs, and which fused
kernels it uses. How to construct a model or call `build_processor` is in
[`api.md`](api.md).

Model keys are the strings in
`bionemo_ir.hubs.FoldingSupportMatrix`. Pass them as
`EngineProcessorConfig.model_source` or as `model_name=` on the constructor.

## Models and data pipeline

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
| `protenix-v2`                                                              | `Protenix`                    | **No**                       | same as Boltz-1                                                                 |
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

Optimized CuTeDSL kernels cover Ampere through Hopper, plus SM100 / SM103 for
pair-weighted averaging, outer-product mean and AdaLN; the kernel table below
lists the exact SMs per kernel. On **SM100**, **SM103**, **SM120**, and
**SM121**, triangle attention and dual GEMM fall back to cuEquivariance. The
PyTorch backend still *runs* on any of these SKUs; rows below describe which
fused kernels apply.

| Architecture       | Compute capability | Datacenter / client GPUs in scope | Optimized kernels                                                               |
| ------------------ | ------------------ | --------------------------------- | ------------------------------------------------------------------------------- |
| **Ampere**         | SM80               | A100, A40, A30                    | CuTeDSL CUBIN                                                                   |
| **Ampere (GA10x)** | SM86               | A10, A16, RTX A6000               | CuTeDSL CUBIN                                                                   |
| **Ada Lovelace**   | SM89               | L40, L40S                         | CuTeDSL CUBIN                                                                   |
| **Hopper**         | SM90               | H100, H200, GH200                 | CuTeDSL CUBIN                                                                   |
| **Blackwell**      | SM100              | B100, B200, GB200                 | CuTeDSL CUBIN (PWA, OPM, AdaLN); cuEquivariance (triangle attention, dual GEMM) |
| **Blackwell**      | SM103              | B300, GB300                       | CuTeDSL CUBIN (PWA, OPM, AdaLN); cuEquivariance (triangle attention, dual GEMM) |
| **Blackwell**      | SM120              | RTX 5090, RTX 6000 Blackwell      | cuEquivariance only (triangle attention, dual GEMM)                             |
| **Blackwell**      | SM121              | DGX Spark (GB10)                  | cuEquivariance only (triangle attention, dual GEMM)                             |

Check the device:

```bash
python -c 'import torch; print(torch.cuda.get_device_capability())'
```

`(8, 0)` is A100 / Ampere, `(8, 9)` is L40 / Ada, `(9, 0)` is H100 / Hopper,
`(10, 0)` is B200 / SM100, `(10, 3)` is B300 / SM103, `(12, 0)` is SM120,
`(12, 1)` is DGX Spark / SM121.

On Boltz-1, Boltz-2, OpenFold3, and Protenix (`protenix-v2`) the diffusion
**module** (including the token transformer) can be captured as a CUDA graph
and replayed across sampling steps (largest win on short sequences).
OpenFold2 / AlphaFold2 have no CUDA-graph module.
How to enable it:
[`api.md` CUDA graphs](api.md#cuda-graphs-boltz-12-openfold3-protenix).

## Fused kernels

`get_pretrained_config` selects CuTeDSL triangle and pairwise attention on
SM80 / SM86 / SM89 / SM90 for fp16 / bf16. On SM100, SM103, SM120, and
SM121, triangle attention and dual GEMM use **cuEquivariance** (no CuTeDSL
CUBIN). Other SKUs or fp32 fall back to cuEquivariance (triangle attention,
if installed) or PyTorch SDPA.

Call through the dispatchers below (`bionemo_ir._torch.attention_backend`
and `bionemo_ir._torch.custom_ops`). Layers wrap the same ops:
`TriangleAttention` / `AttentionPairBias`, `TriangleMultiplicationNode`,
`PairWeightedAveraging`, `OuterProductMean`, `AdaLN`, `LNProjMoveaxisPad`,
`Transition`.

| Kernel                         | Used for                              | Implementation                       | SM (optimized)                                | Calling interface                                                                                                                                                                           |
| ------------------------------ | ------------------------------------- | ------------------------------------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Triangle attention             | Pairformer / Evoformer triangle attn  | CuTeDSL → CUBIN; else CUEQUIV / SDPA | 80, 86, 89, 90; CUEQUIV on 100, 103, 120, 121 | [`create_attention`](../../bionemo_ir/_torch/attention_backend/utils.py) (`AttentionType.TRIANGLE`)                                                                                         |
| Pairwise attention             | Token / atom attention with pair bias | CuTeDSL → CUBIN; else SDPA           | 80, 86, 89, 90                                | [`create_attention`](../../bionemo_ir/_torch/attention_backend/utils.py) (`AttentionType.PAIRWISE`)                                                                                         |
| Dual-GEMM `x_x` / `x0_x1`      | Triangle multiplication               | CuTeDSL → CUBIN; else CUEQUIV        | 80, 86, 89, 90; CUEQUIV on 100, 103, 120, 121 | [`get_dual_gemm_x_x_op`](../../bionemo_ir/_torch/custom_ops/dual_gemm_x_x/ops.py) / [`get_dual_gemm_x0_x1_op`](../../bionemo_ir/_torch/custom_ops/dual_gemm_x0_x1/ops.py)                   |
| Pair-weighted averaging (PWA)  | Pair → single update                  | CuTeDSL → CUBIN                      | 80, 90, 100, 103                              | [`get_pair_weighted_averaging_op`](../../bionemo_ir/_torch/custom_ops/pair_weighted_averaging/ops.py)                                                                                       |
| Outer-product mean (OPM)       | Single → pair update                  | CuTeDSL → CUBIN                      | 80, 86, 89, 90, 100, 103                      | [`get_outer_product_mean_op`](../../bionemo_ir/_torch/custom_ops/outer_product_mean/ops.py)                                                                                                 |
| Gated sigmoid                  | Attention output gate                 | CuTeDSL → CUBIN                      | 80, 86, 89, 90                                | [`get_gated_sigmoid_op`](../../bionemo_ir/_torch/custom_ops/gated_sigmoid/ops.py)                                                                                                           |
| AdaLN (LayerNorm + sigmoid)    | Diffusion adaptive LayerNorm          | CuTeDSL → CUBIN                      | 80, 86, 89, 90, 100, 103                      | [`get_adaln_layernorm_sigmoid_op`](../../bionemo_ir/_torch/custom_ops/adaln_layernorm_sigmoid/ops.py)                                                                                       |
| Fused LN + proj + moveaxis/pad | Pair-bias LayerNorm + linear + layout | Triton                               | all CUDA                                      | [`LNProjMoveaxisPad`](../../bionemo_ir/_torch/custom_ops/fused_ln_proj_moveaxis_pad.py) / [`fused_ln_proj_moveaxis_pad`](../../bionemo_ir/dsl_kernels/triton/fused_ln_proj_moveaxis_pad.py) |
| Fused SwiGLU                   | Transition / FFN                      | Triton                               | all CUDA                                      | [`FusedSwiGLU`](../../bionemo_ir/dsl_kernels/triton/fused_swiglu.py) / [`fused_swiglu`](../../bionemo_ir/dsl_kernels/triton/fused_swiglu.py)                                                |

Override backends on a config if you need a reference path, for example
`config.trunk.set_triangle_attention_backend("SDPA")`.

**CuTeDSL source is not open-sourced, and there is no plan to publish it.**
The public tree and wheels ship the Python callables plus precompiled CUBIN
payloads for every CuTeDSL kernel in the table above. They do not ship the
CuTeDSL kernel implementations used to generate those CUBINs.
Loading a CUBIN skips CuTeDSL JIT (`cute.compile`), so those kernels run at
full speed on the first iterations — no kernel-JIT warmup is required to hide
compile latency. If a CUBIN is missing for the current SM / dtype, those ops
fall back to PyTorch (SDPA or a vanilla reference) rather than JIT-compiling
CuTeDSL. Triton fused ops in `bionemo_ir.dsl_kernels.triton` remain
ordinary Python source and still JIT on first use.
