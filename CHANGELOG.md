---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# Changelog

Each final release of BioNeMo Inference Runtime, newest first. Release
candidates are not listed.

## 0.2.0

### Key Features and Enhancements

- **RFProteína.** BioIR runs RFProteína (RFD4) as a conditional sampler.
  One call returns a sequence and an all-atom structure from a length and
  optional conditioning.
- **Design tasks.** One model covers unconditional generation, fold-class
  and secondary-structure steering, indexed and unindexed motif
  scaffolding, protein and small-molecule binders, nucleic-acid binders,
  partial redesign, inverse folding, and metric-guided beam search.
- **Checkpoints.** Load the released velocity-network and autoencoder
  checkpoints and call `RFProteina`. Checkpoints that carry training
  metadata load.
- **Sampling schedule.** The default is the released medium policy: 60
  steps, 185 trunk passes, a denser tail on both modalities, and a decoder
  conditioned on the predicted sequence.
- **Design replay.** Sampling records one trunk recycle and replays it, so
  a shape is captured once whatever depths that schedule uses.
- **Motif coordinates.** Motif scaffolding reads internal-coordinate
  conditioning from the coordinate group. A motif supplied only as
  internal coordinates is no longer dropped.

### Performance

- **Folding replay.** For inputs up to 1024 tokens, Boltz-2, OpenFold3,
  and Protenix record one trunk recycle, one diffusion step, and the
  confidence Pairformer, and replay them on later calls of the same shape.
  Boltz-1 records the diffusion step. OpenFold2 and AlphaFold2 record one
  trunk recycle. The diffusion step extends to 2048 tokens on GPUs with at
  least 64 GiB. Larger inputs run without replay. Token counts are padded
  so nearby lengths share one recording. A serving request does not rerun
  the first recording to check it.
- **Stable replay inputs.** A replay copies only tensors that changed.
  Diffusion steps keep features, trunk outputs, and prepared biases in
  place.
- **Conditioning reuse.** Noise-independent pair and single conditioning,
  atom-pair biases, and diffusion noise levels are computed once per
  rollout. Boltz deduplicates diffusion conditioning across samples.
  Protenix reuses denoiser conditioning across samples, layers, and steps.
- **Triangle updates.** Boltz-2, Protenix, and OpenFold3 embed relative
  position in one pass. Below 512 tokens, triangle attention uses a
  narrower tile.
- **Fused triangle multiplication.** With token padding on, Boltz,
  Protenix, and OpenFold3 run triangle multiplication in the MSA,
  template, and pairformer stacks as one chain: input normalization and
  gated projections, the contraction, then output normalization,
  projection, gate, and masked residual. OpenFold2 does this in the
  Evoformer and extra-MSA stacks. RFProteína does it in the
  velocity-network pair updates. Confidence heads, OpenFold2 templates,
  and token counts that are not multiples of 8 take the dual-product
  path, including a residual add. On Hopper the output-tile weights stay
  resident while rows stream past. On Blackwell that projection also feeds
  conditioned transitions.
- **Fused attention output.** Triangle attention and pair-biased attention
  fuse the output gate, projection, and residual, including the diffusion
  adaptive output gate. A residual-free form leaves the add to the caller.
  Token transformers use it when they add the residual themselves. The
  sigmoid gate can carry the residual.
- **Fused transitions.** A transition keeps its hidden activation on chip.
  A conditioned transition that does not take that path uses the
  dual-product SwiGLU gate, then a fused SwiGLU. Adaptive normalization
  fuses the norm and the sigmoid gate.
- **Fused pair updates.** Pair-to-single averaging and single-to-pair
  outer products are fused. OpenFold2 runs the dense outer product in
  BF16. Pair-bias normalization, projection, and layout share one pass.
- **Atom transformers.** OpenFold3 and Protenix run atom attention in BF16
  inside the FP32 encoder and decoder, and gather each local atom window
  in one pass. Boltz gathers atoms by token instead of a one-hot product.
  OpenFold3 projects only the atom pairs it reads, applies the encoder
  activation inside the atom reduction, and drops the unused atom-index
  matrix.
- **Protenix trunk.** Pairformer nodes whose weights are zero are skipped.
  Triangle-bias heads are projected in the same pass as LayerNorm.
  Confidence summaries stay on the host. pLDDT weights are not rebuilt for
  every atom.
- **Longer OpenFold3 inputs.** OpenFold3 fits longer sequences with the
  same recycles, diffusion steps, samples, and confidence outputs. Static
  atom conditioning and pair conditioning stay on the device for the
  rollout. Protein conformers are prefetched.
- **Structure math.** OpenFold2 fuses inference rotation products.
  Quaternion conversion is fused. Rigid alignment reuses its solver setup.
  Diffusion noise increments are prepared once per rollout and do not
  change the random draws.
- **Less host time.** Deep MSA decoding, template binning, and the
  OpenFold2 and OpenFold3 tokenizers take less time. Boltz-2 no longer
  waits on the device at every diffusion step. A serial processor overlaps
  CPU stages with inference instead of starting a distributed runtime.
- **Pipeline confidence.** Pipelines keep the confidence scores and maps
  the writer uses and omit unused heads. Distance bins and their
  projection are fused. Existing calls keep the same arguments, and the
  written structure and confidence scores stay in place.
- **Design sampling.** RFProteína projects shared pair biases in one pass
  and conditions pair adaptive normalization once per design. The first
  pass does not recycle a zero state. Sampling does not copy results back
  to the host on every step.

### Fixed Issues

- **Ligand atom names.** Atoms outside the standard chemical vocabulary
  keep their real names and elements when a pipeline writes them.
- **SMILES ligands.** A SMILES ligand is no longer written as protein
  only, and it no longer fails at the writer with duplicate atom ids.
- **PDB templates.** OpenFold2 and OpenFold3 accept protein templates in
  PDB as well as mmCIF. Author chain ids, insertion codes, and unresolved
  residues are preserved.
- **Template chain ids.** PDB inputs use author chain ids. mmCIF inputs
  keep label chain ids. When the sequence is taken from observed residues,
  selenomethionine is mapped to methionine.
- **Protenix MSA.** MSA rows are resampled on each recycle with a fixed
  depth, so a recorded recycle still replays.
- **Protenix seed.** An omitted sampler seed follows the caller's random
  state. An explicit sampling seed still overrides it.
- **Protenix features.** Stock unbatched features are accepted, and missing
  atom windows are built inside the forward pass.
- **Protenix memory.** On GPUs with at most 24 GiB, activations are chunked
  above 1024 tokens instead of running out of memory.
- **Revised structures.** A revised RCSB entry is taken from the wwPDB
  archive when the newest RCSB file is a later remediation.
- **Downloaded archives.** Archives are checked before they are used.

The full public history for this release is on
[GitHub](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/commits/v0.2.0).

## 0.1.1 (2026-10-02)

### Fixed Issues

- **Sequence validation.** Protein, RNA, and DNA sequences with unsupported
  characters are
  [now rejected instead of being silently treated as unknown residues](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/commit/b6d02a7392e9d7d265a3e0f1d4b5d52a79c4582d).
  Ambiguous residues are normalized, and FASTA records that differ only by
  case are kept as one molecule. MSA rows with invalid characters are
  rejected.
- **OpenFold3 confidence.** pTM and ipTM
  [now score only tokens that have a valid structural frame](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/commit/86edd266133634c1ba1ca437316d39a63938ab8c).
  Scores no longer drop valid ligand frames or include polymers with missing
  backbone atoms.
- **Reproducible sampling.** A seed set on a request is
  [now applied when the model runs](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/commit/71c3fab9d0020fa4eada47b114b77fbcfaa2d8f7).
  Results no longer change based on how requests are batched. An explicit
  sampling seed still overrides the request seed.
- **Safer file access.** MSA and template paths
  [stay inside the configured input directory](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/commit/8c2db08b2f57d9650f5fa456a91f3aa99a0e1f00),
  and written outputs
  [stay inside the configured output directory](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/commit/d3505e2f60e071631af84aa374a8face44794752).
- **More reliable runs.** Model artifacts are
  [checked](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/commit/5721de02e95c24cf8027fef8be09d024418fae02)
  [before use](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/commit/6e298cb3c93d067557f7008b4992c9a0e98dde4c),
  installs are
  [more stable](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/commit/6b9b6fed1c843ab189889ad66e2531ff14f39674),
  and inference
  [performance](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/commit/c271d4d4add4e18aefd8771ab6c97072e7a2dd8c)
  and
  [runtime behavior](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/commit/baf13efb4cf15fbd2ac87351ae44a4c4134d251d)
  are improved.

The full public history for this release is on
[GitHub](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/commits/v0.1.1).

## 0.1.0 (2026-09-10)

### Key Features and Enhancements

- **First public release.** BioIR
  [runs structure-prediction models](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/commit/a768bda05e51855b70193e70310ad80a312e942c)
  as ordinary PyTorch modules through a GPU pipeline, from FASTA and MSA
  input to PDB or mmCIF output with confidence scores. Pipelines cover
  AlphaFold2, AlphaFold2-Multimer, OpenFold2, Boltz-1, Boltz-2, and
  OpenFold3.
- **Install from PyPI.** Python 3.12 wheels for Linux x86_64 and aarch64,
  built against CUDA 13.2,
  [install with `pip install bionemo-ir`](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/commit/bbf5f9ecb553a0a6707a5d37747041e0f58c81fe).
  The same wheels are attached to the GitHub release with `SHA256SUMS`.

### Compatibility

- [Release-qualified GPUs](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime/commit/a768bda05e51855b70193e70310ad80a312e942c)
  are H200, H100, A100, L40S, GB200, and GB300. The runtime requires
  NVIDIA driver 580 or newer.
