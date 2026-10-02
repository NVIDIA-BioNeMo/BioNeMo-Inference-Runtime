---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# Changelog

Each final release of BioNeMo Inference Runtime, newest first. Release
candidates are not listed.

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
