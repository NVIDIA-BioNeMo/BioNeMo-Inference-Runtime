---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# Design

RFProteína design through BioIR. The example builds the feature batch itself
and calls `RFProteina`. A checkpoint path is the only input. The designed
protein sequence is printed with the wall time of one call.

`--steps` replaces the checkpoint's sampling schedule with that many
single-recycle steps. Omit it to use the schedule built into the model.

Unconditional chain of 128 residues:

```bash
python examples/design/rfproteina.py \
    --flow "$RFPROTEINA_LDM_CKPT" \
    --autoencoder "$RFPROTEINA_AE_CKPT" \
    --length 128 unconditional
```

Indexed motif. A 16-residue helical CA span is fixed, and the rest of the
chain is designed:

```bash
python examples/design/rfproteina.py \
    --flow "$RFPROTEINA_LDM_CKPT" \
    --autoencoder "$RFPROTEINA_AE_CKPT" \
    --length 128 motif --motif-start 48 --motif-length 16
```

PPI binder. The target is a fixed helix, eight of its residues are hotspots,
and `--length` is the designed binder:

```bash
python examples/design/rfproteina.py \
    --flow "$RFPROTEINA_LDM_CKPT" \
    --autoencoder "$RFPROTEINA_AE_CKPT" \
    --length 64 ppi --target-length 48 --hotspot-residues 8
```

Small-molecule binder. Eight ligand atoms are a fixed pose on their own
chain, and the protein is designed around them:

```bash
python examples/design/rfproteina.py \
    --flow "$RFPROTEINA_LDM_CKPT" \
    --autoencoder "$RFPROTEINA_AE_CKPT" \
    --length 80 small-molecule --ligand-atoms 8
```

Nucleic-acid binder. The DNA sequence is fixed. `--structure` also fixes
the DNA coordinates:

```bash
python examples/design/rfproteina.py \
    --flow "$RFPROTEINA_LDM_CKPT" \
    --autoencoder "$RFPROTEINA_AE_CKPT" \
    --length 64 nucleic-acid --dna-length 16 --structure
```

Partial redesign. An existing all-alanine helix is encoded and redesigned
from partial time 0.6 on both the backbone and the latents:

```bash
python examples/design/rfproteina.py \
    --flow "$RFPROTEINA_LDM_CKPT" \
    --autoencoder "$RFPROTEINA_AE_CKPT" \
    --length 64 partial-redesign --partial-t 0.6
```
