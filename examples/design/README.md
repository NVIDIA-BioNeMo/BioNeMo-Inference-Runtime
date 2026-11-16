---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# RFProteína numerical example

Supply a toy state and explicit ODE grids, then decode numerical predictions:

```bash
python examples/design/rfproteina.py \
    --flow "$RFPROTEINA_LDM_CKPT" \
    --autoencoder "$RFPROTEINA_AE_CKPT" \
    --length 128 --seed 42
```

The example prints the output tensor shapes. `--steps` selects that many
single-recycle steps (default four). This illustrates the numerical API; it
does not choose a production generation policy.

A host application provides structure assembly, motif scaffolding, binders,
partial redesign, inverse folding and search, with the same workflows on native
and BioIR execution. See the
[numerical API guide](../../docs/advanced/design-with-rfproteina.md).
