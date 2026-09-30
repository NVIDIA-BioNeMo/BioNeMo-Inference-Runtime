---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# Module Onboarding

This directory holds a worked example that replaces a module of an existing
PyTorch model with a BioNeMo Inference Runtime (BioIR) module. The
`rf3_pairformer/` example swaps the Pairformer stack of
[RoseTTAFold3 (RF3)][rf3], an open-source (BSD-3-Clause) all-atom structure
model, for a `PairformerModule`. The example reproduces only upstream
parameter and module names.

The example has four files:

- `convert/convert_weights.py` — maps RF3 checkpoint keys to the BioIR
  layout, including fused projections.
- `integration/config.py` — builds the BioIR config from RF3
  hyperparameters.
- `integration/adapter.py` — takes RF3's call signature and translates
  masks, shapes, and dtypes.
- `integration/swap.py` — builds the BioIR module, loads converted weights,
  and replaces the RF3 submodule in place.

`swap.py` imports the converter as `convert.convert_weights`. From the
repository root, put the example directory on `PYTHONPATH`:

```bash
export PYTHONPATH=examples/module_onboarding/rf3_pairformer:$PYTHONPATH
```

For the walkthrough, including how to test and benchmark a swap, refer to
[Accelerate a Custom Model with BioIR Modules][guide].

[guide]: ../../docs/advanced/accelerate-custom-model.md
[rf3]: https://github.com/RosettaCommons/foundry/tree/production/models/rf3
