---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# Folding

Run supported folding models through BioIR's public `build_processor`
pipeline. The pipeline handles parsing, tokenization, feature generation,
inference, and PDB/CIF writing without an OSS model data module.

The default command runs Boltz-2 on the bundled T1031 sample and prints the
structure to stdout:

```bash
python examples/folding/run_demo.py > T1031.cif
```

`--output-dir` writes instead: one `<record>.cif` (or `.pdb`) and one
`<record>_scores.json` per prediction. Without it the structure goes to stdout
and the record id and scores go to stderr, so the redirect above yields a file
with nothing else in it.

Select another model or sample with command-line options:

```bash
python examples/folding/run_demo.py \
    --model-source openfold3 \
    --input examples/data/samples/monomers/T1031.json \
    --output-dir output \
    --sampling-steps 50
```

`--model-source` accepts the models the pipeline registry has a factory for
(`bionemo_ir/registry.py`) — the Boltz, OpenFold and AlphaFold2 variants, named
by their support-matrix keys. `--help` enumerates those values.

`bionemo_ir/hubs/support_matrix.py` is a wider list: it names every model the
library can load weights for, including ones with no pipeline factory or with
pipeline stages that still raise `NotImplementedError`. Those have to be driven
from Python.

The diffusion options (`--recycling-steps`, `--sampling-steps`,
`--diffusion-samples`) apply only to the diffusion models; the rest ignore them.

See `python examples/folding/run_demo.py --help` for all options.
