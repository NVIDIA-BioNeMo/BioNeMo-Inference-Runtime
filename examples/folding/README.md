# Folding

Run supported folding models through TRT-BioNeMo's public `build_processor`
pipeline. The pipeline handles parsing, tokenization, feature generation,
inference, and PDB/CIF writing without an OSS model data module.

The default command runs Boltz-2 on the bundled T1031 sample:

```bash
python examples/folding/run_demo.py
```

Select another model or sample with command-line options:

```bash
python examples/folding/run_demo.py \
    --model-source openfold3 \
    --input examples/data/samples/monomers/T1031.json \
    --output-dir output \
    --sampling-steps 50
```

Supported `model-source` values are defined in
`tensorrt_bionemo/hubs/support_matrix.py`, including `boltz-1`, `boltz-2`,
`openfold3`, OpenFold2 variants, and AlphaFold2 variants.

See `python examples/folding/run_demo.py --help` for all options.
