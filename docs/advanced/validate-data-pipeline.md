---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# Validate a Ported Data Pipeline

A ported data pipeline is correct when it produces the same features as the
upstream pipeline and the model predicts structures as accurately as the
upstream model does. This guide collects evidence for both claims after you
[port a data pipeline][port] into BioNeMo Inference Runtime (BioIR).

Validation runs at two levels:

- **Level 1, feature equivalence.** Compare every feature tensor that BioIR
  produces against the tensor the upstream pipeline produces for the same
  input.
- **Level 2, end-to-end accuracy.** Predict structures with BioIR and with
  the upstream model under identical settings, score both against ground
  truth, and compare the scores.

Level 1 finds bugs precisely: it names the sample and the feature that
differ. Level 2 catches what features cannot show: postprocessing, writing,
and configuration errors. Pass Level 1 before you start Level 2.

[port]: port-data-pipeline.md

## Rules for Trustworthy Results

A passing test means something only when its reference is independent and
its scope is complete. Hold every validation run to these rules:

- **References come from upstream code.** Comparing BioIR with its own saved
  output only proves that it is deterministic. Scripts that produce
  references must not import `bionemo_ir.pipeline.models`.
- **One sample set covers every step.** Write the sample list once. Every
  step checks that it processed exactly that set: no missing, extra, or
  renamed samples.
- **Every reference artifact has provenance.** Record the command, the
  inputs, and the upstream version that produced each artifact.
- **Tolerances stay fixed.** Write them down before the first comparison.
  Relaxing a tolerance after a failure hides the bug that caused it. If a
  tolerance must change, record why.
- **Nothing drops out silently.** No `pytest.skip`, `xfail`, broad `except`
  clauses, dropped keys, or skipped samples. An input the model cannot handle
  is a documented limitation, not a skipped test.
- **Synthetic data stays in unit tests.** Hand-written tensors, mock outputs,
  and simplified metrics never stand in for upstream references or real
  scores.
- **Failures leave evidence.** Save the failing output and a minimal diff
  before you fix anything, and rerun the full sample set after the fix.

## Before You Begin

Complete the following tasks before you start:

- Port and register the pipeline, as described in [Port a Data
  Pipeline][port].
- Install the upstream model in its own environment, separate from BioIR, so
  their dependencies cannot conflict.
- Use one checkpoint for both models: the one BioIR resolves through
  `bionemo_ir.hubs`. Convert its format for the upstream code if needed.
- Use a GPU that runs both models.

Keep validation artifacts in a working directory outside the package:

```text
workdir/<model>/
├── envs/                          # scorer environments
├── ref_data/
│   ├── sample_manifest.json       # the sample set
│   ├── provenance.jsonl           # one line per reference artifact
│   ├── upstream_config.json       # resolved upstream inference config
│   ├── upstream_features/         # Level 1 references, <sample_id>.pt
│   ├── upstream_predictions/      # upstream structures and scores
│   └── upstream_metrics.json      # upstream baseline
├── bioir/                         # BioIR structures and scores
├── debug/                         # failing outputs and diffs
├── tests/test_equivalence.py
└── NOTES.md                       # tolerances, settings, limitations
```

Refer to the working directory and to your BioIR checkout by absolute path,
so that every command and code example runs from any directory:

```bash
export WORKDIR="$HOME/workdir/my-model"
mkdir -p "$WORKDIR"
```

The Python examples use `REPO` and `WORKDIR`, defined as follows:

```python
import os
from pathlib import Path

import bionemo_ir

REPO = Path(bionemo_ir.__file__).resolve().parents[1]  # your BioIR checkout
WORKDIR = Path(os.environ["WORKDIR"])
```

## Fix the Sample Set

List every sample in scope in `ref_data/sample_manifest.json`. For each
sample, record its ID, category, input file, ground-truth file, chains with
their lengths, and multiple sequence alignment (MSA) files. Use paths
relative to `REPO`, as in the following example:

```json
{
  "samples": [
    {
      "sample_id": "T1031",
      "category": "protein_monomer",
      "input": "examples/data/samples/monomers/T1031.json",
      "ground_truth": "examples/data/samples/gt/T1031.pdb",
      "chains": {"A1": 95},
      "msas": ["examples/data/samples/monomers/msas/T1031.a3m"]
    }
  ]
}
```

Use one category per sample: `protein_monomer`, `protein_homopolymer`,
`protein_heterooligomer`, `rna`, `dna`, `ccd_ligand`, `smiles_ligand`, or
`mixed_complex`. Cover every input type the model supports. A protein-only
sample set cannot validate a model that also handles ligands and nucleic
acids.

Every sample needs an experimental structure for Level 2.
`examples/data/samples/gt/` has one for the bundled monomers, the `6m3u`
homopolymer, the `7sfy` heterooligomer, the RNA samples, and the complexes
with chemical component dictionary (CCD) ligands. A `_with_template` sample
uses its base sample's file. The DNA complex (`templates/7r6r.json`) and the
other template samples take their names from Protein Data Bank (PDB) IDs.
Download their mmCIF files from the PDB. `smiles_demo.json` has no
experimental structure, so for the `smiles_ligand` category, write a request
that gives the ligand of a PDB complex as a simplified molecular-input
line-entry system (SMILES) string.

Run both models with the sample MSAs and templates. Without MSAs, both
models score far below their real accuracy, and the comparison says little.

Every later step checks its coverage against the manifest:

```python
import json


def assert_manifest_coverage(processed_ids, manifest_path=WORKDIR / "ref_data" / "sample_manifest.json"):
    with open(manifest_path) as f:
        expected = {sample["sample_id"] for sample in json.load(f)["samples"]}
    processed = list(processed_ids)
    assert len(processed) == len(set(processed)), "duplicate sample IDs"
    missing, extra = expected - set(processed), set(processed) - expected
    assert not missing and not extra, f"missing={sorted(missing)} extra={sorted(extra)}"
```

## Establish the Upstream Baseline

Run the upstream model on every sample and score its predictions before you
test BioIR. Without the baseline, Level 2 has nothing to compare against.

### Match the Inference Settings

Resolve the upstream inference configuration, print it, and save it as
`ref_data/upstream_config.json`. Then record in `NOTES.md` every setting
that changes the output, with the value that both runs use:

- Number of diffusion samples and diffusion steps
- Recycling iterations — check whether "3 recycles" means three or four
  passes through the trunk on each side
- Random seeds
- The rule that selects one sample from several
- Cropping or chunking of long inputs
- MSA depth, and whether each side uses MSAs at all
- Inference precision

Mismatches here produce real-looking regressions. Common mismatches include
best-of-five sampling upstream against a single sample in BioIR, 200
diffusion steps against 50, and an MSA that one side never receives. When a
setting cannot match, record the difference and its expected effect.

### Get the Upstream Model Running

The baseline must finish on every sample. Its speed does not matter. When
the upstream model crashes on an optional accelerated kernel, switch that
kernel to its reference path instead of dropping the sample. Typical errors
include the following:

- `CUDA error: no kernel image is available` — a kernel built for another
  GPU architecture
- `ModuleNotFoundError: No module named 'flash_attn'` — a missing optional
  package
- A Triton compilation error, or an assertion that requires a newer GPU

Search the upstream configuration and command-line flags for the switch,
such as `use_flash`, `use_deepspeed_evo_attention`, or an attention-backend
setting. PyTorch's own flash attention turns off with
`torch.backends.cuda.enable_flash_sdp(False)`. Record every override, the
error it fixed, and its default value in `NOTES.md`. Reference paths compute
the same function more slowly.

### Install the Scorers

Score every structure with standard tools. Do not reimplement the local
distance difference test (lDDT), the template modeling score (TM-score), or
DockQ. Correct implementations handle atom inclusion radii, chain mapping,
symmetric homomers, residue numbering, and atom naming. This guide uses two
scorers:

- **OpenStructure** computes lDDT and TM-score. Install it into its own conda
  environment. The package is on bioconda, which resolves its dependencies
  from conda-forge:

  ```bash
  conda create -y -p "$WORKDIR/envs/ost" --override-channels \
      -c conda-forge -c bioconda openstructure=2.12.0
  OST="$WORKDIR/envs/ost/bin/ost"
  ```

  If you have no conda, install [Miniforge][miniforge] first. Call `ost` by
  its full path. Activating the environment in a BioIR shell can break the
  BioIR interpreter.

- **DockQ** scores interfaces between chains. Install it into its own
  virtual environment, because it requires NumPy older than 2.0:

  ```bash
  python -m venv "$WORKDIR/envs/dockq"
  "$WORKDIR/envs/dockq/bin/pip" install DockQ==2.1.3
  DOCKQ="$WORKDIR/envs/dockq/bin/DockQ"
  ```

A scorer update can change scores. Record both versions in `NOTES.md` and
in the provenance of every score file, and score the upstream and BioIR
predictions with the same installs. DockQ has no version flag, so read its
version from the package metadata:

```bash
"$OST" --version
"$WORKDIR/envs/dockq/bin/python" -c 'import importlib.metadata as m; print("DockQ", m.version("DockQ"))'
```

The install commands pin the tested versions, OpenStructure 2.12.0 and
DockQ 2.1.3, so a rerun resolves the same scorer code. To move to newer
releases, rescore the upstream baseline with them too.

[miniforge]: https://github.com/conda-forge/miniforge

### Score the Upstream Predictions

Score every upstream prediction against its ground truth with the following
commands:

```bash
"$OST" compare-structures \
    -m prediction.cif -r ground_truth.cif -o scores.json \
    --lddt --tm-score --fault-tolerant \
    --min-pep-length 4 --min-nuc-length 4

"$DOCKQ" prediction.cif ground_truth.cif --json dockq.json --short
```

Save the per-sample results, with the scorer versions, as
`ref_data/upstream_metrics.json`. Read and record the scores as follows:

- Read `lddt` and `tm_score` from the OpenStructure JSON.
- Read `GlobalDockQ` from the DockQ JSON. The prediction comes first and the
  ground truth second. Reversed arguments score the wrong direction.
- DockQ scores protein–protein, protein–nucleic-acid, and nucleic-acid
  interfaces. Pass `--small_molecule` for ligand poses, and use the same
  flags on both sides.
- A single chain has no interface: DockQ prints `Need at least two chains`,
  exits 0, and writes no JSON. Check for the JSON file, not the exit code.
  Without one, record `null` and the reason, never `0.0`, which means
  "every interface is wrong."
- Suspicious scores, such as an lDDT near zero, usually mean a chain-mapping
  or file problem. Fix the invocation. Do not fall back to a hand-written
  metric.

Check coverage with `assert_manifest_coverage`, and compare the values with
the accuracy that the upstream authors report. A baseline far below it points
at the inputs or the inference settings, not at the model.

## Check Feature Equivalence

### Generate the References

The references are the features the upstream data pipeline hands to the
upstream model, one `upstream_features/<sample_id>.pt` per sample. Produce
them from the same upstream checkout and settings as the baseline.

Write a script that runs the upstream featurization on each manifest sample
and saves the result with `torch.save`, with NumPy arrays converted to
tensors. The script imports only the upstream package and standard
libraries. Make it refuse to run when any BioIR pipeline module is in
`sys.modules`:

```python
import sys

assert not any(name.startswith("bionemo_ir.pipeline.models") for name in sys.modules), (
    "reference generation must not import the code under test"
)
```

When the upstream pipeline cannot run as a whole in your environment, call
its featurization functions directly. They are still upstream code.

Append one provenance line per artifact to `ref_data/provenance.jsonl`:

```json
{"sample_id": "T1031", "path": "upstream_features/T1031.pt", "sha256": "…", "source": "upstream", "command": "python dump_upstream_features.py --sample T1031", "upstream_commit": "…", "checkpoint": "…", "config_sha256": "…", "upstream_parser": "…", "input_format": "…", "created": "…"}
```

The equivalence test loads a reference only when its provenance line exists
and says `"source": "upstream"`.

### Use the Upstream Production Input Format

Upstream code often accepts several formats for one kind of file, and each
format goes through a different parser. The parsers can extract different
fields, such as per-sequence species identifiers that only one MSA format
carries, and the featurizer branches on those fields. If your reference
script feeds upstream a different format than its inference entry point
uses, the references differ from BioIR by construction. The difference then
looks like a BioIR bug. Match the format as follows:

- Trace the upstream inference entry point from input file to features, and
  note which parser reads each file and which fields the featurizer uses.
- If BioIR reads format A and upstream inference reads format B, convert
  your inputs to format B in the reference script. Cache the converted files
  and record them in the provenance.
- If format B needs a field your inputs lack, fill it with a deterministic
  rule, such as sequential identifiers or the upstream default, and document
  the rule.
- Record the parser and input format in each provenance line.

Suspect a format mismatch in the following cases:

- BioIR produces extra rows whose upstream values are a parser's
  missing-value marker, such as `-1`, `None`, or zeros.
- An upstream flag is always `0`, but BioIR's is mixed.
- Upstream output is empty where BioIR's is not.

### Compute BioIR Features

Run BioIR's parser, tokenizer, and feature stages in process. The following
helper constructs every component the way `build_processor` does, and calls
the production stage code:

```python
import asyncio
import sys

sys.path.insert(0, str(REPO / "examples" / "folding"))
from run_demo import load_requests  # resolves MSA and template paths in the sample JSON

from bionemo_ir.pipeline.stages.feature_generator_stage import FeatureGeneratorUDF
from bionemo_ir.pipeline.stages.parser_stage import ParserUDF
from bionemo_ir.pipeline.stages.tokenizer_stage import TokenizerUDF
from bionemo_ir.registry import get_feature_factory, get_model_class, get_tokenizer, load_metadata

UDF_ARGS = {"compute_by_rows": True, "drop_keys": None, "expected_input_keys": [], "update_row": True}


def bioir_features(model_name: str, request, seed: int = 0, config=None) -> dict:
    """Run BioIR's parser, tokenizer, and feature stages on one request."""
    if config is None:
        config = get_model_class(model_name).get_pretrained_config(model_name)
    metadata = load_metadata(model_name) or None
    tokenizer = get_tokenizer(model_name)
    factory = get_feature_factory(model_name)

    context_generators = {}
    for name, spec in tokenizer.context_generator_specs.items():
        context_generators[name] = spec.generator(config=config, metadata=metadata)
        context_generators[name].required_kwargs = spec.required_kwargs
    transforms = [spec.transform(config=config, **spec.kwargs) for spec in tokenizer.transform_specs]
    generators, collators = [], []
    for specs, built in ((factory.feature_generator_specs, generators), (factory.feature_collator_specs, collators)):
        for spec in specs:
            component = spec.functor(config=config, metadata=metadata, **spec.kwargs)
            component.name = spec.name
            built.append(component)

    parser = ParserUDF(**UDF_ARGS)
    tokenize = TokenizerUDF(
        **UDF_ARGS,
        context_generators=context_generators,
        context_merger_func=tokenizer.context_merger_func,
        transform_funcs=transforms,
        pre_init=factory.pre_init,
    )
    featurize = FeatureGeneratorUDF(
        **UDF_ARGS,
        feature_generators=generators,
        features_merger_func=factory.features_merger_func,
        feature_collators=collators,
        pre_init=factory.pre_init,
    )

    row = {"record": request, "__record_id": request["input_id"], "random_seed": seed}
    row |= asyncio.run(parser.udf_for_item(row))
    row |= asyncio.run(tokenize.udf_for_item(row))
    return asyncio.run(featurize.udf_for_item(row))


request = load_requests(REPO / "examples" / "data" / "samples" / "monomers" / "T1031.json")[0]
features = bioir_features("my-model", request)
```

The helper runs no model, but `get_pretrained_config` queries the GPU, so
run the helper where a GPU is visible. Use the seed that the upstream run
used. When an upstream setting needs a different configuration value, pass a
modified copy of the configuration, and give the same object to Level 2:

```python
config = get_model_class("my-model").get_pretrained_config("my-model")
config = config.model_copy(update={"max_msa_clusters": 128})  # for example
features = bioir_features("my-model", request, config=config)
```

### Compare Every Feature

The equivalence test classifies every feature key in one table in the test
file. A key is deterministic unless a documented source of randomness makes
it vary, and every stochastic key carries a one-line reason. A feature that
is deterministic upstream but varies in BioIR is a bug, not a stochastic
feature. The following test compares every key:

```python
import torch

# Frozen before the first run. Mirror them in NOTES.md.
ATOL = 1e-5
RTOL = 1e-5

STOCHASTIC = {
    "ref_pos": "reference conformer, rotation, and translation drawn per residue",
    "bert_mask": "random MSA masking",
}


def compare(bioir: dict, upstream: dict) -> list[str]:
    errors = []
    if missing := upstream.keys() - bioir.keys():
        errors.append(f"missing keys: {sorted(missing)}")
    if extra := bioir.keys() - upstream.keys():
        errors.append(f"extra keys: {sorted(extra)}")
    for key in sorted(upstream.keys() & bioir.keys()):
        actual, expected = bioir[key], upstream[key]
        if not isinstance(expected, torch.Tensor):  # such as a list of residue names
            if actual != expected:
                errors.append(f"{key}: {actual!r:.60} != {expected!r:.60}")
        elif not isinstance(actual, torch.Tensor):
            errors.append(f"{key}: {type(actual).__name__}, expected a tensor")
        elif actual.shape != expected.shape or actual.dtype != expected.dtype:
            errors.append(f"{key}: {tuple(actual.shape)} {actual.dtype} != {tuple(expected.shape)} {expected.dtype}")
        elif key in STOCHASTIC:
            errors += compare_stochastic(key, bioir, upstream)
        elif not torch.allclose(actual.double(), expected.double(), atol=ATOL, rtol=RTOL, equal_nan=True):
            max_diff = (actual.double() - expected.double()).abs().max().item()
            errors.append(f"{key}: max abs diff {max_diff:.3e}")
    return errors
```

Allow an extra key only when `NOTES.md` explains it. Print one line per key
with its category, its result, and its maximum difference or statistics, so
that a regression is visible at a glance:

```text
token_index   DETERMINISTIC  PASS  max_diff=0.0
restype       DETERMINISTIC  PASS  max_diff=0.0
ref_pos       STOCHASTIC     PASS  bond_length_max_diff=1.2e-02
```

### Test Stochastic Features

An identical seed makes a random feature match exactly only when BioIR draws
random numbers in the same order as upstream. Match the draw order where you
can: it turns a stochastic key into a deterministic one. Where the order
differs, test the properties that the randomness preserves:

1. **Shape, dtype, and padding** match exactly. The `compare` function
   checks shape and dtype, and the padding mask is a deterministic key.
2. **Geometry** matches for coordinates under a random rotation and translation
   per group of atoms, such as a residue's reference conformer. The motion
   preserves every distance within a group. When the pipeline also samples the
   conformer, as Boltz-2 and OpenFold3 do, only bond lengths survive. Compare
   the atom pairs closer than 1.9 Å in the upstream conformer, with a tolerance
   that covers the conformer generator. The RDKit ETKDG conformers that
   OpenFold3 draws differ in bond length by up to 0.12 Å between seeds. To test
   the conformers themselves, match the draw order, or compare each group with
   the conformers that upstream can draw. The following function applies both
   rules:

   ```python
   def geometry_errors(key, actual, expected, groups, sampled_conformer, atol=1e-4, bond_atol=0.2):
       """Compare the distances that a random rotation and translation per group preserve."""
       errors = []
       for uid in torch.unique(groups):
           a, e = actual[groups == uid].double(), expected[groups == uid].double()
           da, de = torch.cdist(a, a), torch.cdist(e, e)
           # A resampled conformer keeps only bond lengths: pairs under 1.9 Å.
           pairs = (de > 0) & (de < 1.9) if sampled_conformer else torch.ones_like(de, dtype=torch.bool)
           diff = (da - de)[pairs].abs()
           if diff.numel() and diff.max() > (bond_atol if sampled_conformer else atol):
               errors.append(f"{key}: distances in group {int(uid)} differ by up to {diff.max():.3f} Å")
       return errors
   ```

3. **Summary statistics** match for sampled and masked values, such as MSA rows
   and masks. For a mask, the mean is the masking rate. Do not apply them to
   coordinates, whose mean and spread change with every rotation. The following
   function compares the mean and the standard deviation:

   ```python
   def stats_errors(key, actual, expected, mean_rtol=0.05, std_rtol=0.10, atol=1e-3):
       a, e = actual.double(), expected.double()
       errors = []
       if abs(a.mean() - e.mean()) > mean_rtol * abs(e.mean()) + atol:
           errors.append(f"{key}: mean {a.mean():+.6f} vs {e.mean():+.6f}")
       if abs(a.std(unbiased=False) - e.std(unbiased=False)) > std_rtol * e.std(unbiased=False) + atol:
           errors.append(f"{key}: std {a.std(unbiased=False):.6f} vs {e.std(unbiased=False):.6f}")
       return errors
   ```

4. **Distributions** match across seeds, when runs are cheap. Run both
   sides with at least 20 seeds, and compare the per-run means with a
   two-sample Kolmogorov–Smirnov test (`scipy.stats.ks_2samp`, p > 0.01).
   Skip this check when each upstream run needs GPU inference. Checks 1 to 3
   catch most regressions.

`compare_stochastic`, which `compare` calls, applies check 2 to coordinate
keys and check 3 to the other stochastic keys. `COORDINATES`
names the group and padding-mask keys of each coordinate key, and whether
the pipeline samples a conformer per group:

```python
COORDINATES = {"ref_pos": {"groups": "ref_space_uid", "mask": "atom_pad_mask", "sampled_conformer": True}}


def compare_stochastic(key: str, bioir: dict, upstream: dict) -> list[str]:
    actual, expected = bioir[key], upstream[key]
    if key not in COORDINATES:
        return stats_errors(key, actual, expected)
    spec = COORDINATES[key]
    valid = upstream[spec["mask"]].bool()
    groups = upstream[spec["groups"]][valid]
    return geometry_errors(key, actual[valid], expected[valid], groups, spec["sampled_conformer"])
```

### Require Non-Empty References

Some features are legitimately empty for most inputs: chirality and
stereochemistry constraints, distance bounds, and template features. An
empty BioIR tensor then matches an empty upstream tensor even when both are
wrong. For each such feature, the manifest must include a sample where the
feature cannot be empty, and the test must assert that the reference is
non-empty there. Such samples include the following:

- A SMILES ligand with an explicit stereocenter, such as `[C@H]`
- An aromatic ring
- An `E` or `Z` double bond
- A template that aligns to the query

For templates, compare the `template_*` tensors on two inputs. A
self-template, the query's own structure used as its template, exercises the
full alignment path. A multi-chain complex exercises per-chain mapping. The
no-template output must equal the upstream no-template output exactly.

### Debug From the Bottom Up

Run the test on every manifest sample. When a sample fails, follow these
steps:

1. Save the failing output and a minimal diff under `debug/`.
2. Find the first stage that diverges. Compare the tokenizer output, then
   each feature generator's output, then each collator's output.
3. Fix the pipeline and rerun the failing sample.
4. Rerun the whole manifest to confirm the fix broke nothing else.

Continue to Level 2 only when every sample passes.

## Check End-to-End Accuracy

### Run the BioIR Pipeline

Level 2 runs `build_processor`, the public entry point, with the same
samples, seeds, and inference settings as the upstream baseline. Compare the
BioIR settings with `upstream_config.json` before you start. The following
example runs the manifest samples serially:

```python
import json

from bionemo_ir.pipeline.processor.engine_proc import EngineProcessorConfig, build_processor
from bionemo_ir.pipeline.stages.configs import FeatureGeneratorStageConfig, WriterStageConfig

with open(WORKDIR / "ref_data" / "sample_manifest.json") as f:
    manifest = json.load(f)
requests = [
    request
    for sample in manifest["samples"]
    for request in load_requests(REPO / sample["input"])
    if request["input_id"] == sample["sample_id"]
]
rows = [{"record": r, "__record_id": r["input_id"], "random_seed": 0} for r in requests]


def processor_config(executor_backend, output_path):
    return EngineProcessorConfig(
        model_source="my-model",
        executor_backend=executor_backend,
        # The values recorded in NOTES.md, matching the upstream run.
        runtime_args={"recycling_steps": 3, "num_sampling_steps": 200, "diffusion_samples": 1},
        feature_generator_stage=FeatureGeneratorStageConfig(init_context={"random_seed": 0}),
        writer_stage=WriterStageConfig(output_path=output_path, format="cif"),
    )


results = build_processor(processor_config(None, str(WORKDIR / "bioir" / "serial")))(rows)
assert_manifest_coverage(row["__record_id"] for row in results)
```

If Level 1 used a modified configuration, pass the same object to
`EngineProcessorConfig` as `engine_kwargs={"config": config}`. Every stage,
including the model, then uses it.

Run the same rows through Ray as well:

```python
import ray

ray_processor = build_processor(processor_config("ray", str(WORKDIR / "bioir" / "ray")))
ray_results = list(ray_processor(ray.data.from_items(rows)).materialize().iter_rows())
```

Score every BioIR structure with the same OpenStructure and DockQ commands
and flags as the baseline.

### Compare Against the Baseline

Compare the BioIR and upstream scores for each sample in lDDT, TM-score,
and DockQ:

- **Pass** when BioIR is lower than upstream by less than 0.05. The two
  implementations draw different diffusion samples, even from the same seed.
- **Fail** when BioIR is lower by 0.05 or more.
- **Investigate before you pass** when BioIR scores higher. A higher score
  can be real, but it can also come from different sample selection, chain
  mapping, scorer inputs, or settings. Pass it only after you confirm
  feature equivalence, identical settings, the same sample set, the shared
  writer, and the same scoring command.

Also require the following:

- No NaN or infinite values in any output, and plausible geometry
- An output for every sample, and no row whose `__inference_error__` holds
  an error message
- Identical per-sample scores from the serial and Ray runs

Missing samples, missing provenance, and unvalidated features fail the
validation even when every score looks good.

### Isolate a Failing Sample

When a sample fails Level 2, follow these steps:

1. Rerun it on both sides with three to five other seeds, and compare the
   score spreads. A rerun with the same seed reproduces the same structure.
   If the upstream and BioIR ranges overlap, sampling variance explains the
   gap.
2. If the ranges do not overlap, rerun Level 1 on that sample. A small
   feature difference can compound through the model.
3. If the features match, the problem lies after the features: in the model
   call, the postprocessor, or the writer. Feed the upstream features into the
   BioIR model once, as in [Calling `forward`][forward], and save the batch and
   the raw output:

   ```python
   torch.save({"batch": batch, "output": output}, WORKDIR / "debug" / f"{sample_id}_model_io.pt")
   ```

   You can then rerun the postprocessor and writer on the saved output, and
   score the result, without running inference again:

   ```python
   from bionemo_ir.data.utils import get_all_atom_types, get_all_residue_types
   from bionemo_ir.data.writers.cif_writer import CIFWriter
   from bionemo_ir.registry import get_postprocessor

   saved = torch.load(WORKDIR / "debug" / f"{sample_id}_model_io.pt")
   folding_output = get_postprocessor("my-model")()(saved["batch"], saved["output"])
   writer = CIFWriter(
       res_type_mapping=dict(enumerate(get_all_residue_types("my-model"))),
       atom_type_mapping=dict(enumerate(get_all_atom_types("my-model"))),
   )
   writer.set_output_path(str(WORKDIR / "debug" / f"{sample_id}.cif"))
   writer.write(folding_output)
   ```

   The writer stage uses the same mappings.

## Report the Results

Attach the evidence to your pull request:

- The manifest, with its per-category sample counts
- The settings table from `NOTES.md`, with every upstream–BioIR difference
- Level 1 results: per sample, every key with its category, result, and
  maximum difference or statistics
- Level 2 results: per sample, upstream and BioIR scores for each metric,
  the difference, and the serial and Ray results
- Limitations: unsupported inputs with example files, and expected
  divergences such as fixed upstream bugs
- For every result, the command, the BioIR commit, the upstream version,
  the scorer versions, and the artifact paths

## Related

- [Port a Data Pipeline][port] — build the pipeline this guide validates.
- [Python API][api] — `build_processor`, `EngineProcessorConfig`, and
  output rows.
- [Benchmarks][benchmarks] — how BioIR compares its own models with their
  upstream baselines.

[api]: ../ref/api.md
[benchmarks]: ../ref/benchmark.md
[forward]: ../ref/api.md#calling-forward
