---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
name: make-data-pipeline
description: Port an open-source bioinformatics data pipeline into the BioIR pipeline architecture. Use when the user asks to create, port, convert, or write a new data pipeline from OSS code, or add a new model's data processing to the BioIR system.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Port OSS Data Pipeline to BioIR

**Input:** OSS model name + source code location. **Output:** Complete BioIR
pipeline module + equivalence tests + summary report.

## Mandatory Anti-Forgery Gates

These gates are non-negotiable. If any gate cannot be satisfied, stop and report
the blocker instead of claiming completion.

1. **No unverifiable PASS claims.** Every PASS in the final report must cite an
   executed command, exit code, artifact path, and the exact sample set. Do not
   summarize tests as passing from memory, assumptions, or partial output.
1. **Reference provenance is mandatory.** Every OSS reference artifact must have
   a provenance record in `$WORKDIR/ref_data/provenance.jsonl` containing:
   `sample_id`, artifact path, SHA256, generation command, OSS source path, OSS
   git commit or archive hash, checkpoint identifier/hash, resolved config hash,
   timestamp, and whether the artifact came from OSS inference, OSS
   featurization primitives, or another approved reference path.
1. **No self-reference.** Reference generation scripts must fail if they import
   `bionemo_ir.pipeline.models.<model_name>` or write references from
   BioIR outputs. Equivalence tests must fail if the reference artifact
   provenance does not say `source="oss"`.
1. **No hidden skips.** Do not use `pytest.skip`, `xfail`, broad `try/except`,
   missing-key allowlists, `continue` on failed samples, or
   environment-dependent shortcuts to make validation green. Any unsupported
   sample, missing tensor, failed metric, or crashed backend is a blocker unless
   the user explicitly narrows scope.
1. **No tolerance inflation.** Tolerances and metric thresholds must be written
   before debugging and recorded in `$WORKDIR/NOTES.md`. Do not relax
   tolerances, recategorize deterministic tensors as stochastic, drop keys, or
   change metric thresholds after seeing failures without user approval and a
   written rationale.
1. **No synthetic correctness.** Do not create fake ground truths, mock model
   outputs, hand-authored reference tensors, or simplified metric
   implementations to satisfy tests. Synthetic fixtures may be used only for
   unit tests and must never replace OSS equivalence or e2e scoring.
1. **Exact sample manifest.** Before Phase 7, write
   `$WORKDIR/ref_data/sample_manifest.json` from `examples/data/samples/` and
   reuse it for OSS baseline, feature equivalence, BioIR e2e, serial
   `build_processor`, and Ray `build_processor`. Every phase must assert exact
   set equality against this manifest.
1. **Failure evidence is required.** When a test fails, preserve the failing
   command output and a minimal diff artifact under `$WORKDIR/debug/failures/`.
   Do not overwrite failing artifacts until the fix is verified by rerunning the
   full manifest.
1. **Unsupported scope must be explicit.** The basic schema now supports
   protein, RNA, DNA, CCD ligands, and SMILES ligands. Do not skip RNA/DNA or
   ligand/small-molecule inputs solely because they are non-protein. If a
   model-specific pipeline cannot support a schema-supported input type, record
   it as a model/pipeline limitation with exact input examples and prove that
   all in-scope samples still pass.
1. **Implementation notes are mandatory.** Maintain
   `$WORKDIR/implementation-notes.md` as a running decision log. Update it as
   work happens, not only at the end. Any design decision, OSS deviation,
   tradeoff, unresolved ambiguity, or user-confirmation question must be
   recorded there with a timestamp.

## Phase 0 — Gather All Resources & Set Up WORKDIR

Network/disk access may require user approval and the user may leave. Do ALL
resource access now and save locally before proceeding.

### Step 1 — Set up WORKDIR

All development artifacts (tests, debug scripts, e2e smoke tests, reference
data) live in a **WORKDIR** — a user-specified working directory outside the
main package tree. Ask the user for the WORKDIR path, or default to
`workdir/<model_name>/` at the repo root.

```bash
WORKDIR=${WORKDIR:-workdir/<model_name>}
mkdir -p $WORKDIR/{tests,debug,e2e,ref_data}
```

**WORKDIR layout:**

```text
$WORKDIR/
├── tests/                       # Equivalence + unit tests
│   ├── test_equivalence.py
│   ├── test_tokenizer.py
│   ├── test_generators.py
│   └── conftest.py
├── debug/                       # Debug scripts, intermediate dumps
│   ├── dump_oss_features.py     # Save OSS outputs as reference .pt files
│   └── compare_single_stage.py  # Compare one stage at a time
├── e2e/                         # End-to-end smoke tests
│   ├── smoke_test.py            # Full pipeline: input → features → (model) → output
│   └── sample_inputs/           # Sample input files (JSON, CIF, A3M)
├── ref_data/                    # Reference data from OSS pipeline
│   ├── reqs.json                # Input requests
│   ├── sample_manifest.json     # Canonical sample set for all phases
│   ├── provenance.jsonl         # SHA256 + command provenance for every reference artifact
│   └── samples/                 # OSS output feature dicts as <input_id>.pt
├── implementation-notes.md      # Running decisions, deviations, tradeoffs, open questions
└── NOTES.md                     # Status, decisions, open questions
```

**Pipeline implementation** files go directly into the codebase
(`bionemo_ir/pipeline/models/<model_name>/`) because they must be
importable via the package's module system. The WORKDIR holds everything else:
tests, debug tools, e2e smoke tests, and reference data.

**"Merging to the codebase"** (Phase 10) means: registering in `registry.py`,
cleaning up debug artifacts, and committing. Tests stay in `$WORKDIR/`.

### Step 1.5 — Maintain implementation notes

As you work, maintain a running `implementation-notes.md` (or an HTML file if
explicitly requested) that captures anything I should know about how the
implementation diverges from or interprets the OSS data pipeline, including:

- Timestamp
- Design decisions: choices you made where some point was ambiguous
- Deviations: places where you intentionally departed from the OSS data
  pipeline, and why?
- Tradeoffs: alternatives you considered and why you picked what you did
- Open questions: anything you'd want me to confirm or revise

Default to `$WORKDIR/implementation-notes.md` unless the user asks for HTML.
Create it during Phase 0 and keep it current throughout all phases. Do not defer
entries until the final report; write an entry immediately when a decision or
ambiguity appears.

Use this template for each entry:

```markdown
## YYYY-MM-DD HH:MM TZ — <short title>

**Category:** Design decision | Deviation | Tradeoff | Open question

**Context:** What OSS behavior, file, function, or ambiguity triggered this note.

**Decision / interpretation:** What was chosen or how the OSS behavior was interpreted.

**Reason:** Why this is correct for BioIR, including production constraints or schema limitations.

**Alternatives considered:** Other approaches and why they were rejected.

**Impact / validation:** Expected effect on feature equivalence, e2e metrics, supported inputs, or future maintenance.

**Needs user confirmation:** Yes/No. If yes, state the exact question.
```

At the end of each major phase, scan the current changes and append any missing
notes before proceeding. The Phase 10 summary must include the path to this file
and summarize unresolved open questions.

### Step 2 — Locate OSS source code

**Ask the user** where the OSS data pipeline code lives. Do NOT guess or search
without asking first — the user knows which version/branch/fork is
authoritative.

If the user doesn't specify, suggest these common locations:

```bash
# 1. Vendored submodules in this repo
ls 3rdparty/<model>/

# 2. Reference copies under examples
ls examples/<model>/original/

# 3. External — clone if needed (code-only, skip weights)
git clone --depth 1 <oss_repo_url> /tmp/<model>_oss
```

Once confirmed, record the path as `$OSS_ROOT` — all subsequent OSS references
are relative to it.

### Step 3 — Identify data pipeline entry points

**Start with the OSS inference script.** Every OSS model has a top-level script
that runs end-to-end inference (e.g., `run_openfold.py`, `predict.py`,
`infer.py`). Read this script first — it reveals:

- How input is parsed (JSON, CIF, FASTA)
- How the data pipeline is invoked (dataset class, feature pipeline, collation)
- How the model is loaded and called
- How output is post-processed and saved
- What config/CLI args control the pipeline

Trace the inference script's call chain to locate the data pipeline files:

| Purpose                    | Typical OSS Files                                          |
| -------------------------- | ---------------------------------------------------------- |
| **Inference entry point**  | `run_*.py`, `predict.py`, `infer.py` — **read this first** |
| Raw input → numpy features | `data_pipeline.py`, `pipeline.py`, `dataset.py`            |
| Numpy → tensor transforms  | `feature_pipeline.py`, `input_pipeline.py`                 |
| Individual transforms      | `data_transforms.py`, `featurizer.py`                      |
| Constants/chemistry        | `residue_constants.py`, `chemical.py`, `const.py`          |
| Post-processing            | `output.py`, `postprocessor.py`, `confidence.py`           |
| Data schemas               | `types.py`, `data.py`, `structure.py`                      |

Record each file path. These are your read-only references for all subsequent
phases.

### Step 4 — Verify BioIR base classes are available

```bash
python -c "from bionemo_ir.pipeline.base import ContextGeneratorBase, TransformBase, FeatureGeneratorBase, FeatureCollatorBase; print('OK')"
```

If this fails, install first: `pip install -v -e '.[dev]'`. Do not add
`--no-build-isolation` unless `cmake`, `nanobind` and `setuptools` are
already installed — the flag skips exactly those, and the nanobind
extension then fails to configure.

______________________________________________________________________

## Phase 1 — Survey Existing Coverage & Analyze OSS Pipeline

### Step 1 — Check for existing BioIR pipeline

Before writing anything, check if a pipeline already exists for this model:

1. Search `bionemo_ir/pipeline/models/` for a directory matching the model
   name.
1. Check `bionemo_ir/registry.py` for existing factory registrations
   (`ModelComponentsFactory` subclasses and `register_all_factories()`).

**If existing code is found:**

- Read it carefully. It may already handle this model — in which case only
  updates are needed.
- If the existing code covers a closely related model but needs adaptation,
  decide whether to **extend** the existing pipeline or create a new one. Prefer
  extending if the changes are minor; create new if the architecture diverges
  significantly. Report the decision and rationale to the user before
  proceeding.

**If no existing code is found:** proceed to Phase 2.

### Step 1.5 — Build the canonical sample manifest

Before any baseline, reference generation, or e2e test, enumerate the exact
samples that are in scope and save them to
`$WORKDIR/ref_data/sample_manifest.json`. This manifest is the source of truth
for all later phases.

The manifest must include each `sample_id`, input path, ground-truth path, chain
IDs, sequence lengths, MSA paths, and category (`protein_monomer`,
`protein_homopolymer`, `protein_heterooligomer`, `rna`, `dna`, `ccd_ligand`,
`smiles_ligand`, `mixed_complex`, or `unsupported`). Protein, RNA, DNA, CCD
ligand, SMILES ligand, and mixed-complex samples are schema-supported. A sample
may be marked `unsupported` only with an explicit model/pipeline limitation, not
because the basic schema cannot represent it.

Every validation script must load this manifest and assert exact set equality
for the samples it processed. A missing, renamed, duplicated, or extra sample is
a hard failure.

### Step 2 — Reference ALL existing data pipelines to discover the best fit

**Read every existing pipeline** under `bionemo_ir/pipeline/models/`
before writing anything. For each one, catalog:

1. **Pattern** — A (flat tensor dict) or B (mixed context row)
1. **Context generator** — what it takes as input, what it returns, key methods
1. **Counts** — how many transforms, generators, collators, and what they do
1. **SampleRepeater** — yes/no, and what it wraps
1. **Unique files** — anything beyond the standard set (e.g., `msa_pairing.py`,
   `structure.py`, `tokenizer_logic.py`)

Current pipelines to read (as of this writing):

| Pipeline     | Pattern               | Key Trait                                                                |
| ------------ | --------------------- | ------------------------------------------------------------------------ |
| `openfold2/` | A (flat tensor dict)  | Residue-level, SampleRepeater for recycling, monomer + multimer variants |
| `boltz1/`    | B (mixed context row) | Wraps Boltz2, protein-ligand, no SampleRepeater                          |
| `boltz2/`    | B (mixed context row) | Token-level, Structure/Token/TokenBond dataclasses, RDKit molecules      |

**Then decide which existing pipeline is the best structural reference** for the
new model. Consider:

- Same model family (OpenFold2 for OpenFold3, Boltz1 for Boltz2)
- Same data flow pattern (flat tensors vs mixed context)
- Same input complexity (protein-only vs multi-entity)
- Same feature set (MSA handling, template handling, atom-level vs
  residue-level)

Report the chosen reference and rationale to the user. Also check the registry
for related models from the same family.

### Step 3 — Deep analysis of OSS data pipeline

Study the OSS code and produce a **function inventory**. For each
function/class, record:

1. **Name** and file location
1. **What it does** (one sentence)
1. **Classification** — see table below
1. **Conditional?** — runs only when a config flag is True → needs
   `is_enabled()`
1. **Curried/parameterized?** — takes factory args → maps to `__init__` params
1. **Ensembled?** — runs per recycling iteration → `FeatureCollatorBase`

| Classification                      | Criteria                                                          | BioIR Target                                      |
| ----------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------- |
| **Raw feature builder**             | Creates numpy arrays from sequences/MSAs/templates                | `ContextGeneratorBase` in `feature_context.py`    |
| **Non-ensembled, modifies dict**    | Runs once, modifies existing keys (cast, reorder, squeeze)        | `TransformBase` in `transforms.py`                |
| **Non-ensembled, creates new keys** | Runs once, adds new feature keys (masks, profiles, atom14)        | `FeatureGeneratorBase` in `feature_generators.py` |
| **Ensembled, stochastic**           | Runs per recycling iter, involves randomness (sample, mask, crop) | `FeatureCollatorBase` in `feature_collators.py`   |
| **Ensembled, deterministic**        | Runs per recycling iter, deterministic (cluster, pad, select)     | `FeatureCollatorBase` in `feature_collators.py`   |
| **Output processing**               | Converts model output → structured output                         | `PostProcessorBase` in `postprocessor.py`         |

Save this inventory to `$WORKDIR/NOTES.md` — it is the blueprint for all
subsequent phases. See [mapping-guide.md](mapping-guide.md) for the full
OSS→BioIR mapping guide with concrete examples.

### Step 4 — Choose pipeline pattern

There are two patterns depending on the model's data flow complexity:

**Pattern A — OpenFold-style** (single context generator → full tensor dict):

- `ContextGeneratorBase` produces the complete initial tensor dict from
  `InputParsed`
- Transforms, generators, collators all operate on flat tensor dicts
- Suitable when: OSS has a single `make_features()` or `data_pipeline.process()`
  entry point

**Pattern B — Boltz2-style** (context generator returns a "row" dict with mixed
tensor/non-tensor data):

- `ContextGeneratorBase` returns a **context dict** ("row") containing
  structures, molecules, parsed MSAs — not just tensors
- Feature generators read `context["_row"]` and produce tensors incrementally,
  one logical step at a time
- Feature collators produce the final tensor set
- Suitable when: OSS has multi-step featurization (structure building, molecule
  loading, MSA per chain, etc.) with intermediate non-tensor state

**How to decide:** If the OSS pipeline passes non-tensor intermediate data
(structures, molecule objects, parsed MSA dicts) between featurization steps,
use Pattern B. If everything is tensors after the initial parsing, use Pattern
A.

Record the chosen pattern in `$WORKDIR/NOTES.md`.

______________________________________________________________________

## Phase 2 — Create Foundation Files

Create the model pipeline directory and foundation files **in the codebase**
(must be importable).

**Reference:** [file-templates.md](file-templates.md) for complete code
templates, [architecture-reference.md](architecture-reference.md) for base class
signatures.

### Target directory (in codebase)

```text
bionemo_ir/pipeline/models/<model_name>/
├── __init__.py
├── const.py
├── common.py
├── feature_context.py
├── transforms.py
├── tokenizer.py
├── feature_generators.py
├── feature_collators.py
├── feature_factory.py
└── postprocessor.py
```

Some models need additional files (e.g., `structure.py` for structure
manipulation, `tokenizer_logic.py` for complex tokenization, `msa_pairing.py`
for multimer MSA pairing). Add them as needed.

### Step 1 — `__init__.py`

Empty file. NVIDIA copyright header only.

### Step 2 — `const.py`

Copy domain-specific constants from the OSS code:

- Residue/atom type mappings and lookup tables
- Standard masks and index arrays
- Physical/chemical constants (bond lengths, angles, etc.)

These are **data, not code** — copying constant values is explicitly allowed.
Reformat to match BioIR style (module-level dicts/lists, no class wrappers
unless needed).

### Step 3 — `common.py`

Reimplement shared math/helper functions used across multiple pipeline files.
Common candidates:

- One-hot encoding
- Torsion angle computation
- Pseudo-beta calculation
- Distance/geometry utilities
- Type conversion helpers

Each function must be a **fresh reimplementation** from understanding the OSS
algorithm — not a copy-paste of OSS code.

______________________________________________________________________

## Phase 3 — Implement Context Generation & Tokenizer

This phase builds the pipeline entry point: raw parsed input → initial tensor
dict.

### Step 1 — `feature_context.py`

Implement one or more `ContextGeneratorBase` subclasses.

**Pattern A (OpenFold-style):**

```python
class FeatureContextGenerator(ContextGeneratorBase):
    def __call__(self, parsed: InputParsed) -> dict[str, torch.Tensor]:
        # 1. Build raw numpy features from parsed input
        # 2. Convert to tensors, filtering to relevant feature set
        # 3. Return flat tensor dict
```

**Pattern B (Boltz2-style):**

```python
class ModelContextGenerator(ContextGeneratorBase):
    def __call__(self, parsed: InputParsed) -> dict[str, Any]:
        # 1. Build structure, tokenize, load molecules, parse MSA
        # 2. Return context row with mixed tensor/non-tensor data
        # Feature generators will read this via context["_row"]
```

### Step 2 — `transforms.py`

Implement `TransformBase` subclasses for non-ensembled transforms from your
function inventory:

- Each modifies and returns the tensor dict (in-place semantics)
- Use `is_enabled()` for conditional transforms
- Map curried OSS args to `__init__` constructor params

### Step 3 — `tokenizer.py`

Wire context generators + transforms into a declarative `TokenizerBase`:

```python
class Tokenizer(TokenizerBase):
    context_generator_specs = OrderedDict({
        'primary': ContextGeneratorSpec(
            name='primary',
            generator=FeatureContextGenerator,
            required_kwargs=['parsed']
        )
    })
    context_merger_func = dict_context_merger
    transform_specs = [
        TransformSpec(name='cast_to_64_bit_ints', transform=CastTo64BitInts),
        # ... all non-ensembled transforms in OSS execution order
    ]
```

If the model has monomer + multimer variants, create separate tokenizer classes
(or parameterize via config).

### Step 4 — Write debug script for tokenizer

Create `$WORKDIR/debug/test_tokenizer_stage.py` to verify the tokenizer output
against OSS before proceeding:

```python
# Quick sanity check: run tokenizer on one sample, print keys + shapes
from bionemo_ir.pipeline.models.<model>.tokenizer import Tokenizer
# ... instantiate, run, compare with OSS output
```

______________________________________________________________________

## Phase 4 — Implement Feature Pipeline

### Step 1 — `feature_generators.py`

Implement `FeatureGeneratorBase` subclasses for each non-ensembled generator
from your function inventory:

- Each returns a **NEW** `feats = {}` dict with ONLY newly-created keys
- Use `is_enabled()` for conditional generators (e.g., template features gated
  on `config.enable_template`)
- Map OSS `common_cfg.X` references to `self.config.X`

**Pattern B note:** generators may read `context["_row"]` to access non-tensor
data from the context generator. The feature stage passes `context` (which
includes `_row`) as the second argument.

### Pipeline metadata (atom tables, chemical dictionaries, auxiliary data)

Many models require **metadata** beyond the raw input sequences — chemical
dictionaries (CCD), atom geometry tables, molecule libraries, or pre-computed
lookup data. This metadata provides per-residue or per-atom reference
information (ideal coordinates, element types, charges, atom names, bond
connectivity) that the model's input embedder uses to construct initial
representations.

**Discovery process — follow the OSS:**

1. **Read the OSS inference script** to find what metadata files it loads at
   startup (CCD pickles, molecule directories, atom constant tables, etc.).
1. **Trace the loading code** — find where the OSS loads metadata: pickle files,
   JSON configs, CSV tables, npz archives, or computed constants. Record the
   format, content, and how it flows into feature generation.
1. **Check existing BioIR pipelines** — other models may already load the same
   or similar metadata. For example, Boltz2 loads CCD as a pickle of
   `dict[str, RDKit.Mol]` and molecule pkls from a directory. See what loaders
   and paths already exist.
1. **Map the OSS metadata to BioIR config** — add metadata paths (e.g.,
   `ccd_path`, `mol_dir`, `atom_table_path`) to the model's config. Load lazily
   in the context generator or feature generator `__init__`.

**Common metadata patterns across models:**

| Metadata                  | What it provides                                                         | Typical format                                    | Used by       |
| ------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------- | ------------- |
| CCD / chemical dictionary | Ideal atom coordinates, element types, charges per residue               | Pickle of `dict[str, RDKit.Mol]`, or CIF, or JSON | Boltz1/2, OF3 |
| Molecule libraries        | Per-component RDKit Mol objects                                          | Directory of `.pkl` files                         | Boltz2        |
| Atom constant tables      | Atom names, backbone definitions, atom counts per residue type           | Python dicts/lists in `const.py`                  | OF2, OF3      |
| CCD component types       | Mapping from CCD component type → molecule type (protein/RNA/DNA/ligand) | Python dict                                       | OF3           |

**For all-atom models** (OF3, Boltz2, AF3-style), the metadata drives atom-level
feature generation:

- `ref_pos` (ideal geometry coordinates) — from CCD conformers or atom tables
- `ref_element` (atomic number) — from CCD or atom definitions
- `ref_charge` (formal charge) — from CCD
- `ref_atom_name_chars` (atom name encoding) — from atom name strings
- `ref_space_uid` (residue grouping) — computed from chain/residue indices
- `num_atoms_per_token` — from metadata atom counts per residue type
- `start_atom_index` — cumulative sum of atom counts

**Key principle:** discover how the OSS loads and uses its metadata, then
reimplement the same loading in BioIR using standard libraries (pickle, json,
rdkit, numpy). The metadata itself (data files) can be shared between OSS and
BioIR — only the loading code is reimplemented.

### Cheminformatics-derived feature pitfalls

Pipelines that process small molecules or non-standard residues frequently
compute features from a chem toolkit (typically RDKit): distance bounds,
chirality, stereochemistry, planarity, atom counts, charges, conformer
coordinates. The exact feature *names* are model-specific and you'll learn them
from the OSS function inventory in Phase 1. The
**pitfalls below are model-independent** — they apply any time you compute
features off an RDKit Mol — and they all share one failure mode: the feature
tensor is produced *empty* (or with wrong values) when it should be populated,
and silently matches an equally broken OSS reference until inference quality
regresses.

**Treat "empty constraint/feature tensor" as a hypothesis, not a result.** For
every cheminformatics-derived feature, the equivalence test must verify both
shape parity with OSS *and* non-emptiness on at least one input where the
feature must exist (a SMILES with an explicit `[C@H]`, an aromatic ring, an
`E`/`Z` double bond, a known bond-length constraint, etc.). Without the
non-emptiness check, you only catch divergences where the OSS reference is
non-zero — and the most common bugs leave both sides at zero.

**Common failure modes:**

1. **Cleaning the molecule loses computed properties.** Operations like
   `RemoveHs(sanitize=False)`, `RemoveAtoms`, copying a mol via SMILES
   round-trip, or serialising/deserialising can strip RDKit-computed atom/bond
   properties (`_CIPRank`, `_CIPCode`, hybridization, ring membership). Any
   feature that *reads* these properties will silently return empty results when
   they're missing. After every mol-mutating step, re-run the perception you
   depend on (`AssignStereochemistry`, `SanitizeMol`, `AssignCIPLabels`, etc.)
   and assert the property is present
   (`assert all(a.HasProp("_CIPRank") for a in mol.GetAtoms())`).
1. **Perception state is not guaranteed on toolkit-loaded mols.** Pickle-loaded,
   CIF-loaded, or vendor-supplied mols often arrive without aromaticity
   perceived, without ring info initialised, without stereo assigned. SMARTS
   patterns that target aromatic atoms / specific ring sizes will not match.
   Decide intentionally between two stances and document it: (a) match OSS
   exactly by not adding perception calls the OSS doesn't make (both sides
   produce the same empty result for these mols), or (b) call the perception
   explicitly and reconcile with OSS. *Do not* call perception in only one of
   the two pipelines — that creates a silent divergence that looks like a
   BioIR bug.
1. **Prep ordering matters for derived properties.** Several RDKit APIs depend
   on prior prep: ring perception (`UpdatePropertyCache` + `GetSymmSSSR`) before
   `GetMoleculeBoundsMatrix`; `AssignStereochemistry` before chiral /
   stereo-bond lookups; `EmbedMolecule` before reading a conformer. If your
   pipeline runs these out of order, the dependent feature comes back
   wrong-but-non-erroring. Verify the prep sequence the OSS uses and replicate
   it.
1. **Failure-prone steps must raise, not pass through silently.** Conformer
   embedding (`EmbedMolecule`) returns a non-zero exit code on failure rather
   than throwing. Sanitisation can partial-fail. SMARTS compilation returns
   `None` for invalid patterns. Treat each of these as a hard failure with a
   clear error — a silent failure leaves a mol with no conformer / no
   aromaticity / no chirality and the downstream features look "fine" while
   being wrong.
1. **Atom enumeration order is part of the contract.** Index tensors are
   *order-sensitive*. If the OSS iterates `mol.GetAtoms()` and you iterate via a
   different traversal, the indices line up only by accident. Always enumerate
   atoms the same way the OSS does, and build the `local_idx → global_idx` map
   in the same order.

**Generic conversion recipe for any cheminformatics-derived feature group:**

1. Identify the OSS function and what RDKit prep it assumes was already done.
1. Reimplement the computation against an `idx_map: dict[local_idx, global_idx]`
   so the caller can shift to global atom indices.
1. Return a list of dicts (one per constraint/feature item) — this stays uniform
   across feature kinds and is easy to thread through the context row.
1. In the feature generator, materialise the indices with a small helper that
   takes a list of dicts and an expected arity and returns `(arity, N)`:

   ```python
   def _stack_idx(items, arity):
       if not items:
           return torch.empty((arity, 0), dtype=torch.long)
       rows = np.asarray([list(c["atom_idxs"]) for c in items], dtype=np.int64).T
       return torch.from_numpy(rows).long()
   ```

1. Pair index tensors with parallel bool / float arrays (e.g. is_reference
   flags, upper/lower bounds, type masks) built from the same items list — keeps
   the ordering tied together by construction.

**Cache compiled SMARTS** at module load if any of the features rely on them —
`MolFromSmarts(...)` is not free and the feature runs once per ligand residue,
which adds up.

**Equivalence-test discipline for cheminformatics features:** include at least
one input where each computed feature kind *must* be non-empty (the manifest's
curated SMILES, a known multi-component CCD ligand with explicit stereo, etc.).
The Phase 8 categorisation table is the place to record these as "DETERMINISTIC,
expect non-empty for sample X" so a future regression that leaves them empty
fails the test loudly.

### Template featurization (protein-only)

If the target model consumes structural templates (OF2/OF3/Boltz-style),
templates are a **first-class, in-scope input** (schema:
`Template`/`TemplateParsed`, protein-only). Do not stub them out as "not
supported" — port the real featurization and gate the whole path on template
presence. Worked reference: the OpenFold3 direct-CIF port in
`workdir/openfold3-port/template_equiv/implementation-notes.md` (L1 21/21 vs OSS
on the NIM `data_with_template` set). Emit templates from a
`FeatureGeneratorBase` (e.g. `TemplateFeatureGenerator`) reading the parsed
templates off `context["_row"]`.

**The direct-CIF path (what inference actually uses).** OSS template pipelines
usually have a search/cache path *and* a direct path where the user supplies the
template mmCIF. Inference uses the direct path. The pipeline is: for each
protein chain, (1) parse each attached template mmCIF, (2) select a chain, (3)
align that chain's sequence to the query, (4) map query token positions →
template residues, (5) extract per-residue backbone atoms, (6) compute the
model's `template_*` tensors (typically restype one-hot, pseudo-beta mask,
backbone-frame mask, distogram, unit-vector).

**Pitfalls — every one of these was a real L1 divergence in the OF3 port:**

1. **The no-template path must be byte-identical to the OSS no-template stub —
   not all-ones.** A disabled/absent-template path does **not** emit `mask=1`
   everywhere; OSS emits **all-zero** masks + restype one-hot at the
   **GAP class**. Getting this wrong silently degrades every no-template
   prediction while looking populated. Verify the no-template branch is
   byte-identical to OSS before touching the real path.
2. **Use the exact same aligner as OSS.** OSS `run_kalign` uses `kalign`;
   substituting biopython `PairwiseAligner` produces different alignments and
   fails L1. Match the aligner (add it as a pinned dependency if OSS also uses
   it) rather than approximating.
3. **Match the OSS altloc policy when reading the CIF.** biotite defaults to
   `altloc="first"` (conformer 'A'); OSS selects highest-occupancy
   (`altloc="occupancy"`). Restype + masks match either way, but
   distogram/unit-vector *values* diverge on multi-altloc templates. Parse with
   the same policy OSS uses.
4. **Source the alignment sequence the way OSS does.** OSS aligns against the
   per-entity `entity_poly.pdbx_seq_one_letter_code_can` (parent one-letter code
   for modified residues, e.g. MSE→'M'). A `pdbx_poly_seq_scheme` 3→1 mapping
   that emits 'X' for modified residues shifts the alignment locally. Use
   `entity_poly` for the alignment sequence; keep `poly_seq_scheme` for
   per-position res_names/coords.
5. **Chain selection.** `chain_id` selects a specific template chain; `None` =
   auto-select the chain with the best alignment (seq_id × coverage) to the
   query — mirror the OSS auto-select rule exactly.
6. **Port the OSS keep/drop rule faithfully; don't approximate with a flat
   coverage heuristic.** OSS drops a template unless it aligns to every query
   residue — *except* its non-standard-residue cleaning branch re-aligns the
   counts, so a modified-residue template survives at partial coverage.
   Replicate the two regimes; a `matched == chain_len` heuristic is wrong on
   modified-residue templates.
7. **Unresolved residues keep their restype (coords stay NaN and are masked) —
   they are not GAP.** OSS runs the missing-backbone check *after* inserting
   canonical atom names for unresolved residues, so every aligned polymer
   residue retains its restype.
8. **Deterministic top-k for inference.** OSS inference sets `take_top_k=True`
   (not the random `TemplateSettings()` default). Hardcode top-k so the single
   supplied template is never randomly dropped.

**Validation (Phase 8 + Phase 9).** Add an L1 equivalence check on the
`template_*` tensors (atol ~1e-4) that also asserts the reference is
**non-empty** (e.g. `unit_vector` nonzero) so a match is real, not empty==empty.
Use a self-template (query == template, seq_id=1) to exercise the full alignment
path, a multichain hetero-dimer to exercise per-chain mapping, and — where the
production format ships real templates — that curated set (for OF3, the NIM
`data_with_template` targets). For Phase 9, report no-template vs with-template
lDDT on **leakage-free homolog** templates (distinct PDB entries, not the
target's GT); a working path shows a non-zero template mask and measurable lDDT
gains on template-amenable targets.

### Step 2 — `feature_collators.py`

Implement `FeatureCollatorBase` subclasses for each ensembled transform from
your function inventory:

- Each modifies and returns the `features` dict (in-place semantics)
- Handle randomness via `context["ensemble_seed"]` + `torch.Generator` — no
  global random state
- Common collators: MSA sampling, masked MSA, nearest-neighbor clustering, extra
  MSA cropping, feature concatenation, feature selection, fixed-size padding

### Step 3 — `feature_factory.py`

Wire everything into a declarative `FeatureFactoryBase`:

```python
class FeatureFactory(FeatureFactoryBase):
    pre_init = pre_init                    # seed setup
    feature_generator_specs = [...]        # from Step 1
    features_merger_func = default_context_and_feature_merger
    feature_collator_specs = [...]         # from Step 2, wrapped in SampleRepeater
```

**Recycling/ensemble loop:** Use `SampleRepeater` (from
`bionemo_ir.pipeline.models.openfold2.feature_factory`) to wrap the
ensembled collator specs. Do NOT implement your own recycling loop.

```python
FeatureCollatorSpec(
    name="repeater",
    functor=SampleRepeater,
    kwargs={
        "feature_collator_specs": ensembled_collator_specs,
        "get_n_iters": lambda config: config.max_recycling_iters + 1,
    }
)
```

______________________________________________________________________

## Phase 5 — Implement PostProcessor

Create `postprocessor.py` with a `PostProcessorBase` subclass.

### Requirements

- **Must return `FoldingOutput`**
  (`bionemo_ir.data.schemas.FoldingOutput`) — this is the enforced output
  schema. Every postprocessor must produce a `FoldingOutput` instance, no
  exceptions.
- **If the model outputs multiple samples** (e.g., multiple diffusion
  trajectories, recycling candidates, or ensemble predictions), the
  postprocessor must **select the best sample** based on confidence scores
  (e.g., highest mean pLDDT, or highest pTM/ipTM) and return a single
  `FoldingOutput`. Do not return lists of outputs or leave sample selection to
  the caller.

### `FoldingOutput` schema (from `bionemo_ir/data/schemas/basic.py`)

| Field             | Shape                         | Required | Description                           |
| ----------------- | ----------------------------- | -------- | ------------------------------------- |
| `atom_positions`  | `(num_res, num_atom_type, 3)` | **yes**  | Cartesian coordinates in angstroms    |
| `residue_types`   | `(num_res,)`                  | **yes**  | Amino-acid type as int (0-20, 20='X') |
| `atom_mask`       | `(num_res, num_atom_type)`    | **yes**  | Binary mask for atom presence         |
| `residue_indices` | `(num_res,)`                  | **yes**  | PDB residue indices                   |
| `b_factors`       | `(num_res, num_atom_type)`    | optional | Temperature factors                   |
| `chain_indices`   | `(num_res,)`                  | optional | Chain indices (multimer)              |
| `plddt`           | `(num_res,)`                  | optional | Per-residue confidence (0-100)        |
| `ptm`             | scalar                        | optional | Predicted TM-score (0-1)              |
| `iptm`            | scalar                        | optional | Interface pTM (0-1, multimer)         |
| `pae`             | `(num_res, num_res)`          | optional | Predicted aligned error matrix        |
| `max_pae`         | scalar                        | optional | Max PAE for normalization             |

### Implementation

```python
class PostProcessor(PostProcessorBase):
    def __call__(self, batch: dict[str, Any],
                 output: dict[str, Any]) -> FoldingOutput:
        # 1. If multiple samples, select best by confidence
        #    e.g., best_idx = output["plddt"].mean(dim=-1).argmax()
        #    then index all output tensors by best_idx
        # 2. Compute confidence scores (pLDDT, pTM, iPTM, PAE)
        # 3. Extract atom positions and masks
        # 4. Determine chain indices (from batch)
        # 5. Convert tensors to numpy arrays
        return FoldingOutput(
            atom_positions=...,  # np.ndarray
            residue_types=...,   # np.ndarray
            atom_mask=...,       # np.ndarray
            residue_indices=..., # np.ndarray
            b_factors=...,
            chain_indices=...,
            plddt=...,
            ptm=...,
            iptm=...,
            pae=...,
            max_pae=...,
        )
```

______________________________________________________________________

## Phase 6 — Register in Registry

### Step 1 — Create factory class

Add a `ModelComponentsFactory` subclass in `bionemo_ir/registry.py`:

```python
class NewModelFactory(ModelComponentsFactory):
    @classmethod
    def get_model_class(cls) -> Type[nn.Module]:
        from bionemo_ir.models.newmodel import NewModel
        return NewModel

    @classmethod
    def get_tokenizer(cls) -> "TokenizerBase":
        from bionemo_ir.pipeline.models.newmodel.tokenizer import Tokenizer
        return Tokenizer()

    @classmethod
    def get_feature_factory(cls) -> "FeatureFactoryBase":
        from bionemo_ir.pipeline.models.newmodel.feature_factory import FeatureFactory
        return FeatureFactory()

    @classmethod
    def get_postprocessor(cls) -> Type["PostProcessorBase"]:
        from bionemo_ir.pipeline.models.newmodel.postprocessor import PostProcessor
        return PostProcessor

    @classmethod
    def get_trt_building_modules(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_supported_model_names(cls) -> list[str]:
        return [SupMat.NewModel_1, SupMat.NewModel_2]
```

### Step 2 — Add to `register_all_factories()`

Append the new factory to the `factories` list in `register_all_factories()`.

______________________________________________________________________

## Phase 7 — Establish OSS Baseline Metrics (in WORKDIR)

### ⚠️ MANDATORY: OSS baseline MUST be established BEFORE any BioIR testing ⚠️

**This phase is a hard prerequisite.** You must run the OSS model on ALL test
samples and collect metrics vs ground truths BEFORE proceeding to Phase 8 or
Phase 9. Without OSS baseline metrics, there is nothing to compare BioIR
results against. Do NOT skip, defer, or partially complete this phase.

### Input data and ground truths

- **Input samples**: `examples/data/samples/` —
  **ALL in-scope samples from `sample_manifest.json`** across protein, RNA, DNA,
  ligand, and mixed-complex categories
- **Ground truth structures**: `examples/data/samples/gt/` — reference PDB/CIF
  structures for accuracy scoring
- **MSAs**: The samples include pre-computed MSA files (A3M).
  **Always use the provided MSAs** when running the OSS model — do NOT run
  without MSAs or with `use_msa_server=False`. The MSAs are critical for
  prediction quality; running without them produces unrealistically low metrics
  that are meaningless as a baseline.

Run on **every sample** — not a subset. Use the full input including sequences
AND MSAs.

### Checkpoints

**Always use checkpoints from the BioIR hub system**
(`bionemo_ir/hubs/`). The hub supports local checkpoints (via environment
variables like `BOLTZ2_CKPT`) and remote checkpoints (HuggingFace hub). See:

- `bionemo_ir/hubs/support_matrix.py` — `FoldingSupportMatrix` lists all
  supported model names
- `bionemo_ir/hubs/local.py` — `LOCAL_CHECKPOINTS` maps model names to env
  vars and loading config
- `bionemo_ir/hubs/checkpoint.py` — `load_weights(name, hub="local"|"hf")`
  loads via either hub

When running the OSS pipeline, the OSS code needs to load the
**same checkpoint**. If the OSS code cannot load the BioIR hub checkpoint
directly (different format, different key names, etc.):

1. **Warn the user** — explain the format mismatch and what conversion is
   needed.
1. **Try to convert** — write a conversion script in
   `$WORKDIR/debug/convert_ckpt.py`.
1. **Document the issue** — record in `$WORKDIR/NOTES.md` which checkpoint was
   used, any conversion steps.
1. **Never download a separate checkpoint** without asking the user.

### Inference config parity — OSS and BioIR MUST use identical settings

**⚠️ The OSS baseline and BioIR runs MUST use the same inference-time
configuration.** A metric comparison is meaningless if the two runs differ in
any parameter that affects prediction quality or output structure. Before
running either side, discover and lock down these parameters:

| Parameter                        | What it controls                                                                    | Example values                                 |
| -------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------- |
| **Diffusion samples**            | Number of diffusion trajectories generated per input                                | `num_samples=1`, `num_samples=5`               |
| **Diffusion steps**              | Number of denoising steps per trajectory                                            | `num_steps=200`, `diffusion_steps=50`          |
| **Recycling / trunk iterations** | Number of model trunk iterations (Evoformer recycling, structure module iterations) | `max_recycling_iters=3`, `num_trunk_iters=4`   |
| **Seeds / determinism**          | Random seed for diffusion, sampling, dropout                                        | `seed=0`, `random_seed=42`                     |
| **Sample selection**             | How the "best" sample is chosen from multiple trajectories                          | `sample_selection="confidence"`, `best_of_n=5` |
| **Cropping / chunking**          | Whether long sequences are cropped or processed in chunks                           | `max_tokens=384`, `crop_size=256`              |
| **MSA depth**                    | Maximum number of MSA sequences used                                                | `max_msa_clusters=128`, `max_extra_msa=1024`   |
| **Precision**                    | Model inference dtype                                                               | `bf16`, `fp32`                                 |

**How to ensure parity:**

1. **Read the OSS inference config/CLI defaults.** Print the full resolved
   config before the run and save it to `$WORKDIR/ref_data/oss_config.json`.
1. **Identify every parameter that affects the output** — not just accuracy, but
   also the structure of the output (number of atoms, chains, samples returned).
1. **Record the exact values in `$WORKDIR/NOTES.md`** under a
   `## Inference Config` section, in a two-column table:
   `| Parameter | Value |`.
1. **When running BioIR in Phase 9, use the SAME values.** Cross-check against
   `oss_config.json` before starting. If the BioIR config uses different names
   for the same parameter, create a mapping table.

**Common pitfalls:**

- OSS defaults to `num_samples=5` but BioIR defaults to `num_samples=1` →
  different output quality (best-of-5 vs single shot)
- OSS uses `max_recycling_iters=3` but BioIR config calls it `num_recycles=3`
  → off-by-one if one means "3 iterations" and the other means "3 additional
  iterations after the first"
- OSS runs diffusion with 200 steps by default but BioIR uses 50 →
  significantly different prediction quality
- OSS uses `seed=42` but BioIR uses `seed=0` → different diffusion
  trajectories, different "best" sample selected
- OSS enables `use_msa=True` by default but the BioIR run script forgets to
  pass MSAs → catastrophic accuracy drop

**If a parameter cannot be matched exactly** (e.g., BioIR does not yet support
a feature the OSS model uses), document the difference and its expected impact
on metrics in `$WORKDIR/NOTES.md`.

### Step 1 — Set up metrics tooling (OpenStructure via Miniforge)

Phases 7 and 9 use **OpenStructure** (`ost compare-structures`) to compute lDDT,
TM-score, and other structural metrics. OpenStructure requires its own conda
environment — it cannot be pip-installed into the main environment. Use
[Miniforge][miniforge], which defaults to conda-forge and carries no
Anaconda-channel terms of service.

**Check if already installed:**

```bash
OST_CMD=${OST_CMD:-$(pwd)/miniforge3/bin/ost}
if command -v "$OST_CMD" &>/dev/null; then
    echo "OpenStructure found: $($OST_CMD --version)"
else
    echo "OpenStructure NOT found — installing..."
fi
```

**Auto-install Miniforge + OpenStructure** (if not found):

```bash
MINIFORGE_DIR=${MINIFORGE_DIR:-$(pwd)/miniforge3}

# 1. Install Miniforge to a local directory (no root required)
if [ ! -f "$MINIFORGE_DIR/bin/conda" ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$MINIFORGE_DIR"
    rm /tmp/miniforge.sh
fi

# 2. Install OpenStructure (conda-forge is Miniforge's default channel)
"$MINIFORGE_DIR/bin/conda" install -y -c conda-forge openstructure

# 3. Verify
OST_CMD="$MINIFORGE_DIR/bin/ost"
"$OST_CMD" --version
```

Record `$OST_CMD` in `$WORKDIR/NOTES.md`. All subsequent
`ost compare-structures` calls use this path. The Miniforge installation is
local and does not affect the main Python environment.

[miniforge]: https://github.com/conda-forge/miniforge

### ⚠️ MANDATORY: Use ONLY `ost compare-structures` for ALL metrics ⚠️

**Never implement your own lDDT, TM-score, DockQ, or any other structural
metric.** Always use `ost compare-structures` — it is the standard tool for
structural comparison in the protein structure prediction community.
Hand-written metric code (e.g., CA-only lDDT, naive distance calculations) will
produce incorrect results because:

- lDDT is an all-atom metric with specific inclusion radius and threshold
  definitions
- Proper structural alignment and chain mapping are required
- Sequence mapping, residue numbering, and atom naming must be handled correctly
- Homopolymer/multimer chain permutations need special handling

If `ost compare-structures` fails or produces unexpected results (e.g., lDDT=0
due to chain mapping issues), **debug the OST invocation** (check chain IDs,
file formats, mapping flags) — do NOT fall back to a hand-written scorer.

**Usage pattern** (used in Steps 3-4, and Phase 9 Step 1):

```bash
"$OST_CMD" compare-structures \
    -m prediction.pdb -r ground_truth.pdb \
    -o scores.json \
    --lddt --tm-score --fault-tolerant \
    --min-pep-length 4 --min-nuc-length 4
```

If chain mapping fails (e.g., chain IDs differ between prediction and ground
truth), use `--chain-mapping` to specify explicit mapping, or preprocess files
to normalize chain IDs.

### Step 2 — Set up OSS environment & disable incompatible kernels

Install the OSS model's dependencies so the OSS inference script can run:

```bash
pip install <oss_package> --no-deps  # or install from $OSS_ROOT
# Install any additional OSS-only dependencies (not needed for BioIR)
```

**⚠️ CRITICAL: Disable accelerated kernels that cannot run on this system.**

OSS models often ship with optimized CUDA/Triton kernels, custom attention
backends, or compiled extensions that require specific GPU architectures (e.g.,
SM90 for Hopper, SM80 for Ampere) or optional packages (`flash-attn`,
`deepspeed`, `xformers`, `triton`). If any of these are missing or incompatible,
the OSS model will crash — **and no baseline metrics will be produced**.

**The goal is to make the OSS model produce correct results first.** Performance
does not matter for the baseline — only correctness. Always prefer a slow
reference path over a fast path that crashes.

**Discovery process:**

1. **Search the OSS config/CLI for kernel selection flags.** Common patterns:

   ```python
   # Look for these in OSS config classes, CLI args, or environment variables:
   use_flash_attn = False          # Flash Attention (needs flash-attn package + SM80+)
   use_triton = False              # Triton kernels (needs triton + compatible GPU)
   use_deepspeed = False           # DeepSpeed kernels
   use_xformers = False            # xFormers memory-efficient attention
   attn_backend = "reference"      # or "math", "eager" — the PyTorch fallback
   use_lma = False                 # Low-memory attention (custom kernel)
   compile = False                 # torch.compile — may trigger Triton codegen
   use_custom_kernels = False      # Generic toggle for custom CUDA extensions
   ```

1. **Check what GPU is available and what the OSS code requires:**

   ```python
   import torch
   cap = torch.cuda.get_device_capability()
   print(f"GPU: {torch.cuda.get_device_name()}, SM{cap[0]}{cap[1]}")
   # SM90 = H100/H200, SM80 = A100, SM89 = L40/RTX4090, SM86 = A40/RTX3090
   ```

1. **Try a dry run.** Run the OSS model on a single small sample first. If it
   crashes with errors like:

   - `CUDA error: no kernel image is available` → kernel compiled for wrong SM
     arch
   - `ModuleNotFoundError: No module named 'flash_attn'` → missing optional
     package
   - `RuntimeError: Triton...` → Triton kernel incompatibility
   - `AssertionError: ...requires SM80+` → GPU architecture check

   Then search the OSS code for the config/env var that controls that kernel and
   disable it.

1. **Common disable patterns per model family:**

   | OSS Model  | Flag / Env Var                                       | What it disables                                          | Fallback            |
   | ---------- | ---------------------------------------------------- | --------------------------------------------------------- | ------------------- |
   | OpenFold   | `--use_flash=false`                                  | Flash Attention                                           | PyTorch SDPA        |
   | OpenFold   | `--use_deepspeed_evo_attention=false`                | DeepSpeed DS4Sci_EvoformerAttention                       | PyTorch attention   |
   | Boltz      | `BOLTZ_USE_FLASH=0` or config `use_flash_attn=False` | Flash Attention                                           | PyTorch SDPA        |
   | AlphaFold3 | env `XLA_FLAGS`, config flags                        | JAX/XLA custom kernels                                    | Reference NumPy/JAX |
   | ESMFold    | `--chunk_size=-1`                                    | Chunked attention (avoids OOM but may use custom kernels) | Full attention      |
   | General    | `torch.backends.cuda.enable_flash_sdp(False)`        | PyTorch's built-in flash SDP                              | Math SDP fallback   |
   | General    | `CUDA_VISIBLE_DEVICES=0`                             | Multi-GPU / NCCL issues                                   | Single GPU          |

1. **Document every flag you set** in `$WORKDIR/NOTES.md` under a
   `## OSS Kernel Config` section:

   - Which flags were changed and why
   - What error they resolved
   - The original default value
   - Whether the fallback is mathematically equivalent (it almost always is —
     just slower)

**Principle:** the OSS baseline must run to completion on every sample. If a
kernel crashes, find the config to disable it. If no config exists, monkey-patch
or set the environment variable before import. Never skip a sample because of a
kernel issue.

Document installed packages and kernel config overrides in `$WORKDIR/NOTES.md`.

### Step 3 — Run OSS model on ALL test samples

Create `$WORKDIR/debug/run_oss_e2e.py`. This script:

1. Enumerates **every in-scope** sample from
   `$WORKDIR/ref_data/sample_manifest.json`.
1. Runs the **OSS inference script** (e.g., `run_openfold.py predict`,
   `boltz predict`) with the checkpoint on each sample.
1. Produces predicted structures (PDB/CIF output) from the OSS model.
1. **Every sample must produce output** — if the OSS model crashes or fails on a
   sample, debug until it runs. If the crash is from an accelerated kernel, go
   back to Step 2 and disable it.

```bash
cd $WORKDIR && python debug/run_oss_e2e.py \
    --samples ../../examples/data/samples/ \
    --checkpoint $CKPT_PATH \
    --output ref_data/oss_predictions/
```

### Step 4 — Score ALL OSS predictions vs ground truths

Score **every** OSS prediction against its corresponding ground truth using
`$OST_CMD` (set up in Step 1):

| Metric       | Use case                                                                      | Tool                                             |
| ------------ | ----------------------------------------------------------------------------- | ------------------------------------------------ |
| **lDDT**     | Per-residue local distance accuracy for supported polymer/complex predictions | `$OST_CMD compare-structures --lddt`             |
| **DockQ**    | Interface quality for multimer complexes                                      | `DockQ` package                                  |
| **GDT-TS**   | Global distance test (optional, usually for protein-only monomer cases)       | `$OST_CMD compare-structures --global-dist-test` |
| **TM-score** | Template modeling score (optional)                                            | `$OST_CMD compare-structures --tm-score`         |

Save per-sample metrics to `$WORKDIR/ref_data/oss_metrics.json`:

```python
import json
import subprocess

OST_CMD = "<repo_root>/miniforge3/bin/ost"  # from Step 1

def score_with_ost(prediction_path, reference_path, output_json):
    """Run ost compare-structures and parse the JSON output."""
    cmd = [
        OST_CMD, "compare-structures",
        "-m", prediction_path,
        "-r", reference_path,
        "-o", output_json,
        "--lddt", "--tm-score",
        "--fault-tolerant",
        "--min-pep-length", "4",
        "--min-nuc-length", "4",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    with open(output_json) as f:
        return json.load(f)

metrics = {}
for sample_id in all_sample_ids:
    pred_path = f"ref_data/oss_predictions/{sample_id}.pdb"
    gt_path = f"examples/data/samples/gt/{sample_id}.pdb"
    scores_json = f"ref_data/oss_predictions/{sample_id}_scores.json"
    raw = score_with_ost(pred_path, gt_path, scores_json)
    lddt = raw.get("lddt", raw.get("lddt_global", None))
    tm = raw.get("tm_score", None)
    metrics[sample_id] = {"lddt": lddt, "tm_score": tm}
    print(f"  {sample_id}: lDDT={lddt:.4f}" + (f"  TM={tm:.4f}" if tm else ""))

with open("$WORKDIR/ref_data/oss_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
```

**Verify:** `oss_metrics.json` must contain metrics for ALL in-scope samples
from `sample_manifest.json`. Print the summary table and sanity-check that lDDT
values are reasonable (typically 0.3-0.9 for structure prediction models). Also
write a provenance entry for every prediction and metric JSON, including SHA256
and the exact `ost compare-structures` command.

### Step 5 — Dump OSS tensor artifacts for Phases 8-9

While running the OSS model (Steps 3-4),
**also dump two sets of tensor artifacts** that are essential for debugging the
BioIR pipeline in later phases:

**Artifact 1: OSS input features** (from OSS data pipeline) → used in Phase 8 to
fix BioIR data pipeline

Run the **OSS featurization code** (NOT BioIR) on each sample input and save
the feature dict. These are the reference features that the BioIR pipeline
must reproduce.

```python
# Run OSS featurization on each sample
oss_features = run_oss_featurization(sample)  # structure + conformer + MSA features
torch.save(oss_features, f"$WORKDIR/ref_data/oss_features/{sample_id}.pt")
```

**Artifact 2: OSS model output tensors** (from OSS model inference using OSS
features) → used in Phase 9 to fix postprocessor + writer

Feed the OSS features into the model and save both the input batch and the raw
model output tensors. This allows debugging the postprocessor and writer without
re-running expensive model inference.

```python
# Run model with OSS features
batch_gpu = {k: v.to("cuda") for k, v in oss_features.items()}
with torch.no_grad():
    output = model(batch_gpu)

# Save both batch (input) and output for later reuse
torch.save({
    "batch": {k: v.cpu() for k, v in oss_features.items()},
    "output": {k: v.cpu() for k, v in output.items()},
}, f"$WORKDIR/ref_data/oss_model_outputs/{sample_id}.pt")
```

**Why both artifacts are needed:**

- If the BioIR **data pipeline** produces wrong features, you compare
  `oss_features/{id}.pt` against BioIR output to find which feature key
  diverges.
- If the BioIR **postprocessor or writer** is broken, you load
  `oss_model_outputs/{id}.pt`, run the postprocessor on the OSS model output,
  and check if the written CIF/PDB scores correctly with OST. This isolates
  postprocessor/writer bugs from data pipeline bugs.
- Re-running inference is expensive. Dumped tensors let you iterate on
  postprocessor + writer fixes without GPU time.

### Step 6 — Verify ALL samples are covered

**Before proceeding, verify the count.** Load
`$WORKDIR/ref_data/sample_manifest.json`, count every in-scope sample, then
verify `oss_metrics.json` has an entry for each one. Print a table showing every
sample with its category and metric — any missing in-scope sample is a hard
failure.

```python
import json
from pathlib import Path

# Count all samples
sample_ids = set()
for f in Path("examples/data/samples").rglob("*.json"):
    with open(f) as fp:
        data = json.load(fp)
    for req in (data if isinstance(data, list) else [data]):
        if "input_id" in req:
            sample_ids.add(req["input_id"])

# Verify metrics
with open("$WORKDIR/ref_data/oss_metrics.json") as f:
    metrics = json.load(f)

missing = sample_ids - set(metrics.keys())
assert not missing, f"MISSING samples in oss_metrics.json: {missing}"
print(f"ALL {len(sample_ids)} samples covered in oss_metrics.json")
for sid in sorted(sample_ids):
    m = metrics[sid]
    print(f"  {sid}: lDDT={m['lddt']:.4f}")
```

**Do NOT proceed to Phase 8 until this verification passes — every sample under
`examples/data/samples/` must have an entry in `oss_metrics.json`.** If any
sample is missing, go back to Step 2 and debug why that sample failed.

______________________________________________________________________

## Phase 8 — Validate Equivalence — Level 1 (in WORKDIR)

**Level 1 testing: feature-level equivalence.** Compare BioIR pipeline outputs
tensor-by-tensor against **OSS-generated** reference outputs. This must pass
before Level 2.

### ⚠️ CRITICAL: Reference features come from OSS, not BioIR

**The reference `.pt` files used for equivalence testing must be generated by
running the OSS data pipeline** on the same inputs. Do NOT generate references
by running the BioIR pipeline and saving its output — that only tests
self-consistency, not correctness.

A self-consistent test (BioIR vs BioIR) will always pass, even if the
features are completely wrong. The entire point of equivalence testing is to
verify that BioIR produces the **same features as OSS**. If you generate
references from BioIR, you are testing nothing.

**How to generate correct references:**

1. Run the **OSS data pipeline** (not the BioIR pipeline) on each sample
   input.
1. Save the OSS feature dict as `<input_id>.pt` via `torch.save()`.
1. The equivalence test then runs the BioIR pipeline on the same input and
   compares against the OSS `.pt` file.

If the OSS pipeline is not runnable in the current environment (dependency
issues), the reference generation script must use the
**OSS featurization primitives** (e.g., the OSS's `featurize_structure_of3`,
`featurize_reference_conformers_of3`, MSA processing) to produce the reference
features — not BioIR's reimplementation.

### ⚠️ CRITICAL: The OSS reference must match the production parser path ⚠️

OSS pipelines often support **multiple input formats per file kind** (different
file types for sequences, MSAs, structures, templates, ligands, etc.). Each
format typically routes to a *different parser*, and the parsers extract
different metadata — annotations, identifiers, masks, alternate states,
ordering, secondary fields. The downstream featurizer then
*branches on what the parser produced*. If your reference script feeds OSS via a
format whose parser drops a field the production parser keeps, the OSS run
completes "successfully" but its output disagrees with BioIR by design —
because BioIR matches the production parser and your reference script doesn't.

This is one of the highest-leverage debugging mistakes. The symptom looks like a
BioIR bug (off-by-one rows, a flag in the wrong place, a value that's there in
one pipeline and not the other) but the root cause is upstream of both
pipelines: the reference was built from the wrong format.

**Discovery: pick the parser the production code uses, then back-derive the
format your reference script must write.**

1. **Trace the production OSS CLI / inference entry point** from input file to
   feature tensor. Note every parser called and the format each one consumes
   (`parse_*` functions, custom loaders, dataset classes, vendor SDKs).
1. **For each parser, list the fields it extracts that the featurizer *reads*.**
   Common asymmetries: identifiers that only one format carries, per-row keys,
   modification annotations, alternate locations, numbering offsets, taxonomy /
   clustering metadata, ordering of repeats. Two parsers for "the same file
   kind" frequently produce subtly different downstream features.
1. **Check what your BioIR stage parser produces.** If BioIR only parses
   format A but production OSS routes through format B, your reference dump must
   also go through format B — or you're comparing apples to oranges.
1. **If the formats differ, synthesise the production format on the fly in your
   reference dump script.** Convert BioIR's parsed objects back to the
   production format and feed *that* to OSS. Cache the converted files under
   `$WORKDIR/ref_data/oss_<format>_inputs/` with provenance entries so the
   conversion is reproducible and auditable. The conversion code itself is part
   of the equivalence-test surface; review it as carefully as the pipeline code.
1. **When the source data lacks a field the production format requires,
   synthesise it deterministically** — e.g. assign sequential identifiers, fill
   missing annotations with the OSS production default, derive a value from
   position. Document the synthesis rule in `implementation-notes.md` so a
   future debugger can tell synthesised values from real ones.

**Symptoms that suggest a format-alignment problem rather than a pipeline bug:**

- BioIR produces N more rows than OSS *and* the OSS reference value at the
  missing positions is the parser's "missing field" sentinel (e.g. `-1`, `None`,
  all-zeros).
- A boolean flag in the reference is uniformly `0` for a feature BioIR
  produces with mixed `0`/`1`.
- The OSS reference is unusually small / empty for inputs where BioIR has
  substantial data — and re-running OSS via a richer input format produces
  different (larger / non-empty) output.

**Provenance is the audit trail.** Every reference artifact's provenance entry
must record the source format and which OSS parser was used (`"source": "oss"`,
`"oss_parser": "..."`, `"oss_input_format": "..."`). When an equivalence
regression later turns up, format mismatch is the first place to look — and the
provenance line tells you immediately whether it's possible.

### Stochastic features

For stochastic features (e.g., `ref_pos` with random augmentation), either:

- Fix the random seed identically in both pipelines, OR
- Exclude stochastic features from exact comparison and validate them via
  statistical tests (Step 4)

See [test-equivalence.md](test-equivalence.md) for the full runnable test
script.

### Step 1 — Generate OSS reference features

Create `$WORKDIR/debug/dump_oss_features.py` that runs the
**OSS featurization code** on each sample and saves the output as `.pt` files.

**This script must:**

- Import ONLY from the OSS package (e.g.,
  `from openfold3.core.data.pipelines.featurization...`) or standard libraries
  (biotite, numpy, torch)
- NOT import anything from `bionemo_ir.pipeline.models.*` — this is the
  code under test, not the reference
- Run the OSS featurization functions (tokenization, structure featurization,
  conformer featurization, MSA featurization) on the same inputs
- Save features as `$WORKDIR/ref_data/oss_features/<input_id>.pt`
- Write one provenance entry per artifact to
  `$WORKDIR/ref_data/provenance.jsonl` with SHA256, source path, command, OSS
  commit/hash, checkpoint/config identifiers, and `source="oss"`

**Verification:** After generating, validate every `.pt` file, not a spot-check.
For each in-scope sample, assert that the feature keys, shapes, dtypes, value
ranges, and SHA256 are recorded. Compare against the OSS model's actual
inference inputs when the OSS inference path exposes them.

### Step 2 — Write equivalence test

Create `$WORKDIR/tests/test_equivalence.py`. This test:

- Loads **OSS-generated** reference `.pt` files from
  `$WORKDIR/ref_data/oss_features/`
- Runs the **BioIR** pipeline (tokenizer → generators → collators) on the same
  inputs
- Compares BioIR output tensors against OSS reference tensors
- The test must NEVER generate its own reference — it only reads pre-generated
  OSS `.pt` files
- Verifies every loaded reference has a matching `source="oss"` provenance entry
  and fails if provenance is missing or ambiguous

**The test fails if:**

- Any feature key is missing in BioIR output that exists in OSS reference
- BioIR output contains unexpected extra keys not documented in
  `$WORKDIR/NOTES.md`
- Any deterministic feature tensor differs beyond `atol` tolerance
- Tensor shapes differ
- The reference `.pt` files don't exist (means Step 1 wasn't run)
- The processed sample IDs differ from `sample_manifest.json`

### Step 3 — Bottom-up validation

Test each stage independently before testing the full pipeline. Do NOT proceed
to the next level until the current level passes.

1. **Tokenizer stage** — context generator + transforms: compare output dict
   keys and tensor shapes against OSS reference.
1. **Feature generators** — run each generator individually: compare new feature
   keys/values against OSS reference.
1. **Feature collators** — run collators with identical seeds: compare against
   OSS reference.
1. **Full pipeline** — run complete `generate_feature()` and compare all output
   tensors against OSS `.pt` files.

### Step 3 — Numerical comparison criteria

- Fix random seeds identically in both pipelines: `init_env={"random_seed": 0}`
- Check tensor shapes match before comparing values
- Use `atol=1e-5` for float comparisons (numerical precision differences are
  expected)
- For stochastic features (MSA sampling, masking), seed alignment is critical
- Report max absolute difference per feature key on failure
- **Never skip any tensor.** Every feature key must be validated — no exceptions
- Tolerances must be declared once in the test file and mirrored in
  `$WORKDIR/NOTES.md`. Changing tolerances after failures requires user
  approval.

### Step 4 — Statistical tests for stochastic features

Some features are inherently stochastic (e.g., `ref_pos` with random
augmentation, MSA masks, bert masks, random crops). These will **never** match
exactly between OSS and BioIR runs.
**Do NOT skip them — validate them with statistical and structural tests.**

**Every stochastic feature must be tested. Mark it as stochastic and apply ALL
tests below. Never leave it unvalidated.**

#### Test 1: Shape and dtype match

The stochastic feature must have the same shape and dtype as the OSS reference.
Non-negotiable.

#### Test 2: Value range and constraint tests

Verify the tensor satisfies the same structural constraints as OSS:

- **Value range**: same min/max bounds (e.g., coordinates within reasonable
  range, probabilities in \[0,1\])
- **Sparsity**: same fraction of zeros/nonzeros
- **Per-residue structure**: for coordinate features like `ref_pos`, each
  residue's atoms should form physically valid geometry

#### Test 3: Internal geometry test (for coordinate features)

For stochastic coordinate features like `ref_pos` that apply random rotation +
translation:

- **Intra-residue pairwise distances must match exactly** — rotation/translation
  preserves internal geometry
- Compare pairwise atom distances within each residue between BioIR and OSS

```python
def test_ref_pos_internal_geometry(trt_ref_pos, oss_ref_pos, ref_space_uid):
    """Verify intra-residue geometry is identical despite different rotations."""
    for uid in torch.unique(ref_space_uid):
        trt_atoms = trt_ref_pos[ref_space_uid == uid]
        oss_atoms = oss_ref_pos[ref_space_uid == uid]
        trt_pdist = torch.cdist(trt_atoms.unsqueeze(0), trt_atoms.unsqueeze(0)).squeeze()
        oss_pdist = torch.cdist(oss_atoms.unsqueeze(0), oss_atoms.unsqueeze(0)).squeeze()
        torch.testing.assert_close(trt_pdist, oss_pdist, atol=1e-4, rtol=1e-4,
            msg=f"ref_pos internal geometry differs for residue uid={uid}")
```

#### Test 4: Per-tensor mean & std comparison (single run)

Even from one BioIR and one OSS run with the same seed, the
**summary statistics** of a stochastic tensor should be close. This catches
subtle bugs (off-by-one slicing, dtype drift, wrong distribution) that the
geometry/shape tests miss, and it is cheap — no multi-seed sweep needed.

Compute, for each stochastic feature:

- **Element-wise mean**: `tensor.mean()` cast to float64 to avoid bf16/fp16
  drift.
- **Element-wise std** (unbiased=False): `tensor.std(unbiased=False)`.
- **Per-axis mean/std along the stochastic axis** (e.g. the random-rotation axis
  for `ref_pos`, the MSA-row axis for masked-MSA features). The marginal
  distribution along the deterministic axes must match even when the per-element
  values don't.

Acceptance:

| Statistic                 | Threshold                      | Why                                                  |
| ------------------------- | ------------------------------ | ---------------------------------------------------- |
| `\|mean_trt - mean_oss\|` | ≤ `0.05 * \|mean_oss\| + 1e-3` | Relative + small additive for near-zero means        |
| `\|std_trt - std_oss\|`   | ≤ `0.10 * std_oss + 1e-3`      | Variance is more sensitive to seed drift; allow ~10% |
| Per-axis mean/std         | same thresholds, per slice     | Catches axis-specific bugs                           |

Record the exact mean/std values per feature in the test output (PASS/FAIL line)
so regressions are visible at a glance — not just the boolean.

```python
def compare_stochastic_stats(
    trt: torch.Tensor,
    oss: torch.Tensor,
    name: str,
    mean_rtol: float = 0.05,
    mean_atol: float = 1e-3,
    std_rtol: float = 0.10,
    std_atol: float = 1e-3,
) -> tuple[bool, str]:
    """Return (ok, summary) comparing mean/std of two stochastic tensors.

    Casts to float64 before reducing to avoid low-precision summation drift
    on large tensors. Both tensors must already have matching shape/dtype
    (those are checked separately in Test 1).
    """
    a = trt.detach().cpu().to(torch.float64)
    b = oss.detach().cpu().to(torch.float64)
    m_t, m_o = float(a.mean()), float(b.mean())
    s_t, s_o = float(a.std(unbiased=False)), float(b.std(unbiased=False))
    mean_ok = abs(m_t - m_o) <= mean_rtol * abs(m_o) + mean_atol
    std_ok = abs(s_t - s_o) <= std_rtol * abs(s_o) + std_atol
    ok = mean_ok and std_ok
    return ok, (
        f"{name}: mean trt={m_t:+.6f} oss={m_o:+.6f} "
        f"diff={m_t - m_o:+.6f} [{'OK' if mean_ok else 'FAIL'}]  "
        f"std trt={s_t:.6f} oss={s_o:.6f} "
        f"diff={s_t - s_o:+.6f} [{'OK' if std_ok else 'FAIL'}]"
    )
```

For coordinate features like `ref_pos` that also apply random rotation, the
**global** mean/std will match (rotations are zero-mean preserving), so this
test is complementary to Test 3 (internal geometry). For MSA features like
`bert_mask` / `msa_mask`, the **mask fraction** ≈ mean of the boolean tensor — a
useful integrity check that the stochastic masking ratio is right.

If a stochastic feature has natural per-axis structure (e.g. `ref_pos` has shape
`(N_atoms, 3)` — the random translation contributes to the mean along axis 0 but
rotation preserves per-axis std), report mean/std along the most informative
axis too:

```python
def compare_axis_stats(trt, oss, axis, name, **thresh) -> tuple[bool, str]:
    a, b = trt.to(torch.float64), oss.to(torch.float64)
    return compare_stochastic_stats(a.mean(dim=axis), b.mean(dim=axis),
                                    f"{name}.mean[axis={axis}]", **thresh)
```

#### Test 5 (optional): Multi-seed distribution test

When the budget allows, also run both pipelines N times (N≥20) with different
seeds, collect per-run mean/std, and compare:

- **Mean of means**: ≤ `0.05 * |mean_oss|` (looser since you now have a sample
  distribution)
- **Std of means** (between-run): BioIR std ≤ 1.5× OSS std
- **KS test**: p-value > 0.01 on the flattened element distribution (same
  distribution)

Skip this when iteration cost is high (e.g. the OSS reference requires GPU
inference). Test 4 alone catches most regressions; the multi-seed sweep is for
borderline cases where Test 4 passes but you suspect distributional drift.

### Step 5 — Equivalence test must validate ALL features

The equivalence test must explicitly categorize every feature key as either:

- **Deterministic** — exact match required (`atol=1e-4`)
- **Stochastic** — statistical + structural tests from Step 4 required

No feature may be left uncategorized. The test report must print each key with
its category and PASS/FAIL status:

The categorization table must live in the test file, not in ad-hoc runtime
logic. It must include a one-line reason for every stochastic classification. If
a feature was deterministic in OSS but noisy in BioIR, treat that as a bug,
not as a stochastic feature.

```text
  token_index:      DETERMINISTIC  MATCH  (max_diff=0.0000)
  restype:          DETERMINISTIC  MATCH  (max_diff=0.0000)
  ref_pos:          STOCHASTIC     PASS   (internal_geom=MATCH, shape=MATCH)
  ref_element:      DETERMINISTIC  MATCH  (max_diff=0.0000)
```

### Step 6 — Run ALL samples and debug loop

Run equivalence tests on **every in-scope sample** from
`$WORKDIR/ref_data/sample_manifest.json` — do not cherry-pick easy protein-only
samples when RNA/DNA/ligand or mixed-complex samples are in scope.

**Every sample must pass.** Deterministic features must match exactly.
Stochastic features must pass ALL statistical tests from Step 4. No feature may
be skipped or left unvalidated.

**If any sample fails, enter a debug loop:**

1. Identify the failing sample and the failing feature key(s).
1. Use `$WORKDIR/debug/` scripts to compare the failing feature stage by stage
   (tokenizer → generators → collators).
1. Fix the pipeline code.
1. Re-run the equivalence test on the failing sample.
1. Once the failing sample passes, re-run **all** samples to confirm no
   regressions.
1. Repeat until all samples pass.

Do NOT proceed to Phase 9 until **all** Level 1 equivalence tests pass for
**all** samples.

______________________________________________________________________

## Phase 9 — Validate E2E Metrics — Level 2 (in WORKDIR)

**Level 2 testing: end-to-end accuracy metrics.** Run the BioIR pipeline on
the **same test samples** that the OSS model was scored on in Phase 7, score
predictions against the **same ground truths**, and compare the BioIR metrics
against the **OSS baseline metrics** saved in
`$WORKDIR/ref_data/oss_metrics.json`.

### Step 1 — Run BioIR model on test samples and score vs ground truth

Create `$WORKDIR/e2e/run_trt_e2e.py`. This script does exactly what the OSS
script in Phase 7 did, but using the BioIR pipeline:

1. Loads the **same test samples** from `examples/data/samples/`.
1. Runs the **BioIR pipeline** (tokenizer → feature gen → model →
   postprocessor) on each sample.
1. Produces predicted structures via `PostProcessor` → `FoldingOutput`.
1. **Writes prediction files using BioIR writers** (see below).
1. Scores each prediction against the **same ground truths** in
   `examples/data/samples/gt/`.
1. Saves per-sample metrics to `$WORKDIR/ref_data/trt_metrics.json`.

**⚠️ MANDATORY: Use `bionemo_ir.data.writers` — never create custom
writers.**

When converting `FoldingOutput` to PDB/CIF files for scoring, you **MUST** use
`PDBWriter` or `CIFWriter` from `bionemo_ir.data.writers`. Do NOT write
custom PDB/CIF writers, ad-hoc formatting code, or model-specific writers — fix
the existing writers and postprocessor instead. The production code path must
work for all models.

If the existing writers produce incorrect geometry (bad bonds, zero lDDT from
OST):

1. Fix the **postprocessor** — the atom37 mapping, coordinate assignment, or
   atom naming may be wrong.
1. Fix the **existing writer** — not by creating a new one, but by fixing the
   shared code.
1. File a bug if the writer has a genuine defect that affects multiple models.

```python
from bionemo_ir.data.writers.pdb_writer import PDBWriter
from bionemo_ir.data.writers.cif_writer import CIFWriter

writer = CIFWriter(
    res_type_mapping=res_type_mapping,
    atom_type_mapping=atom_type_mapping,
)
writer.set_output_path(f"ref_data/trt_predictions/{sample_id}.cif")
writer.write(folding_output)
```

**Dump and reuse model output tensors** to avoid re-running expensive inference:

```python
# After model forward pass, save raw output tensors
torch.save({"batch": batch, "output": output}, f"ref_data/trt_outputs/{sample_id}.pt")

# Later, reload and re-run postprocessor + writer without inference
saved = torch.load(f"ref_data/trt_outputs/{sample_id}.pt")
folding_output = postprocessor(saved["batch"], saved["output"])
writer.write(folding_output)
```

If Level 1 equivalence already passes, the features are correct. Focus debugging
on the postprocessor (FoldingOutput construction) and writer (file output), not
on re-running the model.

The `res_type_mapping` and `atom_type_mapping` are model-specific — get them
from the same source the `WriterStage` uses in the production pipeline (check
the model's registry factory or pipeline config for the `mappings` dict). These
map integer indices to `ResType`/`AtomType` objects that define canonical names
used in PDB/CIF records.

**⚠️ Config parity check — do this BEFORE running.** Load
`$WORKDIR/ref_data/oss_config.json` (saved in Phase 7) and verify that the
BioIR run uses identical values for: diffusion samples, diffusion steps,
recycling/trunk iterations, random seed, MSA depth, precision, and sample
selection strategy. See the "Inference config parity" section in Phase 7 for the
full parameter list. Print both configs side-by-side and confirm they match
before proceeding.

The e2e test **must** use the `build_processor` API — the production entry point
that wires all pipeline stages (parser → tokenizer → feature gen → engine →
writer). Create **two scripts** in `$WORKDIR/e2e/`:

### `$WORKDIR/e2e/test_build_processor_serial.py`

Tests the serial backend on ALL samples:

```python
from bionemo_ir.pipeline.processor.engine_proc import (
    EngineProcessorConfig, build_processor,
)

output_dir = "$WORKDIR/ref_data/build_processor_serial"

config = EngineProcessorConfig(
    model_source="<model_name>",
    executor_backend=None,  # serial — no Ray
    batch_size=1,
    writer_stage={"output_path": output_dir, "format": "cif"},
)
processor = build_processor(config)

# Build records for ALL samples (with __record_id and proper MSA format)
records = [{"record": req, "__record_id": req["input_id"]} for req in all_requests]

# Run through serial processor
results = processor(records)

# Score each output CIF against ground truths using $OST_CMD
# Compare against oss_metrics.json
# Print per-sample table: OSS lDDT vs TRT lDDT, diff, PASS/FAIL, bad_bonds
```

### `$WORKDIR/e2e/test_build_processor_ray.py`

Tests the Ray backend on ALL samples:

```python
import ray
from bionemo_ir.pipeline.processor.engine_proc import (
    EngineProcessorConfig, build_processor,
)

output_dir = "$WORKDIR/ref_data/build_processor_ray"

config = EngineProcessorConfig(
    model_source="<model_name>",
    executor_backend="ray",
    batch_size=1,
    concurrency=1,
    writer_stage={"output_path": output_dir, "format": "cif"},
)
processor = build_processor(config)

# Build Ray dataset from ALL samples
ds = ray.data.from_items(records)

# Run through Ray processor
result_ds = processor(ds)
results = result_ds.take_all()

# Score and compare (same as serial)
```

**Both scripts must:**

1. Run on **ALL** samples under `examples/data/samples/`
1. Score output CIFs with `$OST_CMD compare-structures` (same flags as Phase 7)
1. Compare against `oss_metrics.json` baseline
1. Print per-sample table with OSS lDDT, TRT lDDT, diff, status, bad_bonds
1. PASS only if BioIR satisfies the frozen acceptance thresholds recorded
   before debugging
1. Report ALL SAMPLES PASS or SOME SAMPLES FAILED

**Both backends must produce valid results for ALL samples.** If either fails,
debug and fix before proceeding.

```bash
cd $WORKDIR && python e2e/test_build_processor_serial.py
cd $WORKDIR && python e2e/test_build_processor_ray.py
```

### Step 2 — Compare BioIR metrics against OSS baseline metrics

Load **both** metric files and compare per-sample, per-metric:

```python
import json

with open("$WORKDIR/ref_data/oss_metrics.json") as f:
    oss = json.load(f)  # Baseline from Phase 7
with open("$WORKDIR/ref_data/trt_metrics.json") as f:
    trt = json.load(f)  # BioIR results from Step 1

for input_id in oss:
    for metric in ["lddt", "dockq"]:
        oss_val = oss[input_id].get(metric)
        trt_val = trt[input_id].get(metric)
        if oss_val is None:
            continue
        diff = abs(trt_val - oss_val)
        status = "PASS" if diff < threshold else "FAIL"
        print(f"  {status}  {input_id} {metric}: OSS={oss_val:.4f} TRT={trt_val:.4f} diff={diff:.4f}")
```

### Step 3 — Acceptance criteria

| Condition                                  | Status                       | Reason                                                                                                                         |
| ------------------------------------------ | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| BioIR score >= OSS score                   | **PASS after parity checks** | Equal or better metrics are acceptable only after feature, config, sample-set, writer, and metric-command parity are confirmed |
| BioIR score \< OSS score by less than 0.05 | **PASS**                     | Within stochastic variance (diffusion models vary per run)                                                                     |
| BioIR score \< OSS score by more than 0.05 | **FAIL**                     | BioIR is meaningfully worse — indicates pipeline defect                                                                        |

Applies to all metrics: lDDT, DockQ, TM-score, GDT-TS.

**If BioIR scores better than OSS, that is not enough by itself.** It may be a
legitimate stochastic improvement, but it can also signal different sample
selection, chain mapping, scoring inputs, or config drift. Confirm Phase 8
feature equivalence, config parity, sample-set equality, writer parity, and
metric command parity before marking it PASS.

**Only flag metric-regression FAIL when BioIR is worse than OSS by more than
the threshold.** Parity failures, missing samples, missing provenance, or
unvalidated tensors are independent hard failures even if metrics look good.

If BioIR metrics are worse beyond thresholds:

1. Check if Level 1 equivalence tests truly pass — a small feature difference
   can compound through the model.
1. Check postprocessor logic — confidence scores and atom coordinate extraction
   may differ.
1. Report the discrepancy in the summary report with exact numbers per sample.

### Step 4 — Smoke checks

Also verify basic sanity on BioIR outputs:

- No NaN/Inf in model outputs
- Predicted structures have plausible geometry (no atom clashes, reasonable bond
  lengths)
- All input samples produce output (no crashes or silent failures)
- Serial and Ray backends produce identical metrics

### Step 5 — Run ALL samples and debug loop

Run the e2e test on **every in-scope sample** from
`$WORKDIR/ref_data/sample_manifest.json`. Both OSS baseline (Phase 7) and
BioIR (Step 1) must cover the full set.

**Every sample must pass within acceptable thresholds.** Small divergences due
to stochastic components (diffusion sampling, random seeds) are expected — the
model may produce slightly different structures each run. What matters is that
the **accuracy metrics** (lDDT, DockQ) are within threshold of the OSS baseline,
not that the atom coordinates are identical.

**If any sample fails, enter a debug loop:**

1. Identify the failing sample and which metric diverges.
1. Check if the divergence is from stochastic randomness — re-run the same
   sample 3-5 times to see if the metric variance explains the gap.
1. If the divergence is systematic (consistent across runs), compare
   feature-level outputs (Level 1) for that sample to find the root cause.
1. Fix the pipeline code (feature generation, postprocessor, or model
   integration).
1. Re-run the failing sample through both OSS and BioIR to confirm the fix.
1. Once the failing sample passes, re-run **all** samples to confirm no
   regressions.
1. Repeat until all samples pass.

```bash
cd $WORKDIR && python e2e/compare_metrics.py
```

Do NOT proceed to Phase 10 until **all** samples pass Level 2 metrics within
acceptable thresholds.

### Step 6 — Verify `build_processor` produces matching metrics

**⚠️ The production `build_processor` path MUST reproduce the same metrics as
the manual pipeline in Step 1.**

The e2e script in Step 1 may call the pipeline stages directly (Tokenizer →
FeatureFactory → model.forward → PostProcessor). But in production, users run
inference via `build_processor`, which orchestrates the full pipeline including
parsing, batching, and output writing. These two paths can diverge due to
differences in: input parsing, default config values, batching behavior, seed
handling, or postprocessor invocation.

**Create `$WORKDIR/e2e/verify_build_processor.py`** that:

1. Runs `build_processor` on the same test samples with the same config (from
   `oss_config.json`).
1. Scores the `build_processor` predictions against the same ground truths using
   `$OST_CMD`.
1. Compares the `build_processor` metrics against `trt_metrics.json` (from Step
   1).
1. **They must match exactly** (within floating-point tolerance, `atol=1e-6`) —
   not just within the Phase 9 acceptance thresholds. If the manual pipeline and
   `build_processor` disagree, the pipeline integration has a bug.

```python
from bionemo_ir.pipeline.processor.engine_proc import (
    EngineProcessorConfig, build_processor,
)

config = EngineProcessorConfig(
    model_source="<model_name>",
    executor_backend=None,
    batch_size=1,
)
processor = build_processor(config)

for sample_path in all_sample_paths:
    result = processor.process(sample_path)
    # build_processor uses WriterStage internally, which writes PDB/CIF via
    # bionemo_ir.data.writers (PDBWriter/CIFWriter) — same as Step 1.
    # Score the output files with $OST_CMD, compare vs trt_metrics.json.
```

**If the metrics diverge:**

1. Check if `build_processor` uses different defaults (e.g., different
   `num_samples`, `seed`, `recycling_iters`) than the manual pipeline script.
1. Check if the input parser in `build_processor` handles MSA paths or chain IDs
   differently.
1. Check if batching or collation changes the feature dict.
1. Fix the discrepancy — either in the pipeline code or in the e2e script —
   until both paths produce identical metrics.

**Both serial and Ray backends must match.** Run the verification with
`executor_backend=None` and `executor_backend="ray"` separately and confirm all
three produce the same per-sample metrics: manual pipeline (Step 1) = serial
`build_processor` = Ray `build_processor`.

Do NOT proceed to Phase 10 until `build_processor` metrics match the manual
pipeline metrics for all samples.

______________________________________________________________________

## Phase 10 — Summary Report

Print (not file) after completion:

1. **Model overview** — model name, OSS source location, pipeline pattern (A or
   B), number of OSS functions ported
1. **Function inventory** — complete OSS function → BioIR class mapping table
1. **Files created/modified** — full paths with brief descriptions
1. **Level 1 results (equivalence)** — per-stage and full-pipeline PASS/FAIL,
   numerical tolerance details, number of features compared
1. **Level 2 results (e2e metrics)** — OSS vs BioIR metrics table (lDDT, DockQ
   per sample), PASS/FAIL per threshold
1. **Known limitations** — features not ported, approximations made, missing
   edge cases
1. **Tricky parts needing human review** — non-obvious algorithmic choices, seed
   alignment issues, places where OSS behavior was ambiguous
1. **WORKDIR location** — path and contents summary

______________________________________________________________________

## Phase 11 — Merge to Codebase

Once all tests and smoke tests pass, finalize the pipeline in the main codebase.
Tests stay in `$WORKDIR/`.

### Step 1 — Verify registry

Confirm the factory is registered and all imports resolve:

```bash
python -c "
from bionemo_ir.registry import get_tokenizer, get_feature_factory
tok = get_tokenizer('<model_name>')
ff = get_feature_factory('<model_name>')
print(f'Tokenizer: {type(tok).__name__}')
print(f'FeatureFactory: {type(ff).__name__}')
print('Registry OK')
"
```

### Step 2 — Run final validation from WORKDIR

```bash
cd $WORKDIR && python tests/test_equivalence.py --reqs ref_data/reqs.json --samples ref_data/samples/ --model <model_name>
cd $WORKDIR && python e2e/smoke_test.py --input e2e/sample_inputs/sample.json --model <model_name>
cd $WORKDIR && python e2e/verify_build_processor.py --model <model_name>
```

All three must pass: Level 1 equivalence, e2e smoke test, and `build_processor`
metric parity (Phase 9 Step 6).

### Step 3 — Clean up

- Archive or delete `$WORKDIR/debug/` scratch artifacts
- Keep `$WORKDIR/NOTES.md` if it contains useful decisions/context

### Step 4 — Commit

Stage only the codebase files (pipeline implementation + registry changes).
WORKDIR is the user's development space — do NOT commit it unless the user asks.

______________________________________________________________________

## Phase 12 — Consolidate Implementation Notes

Before ending the task, consolidate `$WORKDIR/implementation-notes.md` so it can
serve as the reusable context for future agents. Do not create a separate
chat-context file unless the user explicitly asks for one.

Ensure the implementation notes include:

- **Pipeline location** — code paths, registry entries, and model names.
- **Input schema** — supported `InputRequest` / `InputParsed` polymer types,
  including any model-specific handling for protein, RNA, DNA, CCD ligands,
  SMILES ligands, templates, MSAs, and unsupported fields.
- **How to parse inputs** — sample formats, required fields, path conventions,
  and any OSS-to-BioIR mapping decisions.
- **How to run each stage** — parser, tokenizer, feature generation, collator,
  engine, postprocessor, writer, and validation commands.
- **Design decisions and deviations** — the timestamped entries accumulated
  during implementation.
- **Open questions** — anything still needing user confirmation or future
  revision.

The final response must point to `$WORKDIR/implementation-notes.md` and
summarize any unresolved open questions. If the notes are missing entries for
decisions made during the port, append them before reporting completion.

______________________________________________________________________

## Critical Rules

### Engineering Philosophy: Understand, Then Rewrite

OSS bioinformatics code is typically **scientist-style** — written to produce
correct results, but not structured for production deployment, extensibility, or
maintainability. It often has: deeply nested functions, implicit state via dict
mutation, copy-paste across models, inconsistent naming, no separation of
concerns, global random state, and data pipeline logic tangled with model logic.

BioIR is **senior-engineer-style** — a structured pipeline framework designed
to support **any** biology model. The goal is not to replicate the OSS code's
structure; it is to **understand the algorithm** the OSS code implements, then
**rewrite it cleanly** within BioIR's architecture.

**What this means in practice:**

- **Read the OSS code to understand the math and data flow**, not to copy its
  structure. A 200-line OSS function may become 3 small, focused BioIR
  classes.
- **Name things for clarity**, not to match OSS. If the OSS calls it
  `_process_features_v2_inner`, name it `MakeAtomFeatures` in BioIR.
- **Separate concerns** that the OSS code tangles. If one OSS function does MSA
  sampling AND masking AND clustering, split those into `SampleMsa`,
  `MakeMaskedMsa`, and `NearestNeighborClusters`.
- **Use the type system.** OSS often passes `protein: dict` everywhere. BioIR
  has typed base classes (`TransformBase`, `FeatureGeneratorBase`,
  `FeatureCollatorBase`) — use them correctly to make the pipeline
  self-documenting.
- **Make each class do one thing.** A generator produces features. A collator
  modifies features. A transform normalizes data. Don't mix responsibilities.
- **Config over hardcoded values.** OSS may hardcode `max_msa = 512` inside a
  function. BioIR reads `self.config.max_msa_clusters` — configurable,
  overridable, documented.
- **Deterministic by default.** OSS may scatter `random.random()` calls. BioIR
  threads `context["ensemble_seed"]` through `torch.Generator` — reproducible,
  testable.
- **No dead code.** OSS may have training-only branches, backward-compat shims,
  and commented-out experiments. Port only what the inference pipeline needs.

**The OSS code tells you WHAT to compute. BioIR's architecture tells you HOW
to structure it. You supply the engineering judgment to bridge the two.**

### No OSS Imports

**DO NOT import or call OSS code directly.** The BioIR pipeline must be a
fresh reimplementation. Use OSS code only as a **read-only reference** to
understand algorithms, data formats, and expected behavior.

- **FORBIDDEN**: `from boltz.data.feature.featurizerv2 import Boltz2Featurizer`
- **FORBIDDEN**: `from openfold.data.data_transforms import make_seq_mask`
- **FORBIDDEN**: Wrapping OSS functions inside BioIR classes
- **FORBIDDEN**: Copy-pasting OSS code and just renaming the function to a class
  — you must understand the algorithm and rewrite it with proper structure
- **ALLOWED**: Reading OSS code to understand algorithms, then reimplementing
  them
- **ALLOWED**: Using standard libraries (numpy, torch, rdkit, etc.) that the OSS
  code also uses
- **ALLOWED**: Loading data files (.npz, .json, .pkl) that the OSS code
  produces, using standard numpy/json/pickle — these are data, not code
- **ALLOWED**: Reproducing the same constants (token lists, amino acid
  definitions, dtypes) by copying values into your own `const.py`

### Conversion Rules

1. **OSS free functions → BioIR classes**: Each `def foo(protein)` becomes a
   class with `__call__(self, batch, context)` (generators/collators) or
   `__call__(self, batch)` (transforms).
1. **Curried functions → constructor args**: OSS `curry1`-decorated
   `make_pseudo_beta(prefix)` becomes `__init__(self, config, prefix="")`.
1. **Config replaces `common_cfg`/`mode_cfg`**: OSS
   `common_cfg.max_msa_clusters` → `self.config.max_msa_clusters` from
   `BaseConfig`.
1. **Return semantics differ by type**:
   - `FeatureGeneratorBase.__call__` → returns `feats = {}` with ONLY new keys
   - `FeatureCollatorBase.__call__` → modifies and returns `features` dict
   - `TransformBase.__call__` → modifies and returns `batch` dict
1. **`is_enabled()` for conditional steps**: If an OSS transform only runs when
   a config flag is True (e.g., `use_templates`), override `is_enabled()`.
   Always keep the spec in the list — never omit it.
1. **Ensembled transforms use `SampleRepeater`**: The recycling loop (OSS
   `map_fn` + `wrap_ensemble_fn`) is replaced by `SampleRepeater` wrapping
   collator specs.
1. **Random seed discipline**: Use `context["ensemble_seed"]` from `pre_init()`,
   not global random state. Pass through `torch.Generator`.

### Current BioIR Basic Schema Scope

The basic input schema (`bionemo_ir.data.schemas.basic`) now supports
these polymer/entity types. Port them when the target model and requested
pipeline need them:

**Schema-supported:**

- **Protein monomer** — `PolymerType.PROTEIN` with a single chain ID.
- **Protein multimer** — `PolymerType.PROTEIN` with multiple chain IDs or
  multiple protein polymers.
- **Protein MSAs** — `msas` and `paired_msas` on protein polymers.
- **RNA** — `PolymerType.RNA`, with `sequence` as a 1-letter nucleotide string.
- **DNA** — `PolymerType.DNA`, with `sequence` as a 1-letter nucleotide string.
- **CCD ligands / small molecules** — `PolymerType.CCD_LIGAND`, with `sequence`
  as one CCD code or an underscore-joined list of CCD codes, e.g. `"ATP"` or
  `"ATP_FAD"`. This maps to AF3/OF3 `ccdCodes` / `ccd_codes` via
  `sequence.split("_")`.
- **SMILES ligands / small molecules** — `PolymerType.SMILES_LIGAND`, with
  `sequence` as the SMILES string.
- **Protein structural templates** — `Template` / `TemplateParsed` on protein
  polymers (see `basic.py`, `Template` at ~L423). Each `Template` carries
  `path`/`content` (mmCIF), `format`, and `chain_id`. Templates are
  **protein-only** (schema validation forbids them on non-protein polymers,
  matching OSS, which only featurizes protein template chains). This is a
  **first-class, in-scope input** — port it when the target model consumes
  templates. See the
  [template featurization recipe](#template-featurization-protein-only) in Phase
  4 and the worked OpenFold3 port in
  `workdir/openfold3-port/template_equiv/implementation-notes.md` (direct-CIF
  path, L1 21/21 vs OSS on the NIM `data_with_template` set).

**Still not represented by the basic schema unless explicitly added elsewhere:**

- **Covalent bonds / bonded atom pairs** between entities.
- **Post-translational or nucleic-acid modifications** as structured per-residue
  metadata.
- **File-backed ligand inputs** such as SDF/MOL files.
- **User-provided CCD blocks or paths** as top-level input data.
- **Glycan-specific semantics** beyond representing a multi-component CCD ligand
  sequence.

Do not skip RNA/DNA or ligand/small-molecule paths just because older BioIR
ports focused on protein. First check whether the target model's OSS data
pipeline supports the entity type and whether the requested BioIR pipeline is
expected to cover it. If a schema-supported type is intentionally not ported for
a model-specific reason, record the limitation in
`$WORKDIR/implementation-notes.md` and `$WORKDIR/NOTES.md` with the exact
reason, affected samples, and what would be needed to add it later.

### Dependencies

**Use pre-existing packages from BioIR's `requirements.txt` first.** Do not
introduce new dependencies without justification. The project already includes
numpy, torch, pydantic, rdkit, and other common libraries — use them.

If the OSS code depends on a library not in `requirements.txt`:

1. Check if the same functionality can be achieved with an existing dependency
   (e.g., use `numpy` instead of `scipy` for simple linear algebra, use `torch`
   instead of a custom CUDA package).
1. If no existing dependency can substitute,
   **suggest the new dependency to the user** before adding it. Explain what
   it's needed for and whether it's a hard requirement or a nice-to-have.
1. Never silently add a new dependency.

### Code Standards

- **NVIDIA copyright header** (Apache 2.0) on every `.py` file.
- **Tensor device agnostic** — use `device=batch["key"].device`, never hardcode.
- **Feature keys are flat strings** — e.g., `"msa"`, `"template_aatype"`,
  `"extra_msa"`.
- **No global random state** — use `torch.Generator` seeded from `context`.
- **`is_enabled()` for conditional logic** — never remove a spec from the list;
  disable via `is_enabled()`.
- **`context` dict carries state** — seeds, parsed input, and intermediate
  non-tensor data flow through `context` between stages.

## Key Gotchas

- **No OSS imports.** Every function must be reimplemented. OSS code is
  read-only reference. This is the single most important rule — violating it
  creates hidden dependencies on OSS packages that may not be installed in
  production.
- **Return semantics matter.** Generators return NEW dicts with only new keys.
  Collators/transforms modify and return the SAME dict. Getting this wrong
  causes silent feature loss or key overwrites.
- **Seed alignment is fragile.** Stochastic collators (MSA sampling, masking)
  must use identical seed flow as OSS for equivalence testing. One extra
  `randperm` call shifts all downstream randomness.
- **Classification drives correctness.** Putting an ensembled function in
  generators (or vice versa) changes when it runs relative to the recycling
  loop. Misclassification produces wrong features silently.
- **`SampleRepeater` replaces the ensemble loop.** Do not implement your own
  recycling loop. Wrap collator specs in `SampleRepeater` with `get_n_iters`.
- **Order matches OSS exactly.** Generators and collators run in spec list
  order. Match the OSS `nonensembled_transform_fns()` and
  `ensembled_transform_fns()` execution order exactly.
- **Multimer is a separate variant.** If the model has monomer + multimer modes,
  implement both (separate tokenizer/factory classes or config-gated logic).
  Don't try to unify incompatible data flows.
- **Templates are a first-class protein input, but the no-template path is a
  trap.** When templates are present, port the real featurization (see
  [Template featurization](#template-featurization-protein-only) in Phase 4) —
  do not stub it. When they are absent, the no-template output must be
  **byte-identical to the OSS no-template stub** (all-zero masks + restype
  one-hot at the GAP class), NOT all-ones. Gate the real path on template
  presence; never emit `mask=1` everywhere for the empty path.
- **Pattern B context rows are not tensors.** In Pattern B pipelines,
  `context["_row"]` may contain non-tensor data (structures, molecule objects,
  parsed MSAs). Don't try to convert everything to tensors in the context
  generator — let feature generators handle the conversion.
- **`features_merger_func` overwrites on key collision.** Both
  `dict_context_merger` and `default_context_and_feature_merger` use
  `dict.update()` — if a generator produces a key that already exists in the
  context, the generator's value wins. Be careful not to accidentally shadow
  important context data.
- **Config must cover all accessed fields.** Every `self.config.X` reference in
  a generator/collator must have a corresponding field in the model's
  `BaseConfig` subclass. Missing fields produce `AttributeError` at runtime.
- **WORKDIR is for development, not production.** Debug scripts and e2e smoke
  tests in WORKDIR are scratch artifacts. Only equivalence tests and pipeline
  code get merged to the codebase.
- **OSS code may have bugs.** While reading the OSS pipeline as reference, you
  may discover bugs — incorrect indexing, off-by-one errors, wrong dtype
  conversions, missing edge cases, race conditions, or silent data corruption.
  If you find a suspected bug, **do NOT silently work around it**. Report it:
  1. Document the bug in `$WORKDIR/NOTES.md` under a `## OSS Bugs Found`
     section: file path, line number, what the bug is, what the correct behavior
     should be, and how it affects outputs.
  1. Alert the user immediately so they can decide whether to report upstream.
  1. Implement the **correct** behavior in BioIR, not the buggy behavior. Note
     in a code comment:

     ```python
     # NOTE: OSS bug at <file>:<line> — <description>.
     # We implement the correct behavior here.
     ```

  1. If the bug affects equivalence testing (BioIR produces different results
     than OSS because BioIR is correct), document this as an
     **expected divergence** in the test notes, not a test failure.
