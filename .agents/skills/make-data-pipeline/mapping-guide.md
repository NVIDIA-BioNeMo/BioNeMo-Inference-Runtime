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
{}
---

# OSS → BioIR Function Mapping Guide

This document shows how OSS data pipeline functions map to BioIR classes. It
covers:

- **Mapping strategy** — how to classify any OSS function into its BioIR
  target
- **Concrete example (Pattern A)** — OpenFold2 (flat tensor dict,
  `data_transforms.py`-style)
- **Common conversion patterns** — reusable code patterns for any model
- **Checklist** — steps to follow for a new model

## Mapping Strategy

### 1. Identify the OSS Pipeline Stages

Typical OSS code has:

- `data_pipeline.py` — raw input → numpy feature dict (parsing, MSA
  construction, template featurization)
- `feature_pipeline.py` — numpy → tensor conversion, calls `input_pipeline`
- `input_pipeline.py` — defines `nonensembled_transform_fns()` and
  `ensembled_transform_fns()`
- `data_transforms.py` — individual transform functions (free functions, often
  curried)
- `data_transforms_multimer.py` — multimer-specific transforms

### 2. Classify Each OSS Function

| Classification                      | Criteria                                                             | BioIR Target                                      |
| ----------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------- |
| **Raw feature builder**             | Creates numpy arrays from sequences/MSAs/templates                   | `ContextGeneratorBase` in `feature_context.py`    |
| **Non-ensembled, modifies dict**    | Runs once, modifies existing keys (cast, reorder, squeeze)           | `TransformBase` in `transforms.py`                |
| **Non-ensembled, creates new keys** | Runs once, adds new feature keys (masks, profiles, atom14)           | `FeatureGeneratorBase` in `feature_generators.py` |
| **Ensembled, stochastic**           | Runs per recycling iter, involves randomness (sample, mask, crop)    | `FeatureCollatorBase` in `feature_collators.py`   |
| **Ensembled, deterministic**        | Runs per recycling iter, deterministic (cluster, make_msa_feat, pad) | `FeatureCollatorBase` in `feature_collators.py`   |
| **Output processing**               | Converts model output → structured output                            | `PostProcessorBase` in `postprocessor.py`         |

### 3. OpenFold2 Concrete Mapping

#### Non-ensembled transforms → `transforms.py` (TransformBase)

| OSS Function                                      | BioIR Class                     | Notes                                                |
| ------------------------------------------------- | ------------------------------- | ---------------------------------------------------- |
| `cast_to_64bit_ints(protein)`                     | `CastTo64BitInts`               | Direct 1:1                                           |
| `correct_msa_restypes(protein)`                   | `CorrectMsaRestypes`            | Direct 1:1                                           |
| `squeeze_features(protein)`                       | `SqueezeFeatures`               | Direct 1:1                                           |
| `randomly_replace_msa_with_unknown(0.0)(protein)` | `RandomlyReplaceMsaWithUnknown` | Curried arg → `__init__` param `replace_proportion`  |
| `fix_templates_aatype(protein)`                   | `FixTemplatesAatype`            | Uses `is_enabled()` tied to `config.enable_template` |

#### Non-ensembled generators → `feature_generators.py` (FeatureGeneratorBase)

| OSS Function                                     | BioIR Class              | Notes                                               |
| ------------------------------------------------ | ------------------------ | --------------------------------------------------- |
| `make_seq_mask(protein)`                         | `MakeSequenceMask`       | Returns new `feats = {"seq_mask": ...}`             |
| `make_msa_mask(protein)`                         | `MakeMsaMask`            | Returns new `feats` with `msa_mask`, `msa_row_mask` |
| `make_hhblits_profile(protein)`                  | `MakeHhblitsProfile`     | Returns `feats = {"hhblits_profile": ...}`          |
| `make_template_mask(protein)`                    | `MakeTemplateMask`       | `is_enabled()` → `config.enable_template`           |
| `make_pseudo_beta("template_")(protein)`         | `MakeTemplatePseudoBeta` | Curried prefix → `is_enabled()` check               |
| `atom37_to_torsion_angles("template_")(protein)` | `Atom37ToTorsionAngles`  | `__init__` takes `prefix` param                     |
| `make_atom14_masks(protein)`                     | `MakeAtom14Masks`        | Complex, uses residue constants                     |
| (inference only) use_clamped_fape                | `UseClampedFape`         | Sets `use_clamped_fape` to zeros                    |

#### Ensembled transforms → `feature_collators.py` (FeatureCollatorBase)

| OSS Function                            | BioIR Class               | Notes                                                            |
| --------------------------------------- | ------------------------- | ---------------------------------------------------------------- |
| `sample_msa(max_seq, keep_extra, seed)` | `SampleMsa`               | Reads `config.max_msa_clusters`, uses `context["ensemble_seed"]` |
| `make_masked_msa(cfg, frac, seed)`      | `MakeMaskedMsa`           | `__init__` takes `profile_prob`, `same_prob`, etc.               |
| `nearest_neighbor_clusters()`           | `NearestNeighborClusters` | `is_enabled()` → `config.msa_cluster_features`                   |
| `summarize_clusters()`                  | `SummarizeClusters`       | `is_enabled()` → `config.msa_cluster_features`                   |
| `crop_extra_msa(max_extra)`             | `CropExtraMsa`            | `is_enabled()` → `config.max_extra_msa > 0`                      |
| `delete_extra_msa`                      | `DeleteExtraMsa`          | `is_enabled()` → `config.max_extra_msa is None`                  |
| `make_msa_feat()`                       | `MakeMsaFeat`             | Concatenates features                                            |
| `select_feat(feat_list)`                | `SelectFeat`              | `__init__` takes `include_feats` list                            |
| `random_crop_to_size(...)`              | `RandomCropToSize`        | Inference: only template cropping                                |
| `make_fixed_size(...)`                  | `MakeFixedSize`           | Pads MSA/template dims to fixed sizes                            |

#### Multimer-specific transforms

| OSS Function                                 | BioIR Class                       | Notes                                                     |
| -------------------------------------------- | --------------------------------- | --------------------------------------------------------- |
| `make_msa_profile(protein)`                  | `MultimerMakeMsaProfile`          | Generator, uses masked mean                               |
| `create_target_feat(protein)`                | `MultimerCreateTargetFeatures`    | Generator, one-hot aatype                                 |
| `sample_msa (multimer)`                      | `MultimerSampleMsa`               | Gumbel-based sampling                                     |
| `make_masked_msa (multimer)`                 | `MultimerMakeMaskedMsa`           | Uses `gumbel_max_sample`                                  |
| `nearest_neighbor_clusters (multimer)`       | `MultimerNearestNeighborClusters` | Soft assignment via softmax                               |
| `create_msa_feat (multimer)`                 | `MultimerCreateMsaFeat`           | Different feature concatenation                           |
| `feature_processing_multimer.pair_and_merge` | `MultimerFeaturePairAndMerge`     | In `feature_context.py`, called during context generation |

#### Raw feature construction → `feature_context.py` (ContextGeneratorBase)

| OSS Component                | BioIR Method                                         | Notes                                                                                                                                                                                                                                                                                             |
| ---------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `make_sequence_features()`   | `FeatureContextGenerator.make_sequence_features()`   | Returns aatype, residue_index, seq_length                                                                                                                                                                                                                                                         |
| `make_msa_features()`        | `FeatureContextGenerator.make_msa_features()`        | Returns msa, deletion_matrix, num_alignments                                                                                                                                                                                                                                                      |
| `make_template_features()`   | `FeatureContextGenerator.empty_template_feats()`     | Placeholder in this OF2 mapping. A full impl reads real templates — see the "Template featurization (protein-only)" recipe in `SKILL.md` (Phase 4) and the worked OF3 direct-CIF port. The empty path must be byte-identical to the OSS no-template stub (all-zero masks + restype at GAP class). |
| `np_to_tensor_dict()`        | `FeatureContextGenerator.np_to_tensor_dict()`        | Numpy → torch conversion with feature filtering                                                                                                                                                                                                                                                   |
| MSA pairing + chain merging  | `MultimerFeaturePairAndMerge.__call__()`             | Combines per-chain features for multimer                                                                                                                                                                                                                                                          |
| `add_assembly_features()`    | `FeatureContextGenerator.add_assembly_features()`    | Adds asym_id, sym_id, entity_id                                                                                                                                                                                                                                                                   |
| `convert_monomer_features()` | `FeatureContextGenerator.convert_monomer_features()` | Reshapes monomer features for multimer                                                                                                                                                                                                                                                            |

## Common Conversion Patterns

### Pattern A: Free function → Class

**OSS:**

```python
def make_seq_mask(protein):
    protein["seq_mask"] = torch.ones(protein["aatype"].shape, dtype=torch.float32)
    return protein
```

**BioIR:**

```python
class MakeSequenceMask(FeatureGeneratorBase):
    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        feats = {}
        feats["seq_mask"] = torch.ones(batch["aatype"].shape, dtype=torch.float32)
        return feats
```

### Pattern B: Curried function → Constructor args

**OSS:**

```python
@curry1
def sample_msa(protein, max_seq, keep_extra=True, seed=None):
    ...
# Usage: sample_msa(max_seq=512, keep_extra=True, seed=seed)
```

**BioIR:**

```python
class SampleMsa(FeatureCollatorBase):
    def __init__(self, config=None, keep_extra=True):
        super().__init__(config)
        self.keep_extra = keep_extra

    def __call__(self, features, context):
        max_seq = self.config.max_msa_clusters  # From config, not arg
        seed = context.get("ensemble_seed", None)  # From context, not arg
        ...
```

### Pattern C: Conditional execution → is_enabled()

**OSS:**

```python
if common_cfg.use_templates:
    transforms.append(data_transforms.make_template_mask)
```

**BioIR:**

```python
class MakeTemplateMask(FeatureGeneratorBase):
    def is_enabled(self) -> bool:
        return self.config.enable_template
    # Always in the spec list, but skipped if not enabled
```

### Pattern D: Ensembled loop → SampleRepeater

**OSS:**

```python
def map_fn(fun, x):
    ensembles = [fun(elem) for elem in x]
    return {feat: torch.stack([d[feat] for d in ensembles], dim=-1)
            for feat in ensembles[0]}

tensors = map_fn(lambda x: wrap_ensemble_fn(tensors, x),
                 torch.arange(num_recycling + 1))
```

**BioIR:**

```python
FeatureCollatorSpec(
    name="repeater",
    functor=SampleRepeater,
    kwargs={
        "feature_collator_specs": [  # The ensembled collators
            FeatureCollatorSpec(name="sample_msa", functor=SampleMsa),
            FeatureCollatorSpec(name="make_masked_msa", functor=MakeMaskedMsa),
            ...
        ],
        "get_n_iters": lambda config: config.max_recycling_iters + 1
    }
)
```

### Pattern E: Seed management

**OSS:**

```python
ensemble_seed = random.randint(0, torch.iinfo(torch.int32).max)
# Passed explicitly to each function
sample_msa(max_seq, seed=msa_seed)
make_masked_msa(cfg, frac, seed=(msa_seed + 1) if msa_seed else None)
```

**BioIR:**

```python
def pre_init(context):
    random_seed = context.get("random_seed", 0)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)
    context["ensemble_seed"] = random.randint(0, torch.iinfo(torch.int32).max)
    return context

# Each collator reads from context:
class SampleMsa(FeatureCollatorBase):
    def __call__(self, features, context):
        seed = context.get("ensemble_seed", None) if not self.config.resample_msa_in_recycling else None
        g = torch.Generator(device=features["msa"].device)
        g.manual_seed(seed)
        shuffled = torch.randperm(num_seq - 1, generator=g) + 1
```

## Pattern B Mapping (Boltz2-style)

When the OSS pipeline uses multi-step featurization with intermediate non-tensor
state, the mapping differs:

| OSS Pattern                                                                                                                                   | BioIR Target                                                                                                                              | Notes                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Structure building from CCD/molecules                                                                                                         | `ContextGeneratorBase` in `feature_context.py`                                                                                            | Returns a row dict with structure objects, tokens, etc.   |
| Per-step featurization (token features, atom features, MSA features)                                                                          | Separate `FeatureGeneratorBase` classes in `feature_generators.py`                                                                        | Each reads `context["_row"]` and produces new tensor keys |
| Final feature assembly                                                                                                                        | `FeatureCollatorBase` in `feature_collators.py`                                                                                           | Assembles/selects the final tensor set                    |
| Tokenization logic (sequence → token IDs)                                                                                                     | `tokenizer_logic.py` (helper) + `ContextGenerator`                                                                                        | Complex tokenization may warrant its own module           |
| Per-residue / per-ligand cheminformatics features (computed from a Mol object — distance bounds, chirality, stereochemistry, planarity, etc.) | Compute in the per-residue parser, accumulate at the structure level with global atom offsets, materialise tensors in a feature generator | See "Cheminformatics feature mapping" below               |

**Key difference from Pattern A:** In Pattern B, the context generator does NOT
produce a flat tensor dict. Instead it produces a mixed dict (the "row"), and
the feature generators are responsible for converting non-tensor data into
tensors. The `_row` key in `context` bridges the tokenizer stage to the feature
stage.

### Cheminformatics feature mapping

OSS pipelines that process small molecules or non-standard residues frequently
compute features off a chem-toolkit Mol object (typically RDKit). The set of
features varies per model — discover the actual list from the Phase 1 function
inventory — but the *mapping shape* is uniform.

**Generic conversion recipe:**

1. **Compute per-residue (or per-Mol) in the parser that already owns the Mol.**
   Return the constraint/feature lists in a single dict so the call signature
   stays stable as you add more kinds later.
1. **Accumulate at the structure level** with the chain's `global_atom_idx`
   offset applied to every `atom_idxs` field. Multi-component ligands and SMILES
   ligands need this — each component's local RDKit indices must be shifted to
   global atom indices.
1. **Thread through the context row** (e.g.
   `row["residue_constraints"] = {...}`) so the feature stage can build tensors
   without re-parsing mols.
1. **Materialise tensors** in the corresponding feature generator via a small
   helper that takes a list of dicts and an expected arity:

   ```python
   def _stack_idx(items, arity):
       if not items:
           return torch.empty((arity, 0), dtype=torch.long)
       rows = np.asarray([list(c["atom_idxs"]) for c in items], dtype=np.int64).T
       return torch.from_numpy(rows).long()
   ```

1. Pair index tensors with parallel bool/float arrays (e.g. `is_reference` /
   `is_r` / `is_e` flags, upper/lower bounds, type masks) built from the
   **same items list** so the ordering between an index tensor and its flag
   arrays is tied together by construction.

**Order discipline:** index tensors are order-sensitive. Iterate atoms / bonds /
matches in the same order the OSS computation does (usually `mol.GetAtoms()` /
`mol.GetBonds()` order, with the same skip-Hs rule). If you traverse
differently, indices line up only by accident.

**Pitfalls that produce silently-empty feature tensors:**

- Mol-mutating operations (e.g. `RemoveHs(sanitize=False)`, atom removal, SMILES
  round-tripping) can strip computed atom/bond properties (`_CIPRank`,
  hybridization, ring info) that the constraint computations read. Re-run the
  relevant perception (`AssignStereochemistry`, `SanitizeMol`,
  `AssignCIPLabels`) on the post-mutation Mol and assert the property is
  present.
- Toolkit-loaded Mols (pickle, CIF, vendor SDK) often arrive without aromaticity
  / ring perception initialised — SMARTS matchers then return empty. Decide
  intentionally vs OSS; do not call perception in only one of the two pipelines.
- Some toolkit APIs depend on prior prep (e.g. ring info initialisation before
  bounds-matrix calls). Replicate the OSS prep sequence exactly.

**Equivalence-test discipline:** include at least one input where each
cheminformatics feature kind *must* be non-empty (a SMILES with an explicit
stereo tag, an aromatic ring, a known E/Z bond, a known constrained distance).
Pure-CCD inputs can pass an "empty == empty" comparison that hides bugs on both
sides.

## Checklist for New Model Mapping

- \[ \] List ALL functions from OSS data pipeline code
- \[ \] Classify each as Transform / Generator / Collator / ContextGenerator /
  PostProcessor
- \[ \] Identify which are conditional (need `is_enabled`)
- \[ \] Map curried args to `__init__` params or `self.config` fields
- \[ \] Map config references (`common_cfg.X` → `self.config.X`)
- \[ \] Identify ensemble boundary (non-ensembled vs ensembled)
- \[ \] Map random seed flow
- \[ \] Identify feature key lists for SelectFeat and MakeFixedSize
- \[ \] Choose pipeline pattern (A or B) based on intermediate data types
- \[ \] Map multimer-specific transforms separately (if applicable)
- \[ \] Map raw feature construction to ContextGenerator
- \[ \] Map postprocessing (model output → structured output)
