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

# Protenix-v2 benchmark profile

Use this profile with [`../SKILL.md`](../SKILL.md). Protenix-v2 is a
folding model with a BioIR compute module but no BioIR data-pipeline
factory, so it always uses **Path B** from
[`../no-pipeline.md`](../no-pipeline.md):

```text
dataset spec -> pinned OSS Protenix featurizer -> one CPU feature dump
                                                   |-> OSS forward
                                                   `-> BioIR forward
```

Do not call `build_processor`, do not build a BioIR feature pipeline, and
do not featurize a second time for BioIR.

## Source and checkpoint lock

- OSS source: repository gitlink `3rdparty/protenix`.
- Required pin: ByteDance Protenix tag `v2.0.0`, commit
  `2475421477ab414b571149ad4a875c390ff8a35d`.
- Model name on both sides: `protenix-v2`.
- Checkpoint registry: `TMF001/protenix-v2-weights/protenix-v2.pt`.
- BioIR construction:
  `Protenix(model_name="protenix-v2", include_load_weights=True)`.
  Omit `config=` so the optimized pretrained config is selected.
- Set `PROTENIX_V2_CKPT` when a local checkpoint is staged. Record the
  resolved path and SHA-256 and strict-load the same file in both
  interpreters before feature dumping or timing.
- Default scratch work directory for this profile: `/tmp/protenix`.

The checkpoint is wrapped as `{"model": state_dict}` and its keys may have
a `module.` prefix. Pinned OSS strips the prefix in
`InferenceRunner.load_checkpoint`; BioIR's hub loader does the equivalent
before its strict conversion. Never use `strict=False`.

## Runtime lock and recycle semantics

BioIR's public argument is a **recycle count**:

```python
num_cycles = recycling_steps + 1
```

Pinned OSS Protenix's `model.N_cycle` is the number of trunk loop
iterations. Therefore map:

```text
OSS model.N_cycle = BioIR recycling_steps + 1
```

The skill baseline (`recycling_steps=5`) consequently means OSS
`model.N_cycle=6`.

For this benchmark, lock:

```json
{
  "bioir": {
    "recycling_steps": 5,
    "num_sampling_steps": 200,
    "diffusion_samples": 5
  },
  "oss": {
    "model.N_cycle": 6,
    "sample_diffusion.N_step": 200,
    "sample_diffusion.N_sample": 5
  }
}
```

This is five recycles / six total trunk cycles on both sides. Setting OSS
`N_cycle=5` would run one fewer cycle and is not parity.

Also lock seed 101, one model seed, `use_msa=true`,
`use_rna_msa=false`, `msa_pair_as_unpair=true`, `use_template=true`,
one discarded warmup, and one measured forward. The release declares no
RNA A3Ms; do not enable RNA-MSA search or invent one. There is no Amber
relaxation stage in this path.

## OSS inference path and effective config

Base the harness on `3rdparty/protenix/runner/inference.py`:

- Build config from `configs_base`, `data_configs`, `inference_configs`,
  and `model_configs["protenix-v2"]`, then apply the locked command-line
  values.
- Use `get_inference_dataloader()` for featurization and `DataDumper` for
  CIF/confidence output.
- For each OSS forward, preserve the native call boundary from
  `InferenceRunner.predict`: move the already-featurized batch to CUDA
  before timing, enter the resolved BF16 autocast context, and call the
  top-level model. Do not time the dataloader, H2D, or `DataDumper`.
- Call `update_inference_configs(configs, N_token)` before each OSS
  sample so the pin's dynamic chunking policy is applied. Serialize the
  expanded thresholds and the resulting per-sample chunk values.
- Use the recommended OSS kernels and inference switches:
  `triangle_attention=cuequivariance`,
  `triangle_multiplicative=cuequivariance`, `dtype=bf16`,
  `enable_tf32=true`, `enable_efficient_fusion=true`, and
  `enable_diffusion_shared_vars_cache=true`.
- Installing cuEquivariance is not enough: the two triangle config
  fields above must resolve to `cuequivariance` in the locked config.
- Install the matching `cuequivariance-torch` frontend as well as the
  base and CUDA-operator wheels. On SM90, pinned Protenix reports that
  cuEquivariance triangle attention is unavailable and uses its native
  reference implementation; record the requested and effective kernels
  separately. Triangle multiplicative update still uses cuEquivariance.
- On a CUDA-13 container use the CUDA-13 cuEquivariance artifacts; do
  not install the upstream requirements file's `*-cu12` wheel.
- Keep `LAYERNORM_TYPE=fast_layernorm` only when its pinned extension
  builds and probes successfully in the isolated OSS interpreter.
  A deliberate `LAYERNORM_TYPE=torch` compatibility column must be
  named and approved as a fallback; do not silently relabel it as the
  recommended OSS path.

With torch `2.12.0a0+5aff3928d8.nv26.05`, the pinned fast LayerNorm
extension builds but fails the real forward with `Cannot access data
pointer of Tensor that doesn't have storage`. The compatible smoke/full
column is therefore named `oss_*_torch_layernorm`, uses
`LAYERNORM_TYPE=torch`, and preserves every other OSS setting. Keep the
failed native probe in the benchmark notes; do not publish it as a
latency row.

The upstream `requirements.txt` pins torch 2.7.1 and
`cuequivariance-ops-torch-cu12==0.8.0`. Keep OSS package changes out of
the BioIR interpreter. Record the actual CUDA-13 remap and all version
differences in `bench_config.json`.

## Convert `spec_full.json` to Protenix input

All 17 items in `$DATASET_ROOT/spec_full.json` are in scope.
Convert each item to the AlphaFold-Server-like list accepted by
`runner/inference.py`:

- `protein` -> `proteinChain`
- `rna` -> `rnaSequence`
- `dna` -> `dnaSequence`
- `ccd_ligand` -> `ligand` with `CCD_<code>`
- preserve explicit chain ids as an `id` list and set `count` to its
  length
- attach each protein's resolved `unpairedMsaPath` and
  `pairedMsaPath`; attach an RNA `unpairedMsaPath` only when the spec
  declares one

The spec sometimes stores one chain id as a string and sometimes as a
one-element list. Normalize both to a list before computing `count`.
Write one generated query JSON per sample plus an aggregate index.
Symlink every A3M from the release into `oss_data/msa/`; never copy or
edit the release.

Pinned Protenix's inference MSA loader takes one unpaired and one paired
A3M path per protein entity. Assert:

- every path declared by the spec is present in the generated query;
- every staged symlink resolves to the recorded release path;
- `N_msa > 1` for declared-MSA samples unless the source A3M genuinely
  contains only the query row;
- `msa`, `has_deletion`, `deletion_value`, `profile`, and
  `deletion_mean` are present and finite;
- paired multichain inputs retain distinct per-chain entries.

## Caller-supplied templates

`T1152`, `T1118v1`, and `T1158v1` carry caller-supplied template
**structures**, while pinned Protenix's JSON accepts a template-search
alignment (`templatesPath`) and then resolves an mmCIF by the hit name.
Stage this deterministic adapter:

1. Symlink every supplied CIF into
   `oss_data/templates/mmcif/<lowercase-entry>.cif`.
1. Select the requested `chain_id`; when it is null, choose the
   best-aligning protein chain in that supplied CIF and record the
   identity/alignment used for the choice.
1. Generate one `hmmsearch.a3m`-compatible hit per supplied template.
   Its description must satisfy pinned
   `HmmsearchA3MParser` (`<entry>_<chain>/start-end`,
   `mol:protein`, and `length:<n>`), and the query entity receives that
   file as `templatesPath`.
1. Use the real Kalign binary when Protenix realigns query and template.
   Do not synthesize coordinates or a dummy template tensor.
1. Scope the input-side compatibility patch to this explicit caller
   list: bypass search-oriented release-date, duplicate, coverage, and
   keep/drop filters; set the processing cutoff far enough in the
   future to accept the supplied CIF. Do not alter template feature
   math, cap, or model compute.

The profile's cap is four, matching pinned inference. Verify attachment
per protein chain from `template_pseudo_beta_mask`: count rows with any
populated token in that chain and require
`min(n_supplied_for_chain, 4)`. A global nonzero mask is insufficient,
especially for `T1152`, which supplies the same structure separately
to two protein entities.

Record every generated alignment, chosen chain, symlink target, and
filter override in `oss_data/index.json` and
`implementation-notes.md`. The resulting template tensors are part of
the shared dump, so BioIR and OSS consume the same template evidence.

## OSS data-pipeline dependencies

Importing `protenix.data` is what the feature dump costs, and the
dependency set is larger than the overlay installs. On a dev box the
base interpreter happens to carry the rest, so this only surfaces in a
clean container running from a wheel, where `run_bioir.py` dies on
`ModuleNotFoundError` before it reaches a single sample.

Both chains — `protenix.data.inference.infer_dataloader` plus
`protenix.model.protenix` for the dump, and `runner.inference` plus
`runner.dumper` for the OSS column — reach the same third-party set,
so there is no cheaper subset to install for one column only.

Three are hard imports and must be installed:

- `ml_collections` — `protenix.config.config`, which `make_oss_config`
  needs.
- `scikit-learn` — `KDTree` in `protenix/data/core/featurizer.py`, real
  featurization work that cannot be stubbed.
- `optree` — module scope in `protenix/model/utils.py`.

Install them into **both** interpreters. `run_bioir.py` dumps features
under `BIOIR_PYTHON` and `run_oss.py` builds the model in the overlay
venv, and a `--user` install in the base is invisible inside a venv, so
neither one covers the other. Every version floor is low enough that
pip leaves BioIR's pinned `numpy` and `scipy` alone.

Three more appear in a static scan but need nothing:

- `lmdb` — `protenix/utils/file_io.py` imports it under
  `if TYPE_CHECKING`, plus one function-local use.
- `orjson` — function-local inside `try/except ImportError`, with a
  `json` fallback.
- `deepspeed` — function-local in `protenix/model/triangular/layers.py`,
  reached only by `DS4Sci_EvoformerAttention`, which this profile does
  not select.

### Skip fair-esm

Do not install `fair-esm`. Nothing in this profile computes ESM
embeddings:

- The `protenix-v2` block in `configs_model_type.py` declares no `esm`
  key, so it inherits `configs_base.py`'s
  `{"enable": False, "model_name": "esm2-3b", "embedding_dim": 2560}`.
- `infer_dataloader.py` reads `esm_info.get("enable", False)` and puts
  every `ESMFeaturizer` construction and call behind that flag.
- The only contact with the package is a module-scope
  `from esm import FastaBatchedDataset, pretrained` in
  `protenix/data/esm/compute_esm.py`, which `protenix/data/esm/__init__.py`
  imports, which `infer_dataloader.py` pulls in for `ESMFeaturizer`.

So the requirement is module surface, not functionality. `common.py`
registers a stub `esm` module at import time — before any entry point
adds the OSS tree to `sys.path` — and its symbols raise when used, so a
config that turns ESM on gets an explicit error rather than features
computed without embeddings.

**The stub must raise `AttributeError` for `_`-prefixed and dunder
names.** `inspect.getmodule` walks `sys.modules` and probes `__file__`
on every entry, so a stub that answers dunders breaks unrelated work:
observed as a `torch.library.custom_op` registration inside `import
torch` failing with `protenix used esm.__file__`. Only real ESM symbols
get a refusing placeholder.

Importing this chain the first time also JIT-builds Protenix's
`fast_layer_norm_cuda_v2` extension with `nvcc`, which took about two
and a half minutes. That pause is the extension build, not a hang.

## Feature dump and BioIR adapter

Run `bench/dump_features.py` only with the OSS interpreter. For
each sample save a trusted local `.pt` containing:

- the CPU `input_feature_dict`;
- the OSS `AtomArray` and `entity_poly_type` needed by `DataDumper`;
- `N_token`, `N_atom`, `N_msa`, chain metadata, and provenance;
- resolved unpaired/paired MSA paths and row counts;
- per-chain populated template counts;
- the OSS commit, query digest, and effective featurizer config.

Call pinned OSS `update_input_feature_dict()` before saving so
`d_lm`, `v_lm`, and `pad_info` are present in the shared artifact.
Pinned OSS recomputes those deterministic layout tensors at the start
of its top-level forward; BioIR uses the dumped copies. Record and
verify bit equality for one sample.

`torch.load()` defaults to `weights_only=True` on recent PyTorch.
These trusted local dumps include an `AtomArray`, so both harnesses
must pass `weights_only=False`.

BioIR adaptation is layout-only:

- add a leading `B=1` dimension to every tensor except `pad_info`;
- leave non-tensor metadata unchanged;
- move the adapted dict to CUDA outside the timing window;
- do not recompute MSA, template, chemistry, or structure features;
- reject a missing required key instead of filling zeros.

Construct BioIR with the default pretrained config and select the
parent `diffusion_module` through `model.optimize()` using
`AcceleratedConfig(backend="torch")` with no `default=`. This preserves
the module-declared exact-shape CUDA-graph routine and its inclusive
1024-token acceptance limit; larger inputs intentionally fall back to
eager. An explicit graph-optimization config replaces that safe
routine, so do not supply one or raise the limit. Do not also graph the
nested token transformer. Pass `compact_output=True`, preserve
`full_data` for the writer, and use the three locked runtime arguments
above.

The tracker rejects an input above 1024 tokens before allocating a graph
state or adding a capture-fallback key. Audit such a zero-state call as
`eager_out_of_range` when the configured acceptance limit rejects its
`N_token`; only treat zero states as unclassified for an accepted input.

For large pair representations, lock and serialize the benchmark's
semantic-preserving chunk policy: pair transition, diffusion pair
transition, and pair-weighted averaging at 256; outer-product mean at
64; contact probability at 256; each with `min_size=1024`. Apply the
same values for smoke and full BioIR runs. These chunks avoid
materializing the corresponding full pair intermediates and do not
change OSS features, model weights, or requested sampling work.

## OSS compile status — skipped

Do not run or publish a `torch.compile` column for pinned Protenix-v2.
OSS eager with `LAYERNORM_TYPE=torch` is the required reference.

The faithful combined `pairformer_stack` + `diffusion_module` probe
passed synthetic A, A, B, B stability and every five-sample measured
forward captured zero new frames. It was nevertheless unusable:

- probe plus smoke startup/compilation took about 81 minutes on H100;
- warmup-only shape adaptations continued through the 1,734-token row;
- mean lDDT collapsed from `0.7108` eager to `0.1308` compiled;
- mean DockQ over three applicable rows collapsed from `0.7709` to
  `0.2310`.

Those are downstream-fitness failures, not publishable latency results.
An isolated Pairformer retry was stopped because compilation remained
too slow. Do not continue the target-isolation or retry ladder unless
the user explicitly asks to reprobe after a Protenix, PyTorch, or
cuEquivariance pin changes.

Keep one transferable fixture lesson: a future direct probe must call
`pairformer_stack` under CUDA BF16 autocast and `diffusion_module` with
autocast disabled (`skip_amp.sample_diffusion=true`). Calling
Pairformer outside its production autocast context creates an
FP32-query/BF16-value SDPA mismatch before the compiler is tested.

## Output and scoring

Use pinned OSS `DataDumper` for both outputs. Build its input from:

- OSS: measured `prediction` from the top-level model;
- BioIR: `coordinate`, `summary_confidence`, and `full_data` from the
  measured compact output;
- shared dump: `AtomArray` and polymer entity metadata.

Score the rank-zero CIF after timing. Run OpenStructure lDDT on every
sample with ground truth. Run DockQ on supported protein-protein and
protein-small-molecule interfaces; use `--small_molecule` for ligand
complexes and record `unsupported_chain_types` when DockQ cannot score
an RNA/DNA interface. Single-chain structures use
`dockq=null, dockq_status="single_chain"`.

## Five-sample smoke selection

For the 17-row release manifest, the percentile rule initially selects
indices 0, 4, 8, 12, and 16 after sorting by residue count. Replace the
middle interior row with the template-bearing row nearest the median.
The locked smoke set is:

- `R1117` — smallest; RNA-ligand; no declared MSA
- `R1136` — lower quartile; RNA-ligand
- `T1118v1` — template-bearing replacement nearest the median
- `8a8o-assembly1_A_B` — upper quartile; paired protein MSA
- `8ic7-assembly1_A_B` — largest

Store ranks, residue counts, and the replacement reason in
`bench_config.json`. Run the full locked 200-step / five-sample
diffusion settings in smoke; do not shorten them.
