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

# Path B — no BioIR data pipeline

Use this when `docs/ref/support-matrix.md` says Pipeline = **No**
**and** the model is still a folding compute path. Today that is
`protenix-v2` (no factory). `boltz-2-affinity` is the same *shape*
(module only) but a different task — do not run it with this
folding dump / CIF / lDDT path. See SKILL.md
[Later: other model kinds](SKILL.md#later-other-model-kinds).

`build_processor` cannot run. Both timed forwards consume **the same
OSS-featurized batch**. The BioIR side is the `nn.Module` constructor
API (`docs/ref/api.md`), not the processor.

```text
the spec item (A3Ms / templates)
        │  MSAs + templates attached (same contract)
        ▼
OSS data pipeline  ──untimed──►  feature dict
        │
        ├─► OSS model.forward(batch)      timed   (oss_python)
        └─► adapt keys ──► BioIR model.forward    timed   (bioir_python)
```

One feature dict for both forwards means template parity is free on
Path B: whatever the OSS pipeline built, both sides see. Verify the
template features are non-empty once, on that shared batch.

Do not invent a BioIR tokenizer / feature factory for the bench.

## Construct the BioIR module

Protenix is not registered. `include_load_weights` defaults to
`False` — pass `True` (`docs/ref/model-weights.md`).

```python
from bionemo_ir.configs import AcceleratedConfig
from bionemo_ir.models.protenix import Protenix

# Default constructor: config=None → get_pretrained_config.
# Do not pass a handmade BaseConfig.
model = Protenix(model_name="protenix-v2", include_load_weights=True)
model = model.cuda().eval()
# Select the diffusion module without overriding its safe graph routine.
model.optimize({
    "diffusion_module": AcceleratedConfig(backend="torch"),
})
```

Omit `default=`. Protenix's module-declared routine uses exact-shape
keys, accepts at most 1024 tokens, and falls back to eager above that
limit. Passing an explicit graph-optimization config replaces the
routine and removes its guard.

`forward` takes Boltz-style `runtime_args`: `recycling_steps`,
`num_sampling_steps`, `diffusion_samples` (`docs/ref/api.md`).
For this bench, Protenix-v2 uses `recycling_steps=5`,
`num_sampling_steps=200`, and `diffusion_samples=5`. Map those to OSS
`model.N_cycle=6`, `sample_diffusion.N_step=200`, and
`sample_diffusion.N_sample=5`; see
[models/protenix.md](models/protenix.md#runtime-lock-and-recycle-semantics).

## Featurize once, time twice

Two venvs cannot import OSS and BioIR in one process. Dump features
from `oss_python`, load them in both harnesses.

```text
$WORKDIR/ref_data/oss_features/<sample_id>.pt
```

Each file is the GPU-ready host dict (CPU tensors) plus the resolved
MSA paths used to build it. Write a provenance line
(`source=oss`, OSS commit, command). A dump that came from BioIR is
invalid.

Reuse the OSS inference script's featurizer (same as Path A).
Do not write a second feature builder from OSS internals.

```python
# oss_python — untimed, via the OSS infer/predict helpers
batch = oss_featurize(oss_input)          # must have seen the A3Ms
torch.save({"batch": cpu(batch), "msas": msa_paths}, path)

# oss_python — eager first (required)
batch = to_device(torch.load(path)["batch"])
time_model_forward(oss_model, batch, oss_runtime_args)

# oss_python — compile target submodules ONCE on a fresh model,
# then reuse oss_compiled for every in-scope dump. Do not compile
# per sample. Do not torch.compile(oss_compiled) as a whole.
compile_oss_hot_modules(oss_compiled, family="af3")
for sample_id, path in in_scope_dumps:
    batch = to_device(torch.load(path)["batch"])
    row = run_oss_timed_forward(oss_compiled, batch, oss_runtime_args)
    # first warmup (and first of a new token_bin) may compile;
    # not every sample.

# bioir_python — timed
raw = torch.load(path)["batch"]
batch = to_device(adapt_oss_batch_to_bioir(raw))
time_model_forward(bioir_model, batch, bioir_runtime_args)
```

Use the same `time_model_forward` helper as
[measurement.md](measurement.md). There is no
`profile_inference` field on this path.

## Feature adapter

OSS key names may not match `Protenix.forward` (`restype`, `msa`,
`ref_pos`, `atom_to_token_idx`, …). Write
`$WORKDIR/bench/adapt_features.py` that maps OSS → BioIR keys.

- Record every rename / reshape / dtype cast in
  `implementation-notes.md`.
- Do not recompute MSA or structure features in the adapter. It is a
  rename/layout step only.
- If a required BioIR key is missing from the OSS dict, stop. Do not
  fill zeros to keep the bench green.
- After adapt, assert MSA-derived tensors are still non-empty.

## Inputs

Still the spec items with MSAs
([samples.md](samples.md)). Convert each in-scope item to the OSS
query / CIF / A3M (or Boltz YAML+CSV) layout the OSS pipeline
expects. Rewrite stale Boltz `msa:` absolutes to
`$DATASET_ROOT/casp15/msa/`. The MSA contract still applies — the
OSS featurizer must receive every listed A3M / CSV. Do not call
`load_requests` + `build_processor`. Do not use
`examples/data/samples/`.

## Isolation

`bioir_python` needs `bionemo_ir` and the dumped `.pt` files only.
It must not import the OSS package. `oss_python` featurizes and times
the OSS module; it must not import `bionemo_ir`.
