---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# Model Weights

Checkpoints are not committed here. A model resolves them in `__init__` when
`include_load_weights=True` (default for every family except Protenix), so a
miss fails at engine startup, not mid-inference.

Weights keep their **upstream** licences, and each upstream repo page is
authoritative. Boltz also needs chemical-component metadata — refer to
[Metadata assets](#metadata-assets).

## Sources

`HF_CHECKPOINTS` in `hubs/hf.py` is the source of truth:

| Family     | Model keys                              | Upstream repo                | File                                         |
| ---------- | --------------------------------------- | ---------------------------- | -------------------------------------------- |
| Boltz-1    | `boltz-1`                               | `boltz-community/boltz-1`    | `boltz1_conf.ckpt`                           |
| Boltz-2    | `boltz-2`                               | `boltz-community/boltz-2`    | `boltz2_conf.ckpt`                           |
| Boltz-2    | `boltz-2-affinity`                      | `boltz-community/boltz-2`    | `boltz2_aff.ckpt`                            |
| OpenFold2  | every `openfold2_*` key                 | `nz/OpenFold`                | `finetuning_*.pt` (incl. `no_templ` / `ptm`) |
| OpenFold3  | `openfold3`                             | `OpenFold/OpenFold3`         | `checkpoints/of3-p2-155k.pt`                 |
| Protenix   | `protenix-v2`                           | `TMF001/protenix-v2-weights` | `protenix-v2.pt`                             |
| AlphaFold2 | `alphafold2_*`, `alphafold2_multimer_*` | none                         | local file only                              |

- **`nz/OpenFold` is a third-party mirror**, not `aqlaboratory/openfold`. Prefer
  upstream parameters + local resolve if provenance matters.
- **AlphaFold2 has no `HF_CHECKPOINTS` entry**, so those keys resolve locally
  only. With nothing staged, the Hub fallback raises `AssertionError` from
  `load_hf_weights` naming the key. Refer to
  [AlphaFold2 parameters](#alphafold2-parameters).
- **`protenix-v2` has weights but no pipeline factory.** Construct
  `models/protenix` and call `load_weights()` yourself (`include_load_weights`
  defaults to `False`).

Downloads are **not revision-pinned** (`hf_hub_download` with no `revision=`).
Boltz-2 files are digest-checked after download — refer to
[Downloads](#downloads).

## Staging With fetch_weights.sh

`scripts/fetch_weights.sh` downloads checkpoints from their upstream publishers
and symlinks them into the probe directory, so pytest and the examples resolve
them with no `*_CKPT` set. This is the normal way to get weights; the manual
route is [Staging and overrides](#staging-and-overrides).

```bash
scripts/fetch_weights.sh                    # everything the source covers
scripts/fetch_weights.sh --model boltz      # one family
scripts/fetch_weights.sh --model boltz-2    # one checkpoint
scripts/fetch_weights.sh --help
```

`--source` selects where they come from. The default `auto` uses NVIDIA's
internal NGC mirror when `BIOIR_NGC_ORG`, `BIOIR_NGC_TEAM`, and NGC credentials
are all present, and the public upstreams otherwise — so outside NVIDIA it is
exactly `--source public` and never needs credentials. `--source ngc` forces the
mirror and fails when it is unusable; `--source none` stages nothing and only
reports what already resolves.

What the public source covers, per family — the per-file list is the
`PUBLIC_URLS` table in the script, not restated here to avoid drift:

| Family     | Source                                          | Licence    |
| ---------- | ----------------------------------------------- | ---------- |
| OpenFold2  | `openfold.s3.amazonaws.com`                     | CC BY 4.0  |
| Boltz      | HuggingFace `boltz-community`, plus metadata    | MIT        |
| OpenFold3  | HuggingFace `OpenFold/OpenFold3` (**gated**)    | Apache-2.0 |
| Protenix   | HuggingFace mirror                              | Apache-2.0 |
| AlphaFold2 | **conversion required**, refer to the following | CC BY 4.0  |

Downloads are resumable and skipped when already present. Anything that cannot
be fetched is reported and skipped, never fatal — so a run with no credentials
completes, and the tests whose weights are missing skip themselves.

OpenFold3 is gated and needs `HF_TOKEN`; refer to
[Authentication](#downloads). AlphaFold2 is not downloadable at all — refer to
[AlphaFold2 parameters](#alphafold2-parameters), then point the script at your
converted files:

```bash
ALPHAFOLD2_DIR=/path/to/converted scripts/fetch_weights.sh --model alphafold2
```

## Resolution Order

`load_weights` (`hubs/checkpoint.py`) tries local, then Hugging Face. Local
order (`hubs/local.py`):

1. Explicit `cache_path` (Python-only; the pipeline never sets it).
2. The model's `<MODEL>_CKPT` env var, **if it names an existing file**. A bad
   path is warned and ignored — resolution continues (a typo can silently fall
   through to a download).
3. A staged file under `<checkpoint root>/<model key>/`: `*.pt` first, then
   `*.ckpt`; sorted-first match wins (keep one file per directory).

A miss returns `None` and Hub is tried. `hub="local"` / `hub="hf"` restrict to
one side; the pipeline uses the default (both). A key absent from
`LOCAL_CHECKPOINTS` raises `KeyError` up front.

## Cache Layout

Under `BIOIR_CACHE` (default `~/.cache/bionemo_ir`):

| Path                           | Purpose                                                         |
| ------------------------------ | --------------------------------------------------------------- |
| `checkpoints/<model key>/`     | staged checkpoints (`BIOIR_CHECKPOINTS` relocates the root)     |
| `metadata/<override env name>` | staged metadata (`BIOIR_METADATA` relocates)                    |
| `model_cache/`                 | raw downloads before staging (`MODEL_CACHE_DIR` relocates)      |
| `<model key>/`                 | Hub-downloaded metadata for that model (and extracted archives) |

Hub **checkpoint** downloads go to `~/.cache/hf` (hardcoded `cache_dir` in
`hubs/hf.py`) — `HF_HOME` / `HF_HUB_CACHE` do not move them.

## Staging and Overrides

```bash
export BIOIR_CHECKPOINTS=/shared/bionemo-ir/checkpoints
mkdir -p "$BIOIR_CHECKPOINTS/boltz-2"
cp boltz2_conf.ckpt "$BIOIR_CHECKPOINTS/boltz-2/"
```

Model keys are the `FoldingSupportMatrix` strings — refer to
[support matrix][support]. Extension matters; file name does not.

```bash
export BIOIR_METADATA=/shared/bionemo-ir/metadata
ln -s /shared/assets/ccd.pkl "$BIOIR_METADATA/BOLTZ_CCD_PATH"
ln -s /shared/assets/mols    "$BIOIR_METADATA/BOLTZ_MOL_DIR"
```

Each key has one `<MODEL>_CKPT` in `LOCAL_CHECKPOINTS` (`hubs/local.py`) —
**do not invent names from the key.** Uppercase + `-` → `_` + `_CKPT` works for
OpenFold2/3, AlphaFold2, and Protenix, but **not** Boltz: `boltz-1` →
`BOLTZ1_CKPT`, `boltz-2` → `BOLTZ2_CKPT`, `boltz-2-affinity` →
`BOLTZ2_AFFINITY_CKPT`.

```bash
BOLTZ2_CKPT=/data/boltz2_conf.ckpt python examples/folding/run_demo.py
```

Metadata overrides are `BOLTZ_CCD_PATH` and `BOLTZ_MOL_DIR`. Unlike checkpoint
vars they are **not** validated — a bad path fails later inside the model.

## What the Loader Does per Family

Local and Hub paths use the same family handlers:

| Family                 | Handling                                                                 |
| ---------------------- | ------------------------------------------------------------------------ |
| Boltz-1/2              | `torch.load` under `safe_globals` (omegaconf), then `["state_dict"]`     |
| OpenFold3              | `torch.load` under `safe_globals`, then `["ema"]["params"]` when present |
| Protenix               | `torch.load`, then `["model"]`, strip a leading `module.` prefix         |
| OpenFold2 / AlphaFold2 | plain `torch.load(weights_only=True)`                                    |

After load, each family's `convert.py` remaps upstream names — refer to
[architecture][architecture].

## Using a Checkpoint You Modified

Point `<MODEL>_CKPT` at your file (or pass a state dict to
`load_weights(weights)` with `include_load_weights=False`). Layout must still
match upstream key names and shapes. Local files are **never** digest-checked —
verification is Hub-download only.

## Verifying Resolution

```bash
python -c "
from bionemo_ir.hubs import load_metadata, load_weights
print(len(load_weights('boltz-2')), 'entries resolved')
print(load_metadata('boltz-2'))
"
```

Logs which branch ran at `INFO` (`BIOIR_LOG_LEVEL`). For an offline
pre-flight, pass `local_files_only=True` (not exposed by the pipeline).

## Downloads

**Interrupted.** Partial Hub transfers stay as `.incomplete` and are not linked
into the snapshot; re-run to finish. Refer to the
[Hugging Face cache guide][hf-cache].

**Integrity.** `boltz2_conf.ckpt` and `boltz2_aff.ckpt` are checked against
`BOLTZ_CHECKPOINT_MD5` in `hubs/local.py` (mismatch → `ValueError`).
`boltz1_conf.ckpt` has no recorded digest. Checks run on **downloads only**,
not on env / staged files.

**Authentication.** BioNeMo Inference Runtime (BioIR) does no token handling —
that is all `huggingface_hub`. OpenFold3 is gated and an anonymous request gets
`401`. Registering is free: create a HuggingFace account, accept the terms on
[`OpenFold/OpenFold3`][of3-hf], then export `HF_TOKEN` (or
`HUGGING_FACE_HUB_TOKEN` / `hf auth login`).

```bash
export HF_TOKEN=hf_...
```

`fetch_weights.sh` passes it to HuggingFace downloads and the library's own Hub
fallback picks it up too. Without it the OpenFold3 tests skip with
`GatedRepoError` in the skip reason; nothing else in the suite needs it.
Alternatively stage the file and set `OPENFOLD3_CKPT` so resolution never hits
the Hub. The grant is per account — CI needs its own.

## Metadata Assets

Boltz needs a CCD pickle and a molecule directory, resolved separately from the
checkpoint through `HF_MODEL_METADATA` in `hubs/metadata.py` (Boltz keys only;
others return `{}`):

| Key        | Asset      | Repo                      | File       | Override         |
| ---------- | ---------- | ------------------------- | ---------- | ---------------- |
| `ccd_path` | CCD pickle | `boltz-community/boltz-1` | `ccd.pkl`  | `BOLTZ_CCD_PATH` |
| `mol_dir`  | molecules  | `boltz-community/boltz-2` | `mols.tar` | `BOLTZ_MOL_DIR`  |

Identical for `boltz-1`, `boltz-2`, and `boltz-2-affinity` (CCD from Boltz-1,
`mols.tar` from Boltz-2 — including for Boltz-1). `mols.tar` is extracted after
download; extraction is skipped when the target already exists. Order: override
→ staged entry → download.

## AlphaFold2 Parameters

BioIR distributes no AF2 weights. Converting DeepMind's published parameters
is manual:

1. Obtain `params_model_<N>[_ptm].npz` from the DeepMind
   [AlphaFold repository][af2], under the terms stated there.
2. Install upstream `openfold` — the converter imports `openfold.config`,
   `openfold.model.model`, and `openfold.utils.import_weights`. It is not a
   BioIR dependency, so use a throwaway environment.
3. Run `examples/folding/openfold2/jax_to_pt.py` with `--jax_path`,
   `--config_preset`, `--output_dir`. The preset ↔ parameter mapping for the
   **monomer** presets is the table in `models/openfold2/config.py` — template
   and pTM variants are **not** interchangeable, so read it before choosing.
   Multimer presets are not in that table; they follow the
   `model_<N>_multimer_v3` form.
4. Set `ALPHAFOLD2_<N>_CKPT` or stage the result under the model key — or hand
   the output directory to `fetch_weights.sh` through `ALPHAFOLD2_DIR`, which
   recognises the converter's `params_model_*.pt` names.

End to end, for one preset:

```bash
curl -O https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xf alphafold_params_2022-12-06.tar -C params/
# In a throwaway env with upstream `openfold` installed. Repeat for
# params_model_{1..5}.npz and params_model_{1..5}_multimer_v3.npz, passing the
# matching --config_preset (model_N / model_N_multimer_v3):
python examples/folding/openfold2/jax_to_pt.py \
    --jax_path params/params_model_1.npz --config_preset model_1 \
    --output_dir converted/

ALPHAFOLD2_DIR=converted scripts/fetch_weights.sh --model alphafold2
```

That tarball is pinned to `2022-12-06` because that release is **CC BY 4.0**;
the earlier ones were non-commercial. Review the licence before using these
weights for anything.

**Do not rename the `.npz` first.** The converter derives the upstream weight
version from the file name (`params_model_1_ptm.npz` → `model_1_ptm`) and
passes it to `import_jax_weights_`; the output is named after the same stem
(`params_model_1_ptm.pt`). A renamed input silently converts against the wrong
version.

Multimer presets convert the same way: `--config_preset model_<N>_multimer_v3`
against `params_model_<N>_multimer_v3.npz`.

## Related

- [Architecture][architecture] — where weight loading sits in the engine.
- [API reference][api] — constructing a model and calling `load_weights`.
- [Support matrix][support] — model keys and what each accepts.

[af2]: https://github.com/google-deepmind/alphafold
[api]: api.md
[architecture]: architecture.md
[hf-cache]: https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
[of3-hf]: https://huggingface.co/OpenFold/OpenFold3
[support]: support-matrix.md
