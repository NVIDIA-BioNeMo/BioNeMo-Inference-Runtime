<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Model weights

Checkpoints are not committed here. A model resolves them in `__init__` when
`include_load_weights=True` (default for every family except Protenix), so a
miss fails at engine startup, not mid-inference.

Weights keep their **upstream** licences, and each upstream repo page is
authoritative. Boltz also needs chemical-component metadata — see
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
  `load_hf_weights` naming the key. See
  [AlphaFold2 parameters](#alphafold2-parameters).
- **`protenix-v2` has weights but no pipeline factory.** Construct
  `models/protenix` and call `load_weights()` yourself (`include_load_weights`
  defaults to `False`).

Downloads are **not revision-pinned** (`hf_hub_download` with no `revision=`).
Boltz-2 files are digest-checked after download — see [Downloads](#downloads).

## Resolution order

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

## Cache layout

Under `BIOIR_CACHE` (default `~/.cache/bionemo_ir`):

| Path                           | Purpose                                                         |
| ------------------------------ | --------------------------------------------------------------- |
| `checkpoints/<model key>/`     | staged checkpoints (`BIOIR_CHECKPOINTS` relocates the root)     |
| `metadata/<override env name>` | staged metadata (`BIOIR_METADATA` relocates)                    |
| `<model key>/`                 | Hub-downloaded metadata for that model (and extracted archives) |

Hub **checkpoint** downloads go to `~/.cache/hf` (hardcoded `cache_dir` in
`hubs/hf.py`) — `HF_HOME` / `HF_HUB_CACHE` do not move them.

## Staging and overrides

```bash
export BIOIR_CHECKPOINTS=/shared/bionemo-ir/checkpoints
mkdir -p "$BIOIR_CHECKPOINTS/boltz-2"
cp boltz2_conf.ckpt "$BIOIR_CHECKPOINTS/boltz-2/"
```

Model keys are the `FoldingSupportMatrix` strings — see
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

## What the loader does per family

Local and Hub paths use the same family handlers:

| Family                 | Handling                                                                 |
| ---------------------- | ------------------------------------------------------------------------ |
| Boltz-1/2              | `torch.load` under `safe_globals` (omegaconf), then `["state_dict"]`     |
| OpenFold3              | `torch.load` under `safe_globals`, then `["ema"]["params"]` when present |
| Protenix               | `torch.load`, then `["model"]`, strip a leading `module.` prefix         |
| OpenFold2 / AlphaFold2 | plain `torch.load(weights_only=True)`                                    |

After load, each family's `convert.py` remaps upstream names — see
[architecture][architecture].

## Using a checkpoint you modified

Point `<MODEL>_CKPT` at your file (or pass a state dict to
`load_weights(weights)` with `include_load_weights=False`). Layout must still
match upstream key names and shapes. Local files are **never** digest-checked —
verification is Hub-download only.

## Verifying resolution

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
into the snapshot; re-run to finish. See the [Hugging Face cache guide][hf-cache].

**Integrity.** `boltz2_conf.ckpt` and `boltz2_aff.ckpt` are checked against
`BOLTZ_CHECKPOINT_MD5` in `hubs/local.py` (mismatch → `ValueError`).
`boltz1_conf.ckpt` has no recorded digest. Checks run on **downloads only**,
not on env / staged files.

**Authentication.** BioIR does no token handling — that is all
`huggingface_hub`. OpenFold3 is gated: accept terms on
[`OpenFold/OpenFold3`][of3-hf], export `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN` /
`hf auth login`), then run the verify snippet with `openfold3`. Alternatively
stage the file and set `OPENFOLD3_CKPT` so resolution never hits the Hub. The
grant is per account — CI needs its own.

## Metadata assets

Boltz needs a CCD pickle and a molecule directory, resolved separately from the
checkpoint via `HF_MODEL_METADATA` in `hubs/metadata.py` (Boltz keys only;
others return `{}`):

| Key        | Asset      | Repo                      | File       | Override         |
| ---------- | ---------- | ------------------------- | ---------- | ---------------- |
| `ccd_path` | CCD pickle | `boltz-community/boltz-1` | `ccd.pkl`  | `BOLTZ_CCD_PATH` |
| `mol_dir`  | molecules  | `boltz-community/boltz-2` | `mols.tar` | `BOLTZ_MOL_DIR`  |

Identical for `boltz-1`, `boltz-2`, and `boltz-2-affinity` (CCD from Boltz-1,
`mols.tar` from Boltz-2 — including for Boltz-1). `mols.tar` is extracted after
download; extraction is skipped when the target already exists. Order: override
→ staged entry → download.

## AlphaFold2 parameters

BioIR distributes no AF2 weights. Converting DeepMind's published parameters
is manual:

1. Obtain `params_model_<N>[_ptm].npz` from the DeepMind
   [AlphaFold repository][af2], under the terms stated there.
2. Install upstream `openfold` — the converter imports `openfold.config`,
   `openfold.model.model`, and `openfold.utils.import_weights`. It is not a
   BioIR dependency, so use a throwaway environment.
3. Run `examples/folding/openfold2/jax_to_pt.py` with `--jax_path`,
   `--config_preset`, `--output_dir`. Preset ↔ parameter mapping is in the
   docstring table in `models/openfold2/config.py` — template and pTM variants
   are **not** interchangeable, so read it before choosing.
4. Set `ALPHAFOLD2_<N>_CKPT` or stage the result under the model key.

**Do not rename the `.npz` first.** The converter derives the upstream weight
version from the file name (`params_model_1_ptm.npz` → `model_1_ptm`) and
passes it to `import_jax_weights_`; the output is named after the same stem
(`params_model_1_ptm.pt`). A renamed input silently converts against the wrong
version.

The converter targets **monomer** presets only; there is no documented multimer
path for `alphafold2_multimer_*`.

## Related

- [Architecture][architecture] — where weight loading sits in the engine.
- [API reference][api] — constructing a model and calling `load_weights`.
- [Support matrix][support] — model keys and what each accepts.

<!-- link definitions -->

[af2]: https://github.com/google-deepmind/alphafold
[api]: api.md
[architecture]: architecture.md
[hf-cache]: https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
[of3-hf]: https://huggingface.co/OpenFold/OpenFold3
[support]: support-matrix.md
