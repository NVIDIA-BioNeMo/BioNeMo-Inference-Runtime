# TensorRT-BioNeMo — Module Onboarding Guide (Claude Code skill)

This document is **for partners onboarding a new model architecture** to TRT-BioNeMo — i.e. you have your own Pairformer / DiffusionTransformer / EvoformerStack-style module in PyTorch and you want to swap it for TRT-BioNeMo's optimized implementation (or the TensorRT engine equivalent), preserving forward-pass numerics.

> **Audience.** Most early-access customers running the bundled scripts ([`scripts/run_pipeline.py`](scripts/run_pipeline.py), [`notebooks/openfold2_full_pipeline.ipynb`](notebooks/openfold2_full_pipeline.ipynb)) **do not need this document** — the supported AlphaFold2 / OpenFold2 / Boltz / OpenFold3 models are already wired in. This guide is only relevant if you're integrating a custom module that isn't in the supported list.
>
> **Prerequisites:** working install of [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) (the CLI) on your developer machine. Without Claude Code installed you can still read [`skills/module-onboard/SKILL.md`](skills/module-onboard/SKILL.md) as a regular markdown playbook and follow it manually, but the slash-command UX described below depends on the CLI.

---

## What this is

[`skills/module-onboard/SKILL.md`](skills/module-onboard/SKILL.md) is an **Agent Skill** ([Agent Skills](https://agentskills.io/) standard): a multi-step playbook for onboarding a customer module to TRT-BioNeMo. It walks the user (you, or Claude on your behalf) through:

1. **Survey** — inspect the customer module's `__init__` / `forward` / config to extract hyperparameters and signature.
2. **Weight conversion** — map the customer's `state_dict` to the TRT-BioNeMo layout (QKV / KV fusion, AdaLN fusion, gate+input fusion, name renames).
3. **Adapter wiring** — write a `nn.Module` adapter that bridges the customer's forward signature to the TRT-BioNeMo module's signature.
4. **In-place swap** — write a helper that replaces the original module on a live model instance.
5. **Hierarchical equivalence tests** — block-level then stack-level numerical comparison vs the customer's reference implementation.
6. **Benchmarks** — measure throughput / memory deltas against the customer's baseline.

In Claude Code, each skill is loaded from **`SKILL.md` inside a dedicated subdirectory**; the YAML frontmatter `name` field becomes the slash command. This skill sets `name: modules-onboard`, so the slash command is **`/modules-onboard`**.

---

## Bundle layout

```text
skills/module-onboard/
├── SKILL.md                                 ← playbook (rename to SKILL.md when installing — see below)
└── samples/
    ├── RF3_Pairformer/                      ← worked example: BakerLab RF3 Pairformer → TRT-BioNeMo
    │   ├── integration/
    │   │   ├── __init__.py
    │   │   ├── adapter.py                   ← forward-signature adapter
    │   │   ├── config.py                    ← PairformerConfig factory
    │   │   └── swap.py                      ← in-place module replacement helper
    │   └── convert/
    │       ├── __init__.py
    │       └── convert_weights.py           ← RF3 PairformerBlock → PairformerLayerV1 weight remap
    └── RF3_DiT/                             ← worked example: BakerLab RF3 DiffusionTransformer → TRT-BioNeMo
        ├── integration/
        │   ├── __init__.py
        │   ├── adapter.py                   ← forward-signature adapter (flattens diffusion-sample dim)
        │   ├── config.py                    ← DiffusionTransformerConfig factory
        │   └── swap.py                      ← in-place module replacement helper
        └── convert/
            ├── __init__.py
            └── convert_weights.py           ← RF3 DiffusionTransformerBlock → DiffusionTransformerLayer weight remap
```

---

## Bundled samples — RF3 module conversions

Two end-to-end reference conversions ship under [`skills/module-onboard/samples/`](skills/module-onboard/samples). Each sample is a pair of minimal packages you can copy into your project and adapt:

| Package | Files | Role |
|---|---|---|
| `integration/` | `config.py`, `adapter.py`, `swap.py` | TRT-BioNeMo `BaseConfig` factory, forward-signature adapter, and in-place module-swap helper. |
| `convert/` | `convert_weights.py` | Weight remapping from the customer's `state_dict` to the TRT-BioNeMo layout (QKV/KV fusion, AdaLN fusion, gate+input fusion, name renames). |

### Sample 1 — RF3 Pairformer

**Integration:** [`config.py`](skills/module-onboard/samples/RF3_Pairformer/integration/config.py), [`adapter.py`](skills/module-onboard/samples/RF3_Pairformer/integration/adapter.py), [`swap.py`](skills/module-onboard/samples/RF3_Pairformer/integration/swap.py).
**Conversion:** [`convert/convert_weights.py`](skills/module-onboard/samples/RF3_Pairformer/convert/convert_weights.py).

- Replaces BakerLab RF3's `Recycler.pairformer_stack` (a list of `PairformerBlock`s) with `tensorrt_bionemo._torch.layers.transformers.pairformer.PairformerModule`.
- **Signature bridge** (`BakerLabPairformerAdapter`): customer calls `forward(S_I, Z_II, is_padding_I)` with a bool padding mask; TRT-BioNeMo expects `forward(s, z, mask, pair_mask)` with `float` valid-masks. Adapter flips the mask polarity (`~is_padding_I`) and builds the outer-product `pair_mask`.
- **Config** (`make_bakerlab_pairformer_config`): maps RF3's `c_s=384, c_z=128, num_heads=16, num_blocks=48` onto `PairformerConfig`, defaults to `triangle_attention_backend="CuTeDSL"` / `pairwise_attention_backend="CuTeDSL"` (the optimized PyTorch backend recommended in [`ADVANCED.md`](ADVANCED.md#0-choose-a-backend-pytorch-vs-tensorrt) §0).
- **Swap** (`swap_pairformer_stack`): builds the TRT-BioNeMo module, optionally runs `convert_pairformer_stack_weights` (from the sibling `convert/` package) on the customer's `state_dict`, and assigns the adapter back to `recycler.pairformer_stack`.
- **Weight remap** (`convert_pairformer_stack_weights` in `convert/convert_weights.py`): handles `tri_mul_outgoing → tri_mul_out` renames, QKV fusion for triangle attention, KV fusion and zero-bias initialization for `AttentionPairBias`, and gate+input fusion for transitions.

### Sample 2 — RF3 DiffusionTransformer (DiT)

**Integration:** [`config.py`](skills/module-onboard/samples/RF3_DiT/integration/config.py), [`adapter.py`](skills/module-onboard/samples/RF3_DiT/integration/adapter.py), [`swap.py`](skills/module-onboard/samples/RF3_DiT/integration/swap.py).
**Conversion:** [`convert/convert_weights.py`](skills/module-onboard/samples/RF3_DiT/convert/convert_weights.py).

- Replaces BakerLab RF3's `DiffusionModule.diffusion_transformer` (a stack of `DiffusionTransformerBlock`s) with an `nn.ModuleList` of `tensorrt_bionemo._torch.layers.transformers.diffusion_transformer.DiffusionTransformerLayer`.
- **Signature bridge** (`BakerLabDiTBlockAdapter`): customer passes a diffusion-sample dim `D` — `A_I[B, D, I, C]`, `S_I[B, D, I, C]`, `Z_II[B, I, I, Cz]`. The adapter flattens `D` into the batch axis (`B*D`), broadcasts the pair tensor and mask across `D`, runs the TRT-BioNeMo layer, and reshapes back to `[B, D, I, C]`.
- **Config** (`make_bakerlab_dit_config`): maps `c_token=384, c_s=384, c_tokenpair=128, num_heads=16, num_blocks=24` onto `DiffusionTransformerConfig` with the BakerLab-specific toggles (`bias_proj=True`, `conditioned_transition_using_silu=True`, `attn_output_gate=False`, no pre/post layer-norm).
- **Swap** (`swap_diffusion_transformer`): builds one `DiffusionTransformerLayer` per RF3 block, loads per-block weights via `convert_dit_block_weights` (sibling `convert/` package), wraps the stack in `BakerLabDiTStackAdapter`, and reassigns `diffusion_module.diffusion_transformer`.
- **Weight remap** (`convert/convert_weights.py`): targets the real-checkpoint layout (`c_token=768`, `c_s=384`, `c_z=128`, 24 blocks, AdaLN named `ada_ln_1` / `ada_ln`, QK-norm present, attention output gate present). Handles AdaLN gain/bias fusion, `Sequential(to_g)` unwrapping, QKV fusion, and the output gate projection.

> **`convert/` is bundled with each sample** — you do **not** need to generate it from scratch. Both `swap.py` files resolve `convert.convert_weights` from the sibling `convert/` directory. If you want to wire the swap before feeding a real `state_dict`, leave `customer_state_dict=None` and the TRT-BioNeMo module will use default (random) initialization — useful for shape/interface smoke tests. When adapting to a new RF3 release, edit the bundled `convert/convert_weights.py` rather than rewriting it.

---

## Reading the playbook without Claude Code

If Claude Code is not installed (or not yet allowed on your machine), [`skills/module-onboard/SKILL.md`](skills/module-onboard/SKILL.md) is still a normal markdown file you can read top-to-bottom. The two `samples/RF3_*/` packages above are runnable Python — you can copy them into your project, point the `swap_*` helper at your model instance, and run the equivalence tests manually. The Claude Code integration below is purely a UX accelerator that automates the survey + scaffolding steps.

---

## Installing the skill into Claude Code

Claude Code auto-discovers skills only when they use the expected layout (see [Extend Claude with skills](https://docs.claude.com/en/docs/claude-code/SKILL.md)):

| Scope | Suggested path |
|---|---|
| **Project skill** (this repo only) | `<project>/.claude/skills/modules-onboard/SKILL.md` (+ `samples/`) |
| **Personal skill** (all projects on this machine) | `~/.claude/skills/modules-onboard/SKILL.md` (+ `samples/`) |

Additionally, Claude Code expects the playbook file to be named **`SKILL.md`** (singular) inside a directory that matches the frontmatter `name`.  If you use this TensorRT-BioNeMo distribution root as your project workspace, then the Claude skills will be organized according to the **Project skill** template above.

If you use a different project workspace, copy the whole `skills/module-onboard/` tree so the **bundled samples travel with the playbook** and the skill's sample references resolve:

```bash
mkdir -p .claude/skills/modules-onboard
cp -r skills/module-onboard/samples .claude/skills/modules-onboard/samples
cp    skills/module-onboard/SKILL.md .claude/skills/modules-onboard/SKILL.md
# or, with symlinks (keeps a single source of truth):
# ln -sf "$(pwd)/skills/module-onboard/samples"   .claude/skills/modules-onboard/samples
# ln -sf "$(pwd)/skills/module-onboard/SKILL.md" .claude/skills/modules-onboard/SKILL.md
```

Then start a Claude Code CLI session in the project directory (or any workspace that contains `.claude/skills/`). Updates usually apply live; if you created the entire `.claude/skills/` tree for the first time after the session started, you may need to **restart** Claude Code.

---

## Invoking the skill in the CLI

- **Manual:** type **`/modules-onboard`** (matches frontmatter `name`) and follow the instructions. 
    - You can add a module file path or working directory as extra context if you want to anchor the session — e.g. `/modules-onboard ~/code/rf3/src/rf3/model/pairformer.py` or `/modules-onboard ~/code/rf3/src/rf3/model/diffusion_transformer.py` to reproduce the two bundled samples.  
    - If you intend to run Claude within a docker container, you may need to bind mount the source code project containing the module you intend to convert.
- **Automatic:** Claude may load the skill when the request matches the frontmatter `description` (e.g. TRT-BioNeMo module onboarding, weight conversion, numerical validation).

### Additional directories (`--add-dir`)

With `--add-dir <path>`, the primary goal is filesystem access; **skills** are still loaded only from `.claude/skills/` inside the added path (a documented exception). If you add a parent folder that contains this repo, place a copy at `<path>/.claude/skills/modules-onboard/SKILL.md` (plus `samples/`) or use a personal skill under `~/.claude/skills/`.

---

## See also

- Skill layout, frontmatter, `disable-model-invocation`, supporting files: [Claude Code — Skills](https://docs.claude.com/en/docs/claude-code/SKILL.md)
- Full playbook: [`skills/module-onboard/SKILL.md`](skills/module-onboard/SKILL.md)
- Reference conversions:
    - RF3 Pairformer — [integration](skills/module-onboard/samples/RF3_Pairformer/integration) · [convert](skills/module-onboard/samples/RF3_Pairformer/convert)
    - RF3 DiT — [integration](skills/module-onboard/samples/RF3_DiT/integration) · [convert](skills/module-onboard/samples/RF3_DiT/convert)
- Backend selection (PyTorch vs TensorRT) used by the bundled samples: [`ADVANCED.md` §0](ADVANCED.md#0-choose-a-backend-pytorch-vs-tensorrt)
- First-time setup of TRT-BioNeMo (image, container, wheel, sample run): [`README.md`](README.md)
