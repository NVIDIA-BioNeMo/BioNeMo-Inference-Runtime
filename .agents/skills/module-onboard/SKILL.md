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
name: module-onboard
description: Converts a source model's fundamental module (Pairformer, DiffusionTransformer, etc.) to use BioIR optimized layers, with weight conversion and numerical validation.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Module Onboarding — Converting Source Modules to BioIR

- **Input:** Source module (an `nn.Module` subclass) + checkpoint.
- **Output:** Weight conversion function + adapter wrapper + hierarchical
  equivalence tests.

## Reference Samples (RF3)

Complete worked examples live under `samples/`, built against [RoseTTAFold3
(RF3)][rf3] — an open-source, BSD-3-Clause all-atom structure prediction model.
Its module layouts are public, so the samples double as a reference for the
naming and fusion patterns a conversion has to handle:

```plaintext/none
samples/
├── RF3_Pairformer/
│   ├── convert/convert_weights.py   # PairformerBlock -> PairformerLayerV1 weight mapping
│   └── integration/                 # adapter.py, config.py, swap.py
└── RF3_DiT/
    ├── convert/convert_weights.py   # DiffusionTransformerBlock -> DiffusionTransformerLayer
    └── integration/                 # adapter.py, config.py, swap.py
```

Use these as templates when the source module matches the RF3 layout
(Pairformer or DiffusionTransformer).

[rf3]: https://github.com/RosettaCommons/foundry/tree/production/models/rf3

**Before starting, confirm the following with the user** (ask if not provided):

1. **Path to the source module file** — the `.py` file containing the
   `nn.Module` subclass to convert.
1. **Class name** — if the file contains multiple modules, which class is the
   target.
1. **Path to checkpoint** (optional) — `.pt`, `.safetensors`, or checkpoint
   directory. If none, random weights will be used.
1. **Working directory** — where to place all generated artifacts (default:
   `/workspace/onboard_<source>_<module>/`).

## Working Directory Isolation (CRITICAL)

**All generated artifacts MUST live in a dedicated working directory outside the
BioIR codebase.** Do NOT create, modify, or place any files inside the BioIR
install/source tree (`bionemo_ir/`, `tests/`, `examples/`, etc.) or the
source repository. Both the BioIR repo and the source repository are
read-only dependencies.

At the start of the onboarding, create a working directory and use it for
everything:

```plaintext/none
<workdir>/                          # e.g., /workspace/onboard_<source>_<module>/
├── convert/                        # Weight conversion scripts + converted checkpoints
│   ├── convert_weights.py          # Conversion function (Phase 2)
│   ├── <module>_ckpt/              # Optional serialized checkpoint (safetensors)
│   └── ...
├── integration/                    # Adapter wrapper + module swap + config (Phase 3)
│   ├── adapter.py
│   ├── swap.py
│   └── config.py
├── tests/                          # All equivalence tests (Phase 4)
│   ├── test_subcomponent.py
│   ├── test_layer.py
│   └── test_full_module.py
├── benchmarks/                     # Benchmark scripts (Phase 5)
│   └── bench_<module>.py
└── results/                        # Benchmark results (auto-saved by benchmark scripts)
    ├── bench_<timestamp>.csv       # Tabular results for spreadsheets
    └── bench_<timestamp>.json      # Full metadata (GPU, config, per-row data, peak memory)
```

**Rules:**

- Import from `bionemo_ir` as installed package — never modify its source.
- Import from the source repository as needed — never modify it either.
- All converted weights, test scripts, adapter code, and benchmark results go
  under `<workdir>/`.
- The working directory path should be confirmed with the user in Phase 0 before
  any files are created.
- Add `<workdir>` to `PYTHONPATH` if needed so that test scripts can import the
  conversion and adapter modules.

  ```bash
  export PYTHONPATH=<workdir>:${PYTHONPATH}
  ```

## Phase 0 — Gather Resources & Feasibility Check

Do ALL file reads and analysis upfront before proceeding.

### Step 0 — GPU & environment sanity check

1. Run `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits` to
   get available VRAM.
1. Estimate module memory footprint (parameter count x bytes-per-dtype). If it
   exceeds VRAM, **stop and report to the user**.
1. Verify BioIR is installed:
   `python -c "import bionemo_ir; print(bionemo_ir.__file__)"`.

### Step 1 — Create working directory

Confirm the working directory path with the user, then create the directory
structure:

```bash
export WORKDIR=/workspace/onboard_<source>_<module>
mkdir -p $WORKDIR/{convert,integration,tests,benchmarks}
```

All subsequent phases write files exclusively under `$WORKDIR`. Never write into
the BioIR source tree or the source repository.

### Step 2 — Locate the source module & install dependencies

Read the source model's module file. Extract:

- Class name and inheritance chain
- All `__init__` sub-modules (name, class, constructor args)
- `forward()` signature and data-flow graph (which sub-module is called in what
  order, what residual connections exist)
- Training-only features (dropout, activation checkpointing, auxiliary losses)

**Install only the packages needed to import and run the source module.** Do
NOT install the source model's full package or all of its dependencies — only
install the minimal set required to instantiate the module and run its
`forward()`. This avoids dependency conflicts with BioIR.

1. **Identify required imports** — read the source module and trace its
   import chain. List only the packages the module actually imports (e.g.,
   `einops`, `ml_collections`, a custom ops package). Ignore packages used only
   by training, data loading, or CLI code that the onboarding does not touch.
1. **Install missing packages** — for each required import that is not already
   installed:

   ```bash
   pip install <package>
   ```

   If the source model is a proper package and the module cannot be imported
   without it being installed (e.g., relative imports), add the source
   root to `sys.path` instead of installing:

   ```bash
   export PYTHONPATH=<source_root>:$PYTHONPATH
   ```

1. **Verify the import works** — confirm the source module can be instantiated:

   ```bash
   python -c "from <source_package>.<module_path> import <ClassName>; print('OK')"
   ```

If any required package conflicts with BioIR's dependencies (e.g., different
PyTorch version), **stop and report the conflict to the user**. Do not
force-install conflicting packages.

### Step 3 — Locate source checkpoint (or use random weights)

Check whether the user has placed a checkpoint file in `$WORKDIR` (e.g.,
`$WORKDIR/source_checkpoint.pt`, `$WORKDIR/*.safetensors`).

**If a checkpoint is provided:**

- Identify the format (`.pt`, `.safetensors`, state_dict structure).
- List all parameter keys for the module being converted.
- If the module appears N times (e.g., stacked layers), confirm the naming
  pattern (e.g., `layers.{i}.tri_mul_out.weight`).

**If no checkpoint is provided:**

- Proceed with **randomly initialized weights**. Instantiate the source module
  and the BioIR module with matching hyperparameters, then use their
  default-initialized `state_dict()` for conversion and testing.
- This is sufficient for validating the conversion pipeline (weight mapping,
  shape correctness, forward-pass equivalence). Numerical outputs won't be
  meaningful, but structural correctness is fully testable.
- Note in the summary report (Phase 6) that random weights were used and
  real-checkpoint validation is still pending.

## Phase 1 — Survey BioIR Coverage & Architecture Analysis

BioIR accelerates source modules via the **optimized PyTorch backend**
(`bionemo_ir/_torch/`). Survey it to determine which sub-modules have
matching implementations.

### Preamble — Investigating BioIR when only the wheel is available

When the BioIR git repository is not present (e.g., the user installed from a
wheel), you cannot use Read/Glob/Grep on source files directly. Use these
techniques instead — they work on any installed Python package because wheels
include `.py` source files:

**1. Find the install location:**

```bash
python -c "import bionemo_ir, os; print(os.path.dirname(bionemo_ir.__file__))"
```

**2. List all source files under a subpackage:**

```bash
python -c "
import bionemo_ir, os
root = os.path.dirname(bionemo_ir.__file__)
for dirpath, _, files in os.walk(root):
    for f in files:
        if f.endswith('.py'):
            print(os.path.join(dirpath, f).replace(root+'/', ''))
" | grep -E '_torch/layers|configs/'
```

**3. Read a module's full source via `inspect`:**

```python
import inspect
import bionemo_ir._torch.layers.transformers.diffusion_transformer as m
print(inspect.getsource(m))                          # full file
print(inspect.getsource(m.OpenFold3DiffusionTransformer))   # one class
```

**4. Discover available classes/functions in a module:**

```python
import bionemo_ir._torch.layers.transformers.pairformer as m
import inspect
print([n for n, o in inspect.getmembers(m, inspect.isclass) if o.__module__ == m.__name__])
```

**5. Inspect constructor signatures:**

```python
import inspect
from bionemo_ir._torch.layers.transformers.diffusion_transformer import DiffusionTransformerLayer
print(inspect.signature(DiffusionTransformerLayer.__init__))
```

**6. Inspect config fields (Pydantic models):**

```python
from bionemo_ir.configs.modules import DiffusionTransformerConfig
for name, field in DiffusionTransformerConfig.model_fields.items():
    print(f"{name}: default={field.default}, type={field.annotation}")
```

**7. List all config classes:**

```python
import inspect, bionemo_ir.configs.modules as m
from pydantic import BaseModel
print([n for n, o in inspect.getmembers(m, inspect.isclass) if issubclass(o, BaseModel)])
```

**8. Find weight conversion helpers in installed models:**

```bash
python -c "
import bionemo_ir.models.boltz1.convert as m, inspect
print([n for n, _ in inspect.getmembers(m, inspect.isfunction)])
"
```

Use `inspect.getsource()` freely throughout Steps 1–3 to read any class or
function exactly as you would read a file — the source is embedded in the
installed `.py` files. When `getsource` fails (C extensions, `.so` files), fall
back to `help()` or `__doc__`.

### Step 0 — Verify installed packages

Before proceeding, verify that all packages required across the onboarding
pipeline are installed. A missing package mid-process wastes time and can
corrupt partial results. Run the checks below and
**stop if any critical package is missing** — report the gap to the user with
the install command.

```bash
python -c "
import sys, importlib

checks = {
    # ── Core (required for all phases) ──
    'torch':              'PyTorch (core dependency)',
    'bionemo_ir':   'BioIR (the framework being onboarded to)',

    # ── Torch backend (Phase 1–4) ──
    'triton':             'Triton (fused Triton kernels in _torch/)',
    'cuequivariance':     'cuEquivariance (CUEQUIV attention backend)',
    'cutlass.cute':       'CUTLASS DSL (CuTeDSL attention backend)',

    # ── Weight conversion & serialization (Phase 2) ──
    'safetensors':        'safetensors (optional checkpoint serialization)',

    # ── Testing (Phase 4) ──
    'pytest':             'pytest (test runner)',

    # ── Source model (Phase 0) ──
    # Source-specific imports are checked separately in Phase 0.
}

missing = []
for mod, desc in checks.items():
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append((mod, desc))

if missing:
    print('MISSING PACKAGES:')
    for mod, desc in missing:
        print(f'  - {mod}: {desc}')
    sys.exit(1)
else:
    print('All required packages are installed.')
"
```

Also verify GPU architecture for backend-specific kernels:

```bash
python -c "
import torch
sm = torch.cuda.get_device_capability()
print(f'GPU: {torch.cuda.get_device_name()} (SM{sm[0]}{sm[1]})')
print(f'  CUEQUIV attention:  SM70+ — {\"YES\" if sm >= (7,0) else \"NO\"}'  )
print(f'  CuTeDSL attention:  SM80+ — {\"YES\" if sm >= (8,0) else \"NO\"}'  )
print(f'  Triton kernels:     SM80+ — {\"YES\" if sm >= (8,0) else \"NO\"}'  )
print(f'  CUTLASS DSL (SM90): SM90+ — {\"YES\" if sm >= (9,0) else \"NO\"}'  )
"
```

Record the results. If any critical package is missing:

| Package                    | Install command                                                    | Required for                      |
| -------------------------- | ------------------------------------------------------------------ | --------------------------------- |
| `bionemo_ir`               | `pip install -v -e '.[dev]'` (from repo root)                      | All phases                        |
| `safetensors`              | `pip install safetensors`                                          | Optional checkpoint serialization |
| `triton`                   | `pip install triton==3.5.0`                                        | Triton fused kernels              |
| `cuequivariance`           | `pip install cuequivariance==0.8.1`                                | CUEQUIV attention backend         |
| `nvidia-cutlass-dsl[cu13]` | `pip install 'nvidia-cutlass-dsl[cu13]>=4.4.2'`                    | CuTeDSL attention backend         |
| `pytest`                   | `pip install pytest`                                               | Running equivalence tests         |

### Step 1 — Survey the Torch backend (`bionemo_ir/_torch/`)

The Torch backend provides optimized `nn.Module` implementations that run in
PyTorch eager or `torch.compile` mode. They use fused kernels (Triton, CUTLASS
DSL, cuEquivariance) under the hood but remain standard PyTorch modules with
`state_dict()`.

1. **Composite modules** — Search `bionemo_ir/_torch/layers/transformers/`
   for full module implementations:

   - `pairformer.py` — `PairformerLayerV1`, `PairformerLayerV2`,
     `PairformerNoSeqLayer`, `PairformerModule`, `PairformerNoSeqModule`
   - `diffusion_transformer.py` — `DiffusionTransformerLayer`,
     `DiffusionTransformerModule`
   - `atom.py` — `AtomTransformerLayer`, `AtomTransformerModule`
   - Other files in that directory

1. **Primitive layers** — Search `bionemo_ir/_torch/layers/` for
   building-block layers:

   - `triangle_nodes.py` — `TriangleMultiplicationNode`,
     `TriangleAttentionStartingNode`, `TriangleAttentionEndingNode`
   - `attention.py` — `AttentionPairBias`, `TriangleAttention`
   - `transition.py` — `Transition`, `ConditionedTransitionBlock`
   - `normalization.py` — `AdaLN`
   - `linear.py` — `Linear`

1. **Attention backends** — Search `bionemo_ir/_torch/attention_backend/`
   for available attention implementations:

   - **Triangle attention** backends: `"VANILLA"`, `"CUEQUIV"` (cuEquivariance,
     **default**), `"CuTeDSL"` (CUTLASS DSL).
   - **Pairwise attention (AttentionPairBias)** backends: `"VANILLA"`, `"SDPA"`
     (PyTorch scaled dot-product attention, **default**).

1. **Config classes** — Check `bionemo_ir/configs/modules.py` for existing
   `BaseConfig` subclasses (e.g., `PairformerConfig`,
   `DiffusionTransformerConfig`).

1. **Weight conversion helpers** — Check `bionemo_ir/models/*/convert.py`
   for reusable per-component functions (e.g., `get_tri_attn_node_weights`,
   `get_tri_mul_node_weights`, `get_transition_weights`).

For each source sub-module, record whether a Torch-backend counterpart exists
at the composite or primitive level.

### Step 2 — Determine Torch backend support

For the source module, produce a **Torch backend support matrix**. For each
source sub-module, assess whether the Torch backend can support conversion;
when it cannot, state the specific reason.

#### Torch backend blockers

When a source sub-module cannot be converted to the Torch backend, classify
the reason:

| Blocker                           | Description                                                                                                                                                                                                                                                         |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **No primitive match**            | No `nn.Module` in `_torch/layers/` implements the same operation (e.g., a novel gating mechanism or custom geometric layer).                                                                                                                                        |
| **Incompatible math**             | A BioIR primitive exists but computes a different mathematical operation — for example, the source uses a non-SwiGLU activation in its transition block while BioIR `Transition` hardcodes SwiGLU.                                                                  |
| **Dimensional constraint**        | The source model's dimensions violate BioIR assumptions — e.g., hidden dim not divisible by the required head count, or a non-standard expansion factor that doesn't match the `2 * hidden` fusion layout.                                                          |
| **Unsupported attention pattern** | The source uses an attention variant not covered by any available attention backend (triangle: `VANILLA`, `CUEQUIV`, `CuTeDSL`; pairwise: `VANILLA`, `SDPA`) — e.g., a custom sparse attention pattern, windowed triangle attention, or non-standard masking logic. |
| **Feature gap**                   | The source requires a feature the Torch backend doesn't support — e.g., custom bias terms, non-standard normalization placement, auxiliary outputs consumed downstream.                                                                                             |

#### Support matrix format

Report to the user in this format:

```text
Sub-Module              | Torch | Blocker (if any)
------------------------|-------|------------------
TriangleMultiplication   | YES   | —
TriangleAttention        | YES   | —
CustomGatingLayer        | NO    | no primitive match
Transition (SwiGLU)      | YES   | —
Transition (GeLU variant)| NO    | incompatible math (GeLU vs SwiGLU)
AttentionPairBias        | YES   | —
SparseAttention          | NO    | unsupported attention pattern
```

Then determine the overall conversion options:

- **Full conversion** — All sub-modules have Torch-backend counterparts. The
  converted module runs as an optimized `nn.Module`.
- **Partial conversion** — Some sub-modules have blockers. These remain in the
  source model's original eager PyTorch. The rest can still be converted. Report
  which sub-modules are left unconverted and why.

**Report the support matrix to the user before proceeding.** For any sub-module
with blockers, include actionable guidance: whether the gap can be closed by
writing new BioIR code, or whether the sub-module should stay in eager
PyTorch.

#### Feasibility report for BioIR developers

After completing Steps 1–2, produce a **feasibility report** as a markdown file
at `$WORKDIR/feasibility_report.md`. This report consolidates all Phase 1
findings into a single handoff document for BioIR developers. Use the template
below — fill every section, remove nothing, mark empty sections with "None".

```markdown
# BioIR Onboarding Feasibility Report

## 1. Module Overview

| Field | Value |
|---|---|
| Source module class | `<ClassName>` |
| Source file | `<path/to/module.py>` |
| Checkpoint | `<path or "random weights">` |
| Parameter count | `<N>M` |
| Closest BioIR equivalent | `<e.g., PairformerModule, DiffusionTransformerModule, or "none">` |
| Number of sub-modules | `<N>` |
| Number of stacked layers | `<N>` |
| Inference dtype | `<bfloat16 / float16 / float32>` |

## 2. Torch Backend Support Matrix

| # | Source Sub-Module | Source Class | Torch Backend | Blocker (if any) |
|---|---|---|---|---|
| 1 | `<name>` | `<SourceClass>` | YES / NO | `<blocker or "—">` |
| 2 | `<name>` | `<SourceClass>` | YES / NO | `<blocker or "—">` |
| ... | ... | ... | ... | ... |

**Conversion path:** `<Full / Partial>`

## 3. Sub-Module Mapping

| # | Source Sub-Module | Source Class | BioIR Torch Class | Notes |
|---|---|---|---|---|
| 1 | `<name>` | `<SourceClass>` | `<TorchClass or "GAP">` | `<notes>` |
| 2 | ... | ... | ... | ... |

## 4. Hyperparameter Mapping

| Parameter | Source Value | BioIR Field | Match? | Notes |
|---|---|---|---|---|
| hidden_dim | `<value>` | `token_s` / `token_z` | YES / NO | |
| num_heads | `<value>` | `num_heads` | YES / NO | |
| num_layers | `<value>` | `num_blocks` | YES / NO | |
| head_dim | `<value>` | `head_dim` | YES / NO | |
| expansion_factor | `<value>` | `<field>` | YES / NO | |
| ... | ... | ... | ... | ... |

## 5. Forward Signature Differences

| Aspect | Source | BioIR | Adapter Needed? |
|---|---|---|---|
| Mask format | `<e.g., bool padding mask>` | `<e.g., float valid mask>` | YES / NO |
| Extra BioIR args | — | `<e.g., attn_metadatas, precomputed_masks>` | YES |
| Return type | `<e.g., Tensor>` | `<e.g., Tensor>` | YES / NO |
| ... | ... | ... | ... |

## 6. Blockers Requiring BioIR Development

| # | Sub-Module | Blocker Type | Description | Estimated Effort |
|---|---|---|---|---|
| 1 | `<name>` | `<blocker type>` | `<description>` | `<Low / Medium / High>` |
| ... | ... | ... | ... | ... |

## 7. Inference-Irrelevant Features (to strip)

| Feature | Location | Notes |
|---|---|---|
| `<dropout / activation_checkpointing / aux_loss / ...>` | `<file:line>` | `<notes>` |
| ... | ... | ... |

## 8. Recommended Plan

| Phase | Action | Depends On | Estimated Effort |
|---|---|---|---|
| 1 | Torch conversion (all supported sub-modules) | — | `<effort>` |
| 2 | Weight conversion + validation | Phase 1 | `<effort>` |
| 3 | Close blocker: `<description>` | — | `<effort>` |
| ... | ... | ... | ... |

## 9. Open Questions

- `<Any unresolved questions for the maintainers>`
- ...
```

### Step 3 — Analyze architectural differences

Compare the source module against the closest BioIR equivalent(s). Document:

1. **Sub-module mapping table** — For each source sub-module: source class,
   Torch-backend class (or gap), file paths.
1. **Forward signature differences** — Compare input/output signatures. Common
   differences:
   - Mask format (bool padding mask vs float valid mask vs precomputed bias)
   - Extra BioIR args (`attn_metadatas`, `precomputed_masks`, `buffers`)
   - Return type (tuple ordering, extra outputs)
1. **Hyperparameter mapping** — Map constructor args: hidden dims, head counts,
   expansion factors, epsilon values, etc. Note any without a direct equivalent.
1. **Inference-irrelevant features** — List source features to strip: dropout,
   activation checkpointing, training-mode branches, auxiliary losses.

Report findings to the user before proceeding.

## Phase 2 — Weight Conversion

The Torch backend (`_torch/`) consumes weights via `load_weights()` /
`recursive_calling_load_weights`. Perform weight conversion for every sub-module
that Phase 1 Step 2 marked as supported.

### Step 1 — Extract source weight names

If a source checkpoint was provided (Phase 0 Step 3), load it and list all
parameter keys for the module being converted. If no checkpoint was provided,
instantiate the source module with the correct hyperparameters and use
`model.state_dict().keys()` to get the key names — the values will be
random-initialized but the names and shapes are what matter here.

### Step 2 — Extract BioIR weight names

Instantiate the `nn.Module` from `_torch/layers/` with matching hyperparameters
and print `state_dict().keys()`, or read the source to identify `nn.Parameter` /
`nn.Linear` / `nn.LayerNorm` attributes.

### Step 3 — Build the weight name mapping

Create a complete mapping from source keys to BioIR keys. This MUST account
for:

1. **Name renames** — Different attribute names for the same logical weight
   (e.g., `layer_norm_1` vs `norm`, `to_out` vs `o_proj`).
1. **Weight fusions** — BioIR fuses certain weights for performance. Check
   `bionemo_ir/models/*/convert.py` for the established fusion patterns.
   Common fusions:
   - Separate Q, K, V projections fused into a single `qkv_proj` via
     `torch.cat([q, k, v], dim=0)`
   - Separate K, V projections fused into `proj_kv` via
     `torch.cat([k, v], dim=0)`
   - Parallel gate + input linear layers fused into `fused_fc2_fc1` via
     `torch.cat([gate, input], dim=0)` (note ordering!)
1. **Shape transforms** — Transpositions, reshapes, or padding needed due to
   different layout conventions.
1. **Missing weights** — BioIR weights with no source equivalent (should be
   initialized, not converted). Source weights with no BioIR equivalent
   (should be discarded or flagged).

### Step 4 — Write the conversion function

Create `$WORKDIR/convert/convert_weights.py` following the established pattern
in `bionemo_ir/models/*/convert.py`:

```python
def convert_<module>_weights(state_dict, prefix, bioir_prefix, mapping, dtype, ...):
    """Convert source checkpoint weights to BioIR format.

    The returned dict is consumed by the Torch backend via module.load_weights(weights).
    """
    # 1. Read source weights by key
    # 2. Apply fusions (cat, reshape, etc.)
    # 3. Apply dtype conversion
    # 4. Return dict with BioIR key names
```

**Critical: Reuse existing per-component helpers** from
`bionemo_ir/models/boltz1/convert.py` when the sub-component weight layout
matches (e.g., `get_tri_mul_node_weights`, `get_tri_attn_node_weights`,
`get_transition_weights`). Only write new
conversion code for sub-components whose weight layout genuinely differs from
what these helpers expect.

### Step 5 — Serialize (optional)

The conversion function produces an in-memory weight dict. For the Torch
backend:

- Load directly into the `nn.Module` via
  `module.load_weights(converted_weights)`.
- Optionally cache to `$WORKDIR/convert/<module>_ckpt/` via `torch.save()` or
  `safetensors.torch.save_file()` for reuse.

### Step 6 — Validate conversion completeness

1. **Weight completeness** — compare converted keys against
   `nn_module.state_dict().keys()`:
   - No missing weights (every BioIR key has a value).
   - No extra weights (every source key was consumed or explicitly discarded).
1. **Shape match** — each converted tensor's shape matches the `nn.Module`
   parameter's shape.
1. **Dtype match** — each converted tensor has the target dtype.
1. **Load test** — call `module.load_weights(converted_weights)` and confirm it
   succeeds without error.

## Phase 3 — Integration

All integration code goes under `$WORKDIR/integration/`. Import
`bionemo_ir` as an installed package and the source model's module from its
source path.

### Step 1 — Write the adapter wrapper

Create `$WORKDIR/integration/adapter.py` — a thin `nn.Module` that bridges the
source model's `forward()` signature to the BioIR module's `forward()`. The
adapter handles:

- **Mask conversion** — Transform the source model's mask format to BioIR's
  expected format.
- **Argument bridging** — Supply BioIR-specific args with sensible defaults
  (e.g., `attn_metadatas=None`, `precomputed_masks=None`).
- **Output reshaping** — Match the source model's expected return type.
- **Inference-mode stripping** — Do not replicate dropout or training-only
  logic.

The adapter must be a **drop-in replacement**: same `forward()` signature as the
source module, so the source model's calling code doesn't change.

### Step 2 — Write the module swap function

Create `$WORKDIR/integration/swap.py` — a function that patches the source
model's model in-place:

1. Instantiate the BioIR module with a config matching the source model's
   hyperparameters.
1. Load and convert weights using the function from Phase 2.
1. Load converted weights into the BioIR module.
1. Wrap in the adapter from Step 1.
1. Replace the source model's sub-module attribute (e.g.,
   `model.trunk.pairformer = adapter`).

### Step 3 — Write or reuse the config

Check `bionemo_ir/configs/modules.py` for an existing `BaseConfig`
subclass that fits. If one exists, instantiate it with the source model's
hyperparameters. If not, create a minimal new one — only add fields that the
source model's module actually needs.

Key config fields to set:

- Tensor dimensions (`token_s`, `token_z`, hidden sizes)
- Head configuration (`num_heads`, `pairwise_head_width`, `pairwise_num_heads`)
- `dtype` (usually `"bfloat16"` for inference)
- Attention backend — use `triangle_attention_backend="CUEQUIV"` and
  `pairwise_attention_backend="SDPA"` (these are the defaults).

## Phase 4 — Hierarchical Equivalence Tests

Create test files under `$WORKDIR/tests/`. Run them with
`pytest $WORKDIR/tests/`.
**Bottom-up testing — each level must pass before the next.**

These tests work with both real and random weights. When no source checkpoint
was provided, use random-initialized weights — the tests validate that the
conversion pipeline and adapter produce **identical outputs** for both modules
given the same weights and inputs, regardless of whether the weights are
meaningful.

### Step 1 — Sub-component equivalence

Test each primitive sub-module independently: identical weights + identical
input → compare output.

- **TriangleMultiplication, TriangleAttention, AttentionPairBias, Transition,
  etc.**
- Use small tensor sizes (e.g., `N_res=32`, `hidden=64`) for speed.
- Use `torch.testing.assert_close` with tight tolerance for identical-math
  blocks (atol=1e-3, rtol=1e-3).
- For blocks using different math (e.g., fused kernels, different attention
  implementations), use RMSE-based comparison with relaxed tolerance (rmse_ratio
  \< 0.05).

### Step 2 — Layer equivalence

Test one full layer (e.g., one PairformerBlock / one DiffusionTransformerLayer):
source layer vs BioIR Torch layer via adapter wrapper.

- Load identical weights (via conversion function).
- Feed identical random inputs.
- Compare outputs with relaxed tolerance (atol=1e-2, rtol=1e-2 or rmse_ratio \<
  0.05).

### Step 3 — Full module equivalence

Test the complete stacked module (e.g., 2-3 layers, not the full depth):

- Confirm error does not diverge across layers.
- Use representative input shapes from the source model's use case.

### Step 4 — Weight round-trip test

Verify the conversion is lossless:

```python
# source_state -> convert -> load into BioIR -> extract state_dict -> compare shapes/values
```

## Phase 5 — Performance Benchmarks (Optional)

Once correctness is confirmed (Phase 4 passes), benchmark against the source
model's original module. **All benchmarks use bfloat16 precision** — this is the
target inference dtype.

### Benchmark table format

The baseline is always the
**source model's own e2e model with its default configuration** (bfloat16). This
means
using whatever attention backends, kernel settings, and optimizations the
source model ships — not a stripped-down module in isolation. The baseline must
reflect the real-world performance the source model currently achieves. Report
results in this table format:

```text
Input Shape     | Baseline (Source e2e, bf16) | [TriAttn=CUEQUIV, AttnPB=SDPA] | [TriAttn=CuTeDSL, AttnPB=CuTeDSL]*
----------------|-------------------------------|--------------------------------|-------------------------------------
N_res=128       | X ms                          | Y ms (Z× speedup)             | Y ms (Z× speedup)
N_res=256       | X ms                          | Y ms (Z× speedup)             | Y ms (Z× speedup)
N_res=384       | X ms                          | Y ms (Z× speedup)             | Y ms (Z× speedup)
N_res=512       | X ms                          | Y ms (Z× speedup)             | Y ms (Z× speedup)
N_res=640       | X ms                          | Y ms (Z× speedup)             | Y ms (Z× speedup)
...             | ...                           | ...                            | ...
N_res=2048      | X ms                          | Y ms (Z× speedup)             | Y ms (Z× speedup)
Peak GPU memory | X MB                          | Y MB                           | Y MB

*  CuTeDSL column: only if GPU SM version supports it (SM80+ for CUTLASS DSL kernels). Skip this column if not available.
```

Sweep `N_res` from 128 to 2048 with step size 128 (i.e., 128, 256, 384,
512, ..., 1920, 2048). If a configuration runs out of GPU memory at a given
size, report OOM and stop that column — do not skip to larger sizes.

### Benchmark columns

All columns use **bfloat16** dtype — both for the source baseline and all
BioIR configurations.

| Column                                  | Config                                                                                                                            | Notes                                                                                                                                                            |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Baseline (Source e2e, bf16)**         | The source model's e2e model with its default config (attention backends, kernels, `torch.compile`, etc.), `dtype=torch.bfloat16` | Always present. This is the reference. Must inherit the source model's own optimizations — use their recommended inference settings, not a naive eager fallback. |
| **\[TriAttn=CUEQUIV, AttnPB=SDPA\]**    | Torch backend, `dtype=bfloat16`, `triangle_attention_backend="CUEQUIV"`, `pairwise_attention_backend="SDPA"`                      | Default Torch backend config. Always present.                                                                                                                    |
| **\[TriAttn=CuTeDSL, AttnPB=CuTeDSL\]** | Torch backend, `dtype=bfloat16`, `triangle_attention_backend="CuTeDSL"`, `pairwise_attention_backend="CuTeDSL"`                   | Only if GPU SM version supports CuTeDSL. Check with `torch.cuda.get_device_capability()` — requires SM80+.                                                       |

### Benchmark procedure

1. **Use a realistic layer stack, not a single layer.** Benchmark at least 8
   layers in a loop (matching a typical transformer stack depth). Each iteration
   feeds the previous layer's output into the next. This captures layer-to-layer
   memory reuse, kernel launch amortization, and realistic activation memory
   pressure — a single-layer benchmark understates memory usage and overstates
   throughput.
1. **Shared weights across all configs.** Instantiate the source module stack,
   extract each layer's `state_dict()`, convert via the Phase 2 conversion
   function, and load the converted weights into every BioIR configuration.
   This ensures all columns benchmark with identical weights — differences in
   latency are purely from the execution backend, not from weight values.
1. Sweep `N_res` from 128 to 2048 with step size 128.
1. All modules and inputs use `torch.bfloat16`. For the source baseline, cast
   the module and inputs to bfloat16 before benchmarking.
1. Warm up each configuration with 3 runs, then measure over 10 runs. Report
   median latency.
1. Measure peak GPU memory for each configuration.
1. All benchmarks run with `torch.no_grad()` and `model.eval()`.

Store benchmark scripts under `$WORKDIR/benchmarks/`. Save results automatically
to `$WORKDIR/results/` in two formats:

- **CSV** (`bench_<timestamp>.csv`) — tabular data for spreadsheets and quick
  comparison.
- **JSON** (`bench_<timestamp>.json`) — full metadata including GPU name, SM
  version, dtype, layer count, warmup/repeat counts, hyperparameters, per-row
  latencies with memory, and peak memory. This enables reproducibility and
  automated comparison across runs.

## Phase 6 — Summary Report

Print (not file) after completion:

1. Module overview — what was converted, which sub-components matched, any gaps
1. Working directory location (`$WORKDIR`) and files created (tree listing)
1. Weight conversion summary — number of params, fusions applied, any discarded
   weights
1. Test results table — test name | what it validates | PASS/FAIL
1. Known limitations (features not supported, precision differences)
1. Recommended next steps (upgrade attention backend, enable TP, try CuTeDSL)

## Key Gotchas

- **Never write into the BioIR codebase.** All generated files (conversion
  scripts, checkpoints, adapters, tests, benchmarks) go under `$WORKDIR`. The
  BioIR repo and install directory are read-only dependencies — import from
  them, never modify them. Same applies to the source repository.
- **Use the optimized PyTorch backend.** This skill converts source modules to
  BioIR's `_torch/` implementations. TensorRT engine build is out of scope for
  this release.
- **Weight fusion ordering matters.** When fusing gate + input into
  `fused_fc2_fc1`, the gate weight comes first:
  `torch.cat([gate, input], dim=0)`. Getting this wrong produces silent
  numerical errors, not crashes.
- **Mask polarity.** Source models often use `is_padding=True` for padded
  positions. BioIR uses `mask=1.0` for **valid** positions. Invert carefully.
- **Precomputed masks for performance.** BioIR attention backends support
  precomputed mask biases (`precompute_pair_masks`, `precompute_single_masks`).
  Compute these once outside the layer loop, not per-layer.
- **Dropout is inference-irrelevant.** Source modules often have dropout in
  `forward()`. BioIR modules do not apply dropout (they're inference-only). Do
  not add dropout to the adapter.
- **dtype matters.** BioIR modules expect explicit dtype at construction time.
  Some internal paths (e.g., `s_path_dtype`) may use a different precision than
  the main dtype. Match the source model's precision policy.
- **Reuse existing conversion helpers.** Check `bionemo_ir/models/*/convert.py`
  — the per-component weight conversion functions (`get_tri_attn_node_weights`,
  `get_tri_mul_node_weights`, `get_transition_weights`) are designed to be
  reusable across models. Only write new ones when the source model's weight
  layout genuinely differs.
- **Torch attention defaults: CUEQUIV + SDPA.** The Torch backend defaults are
  `triangle_attention_backend="CUEQUIV"` and
  `pairwise_attention_backend="SDPA"`. Use these for initial correctness
  validation. `"CuTeDSL"` is available as a further optimization for triangle
  attention but may have different numerical behavior — re-run tests after
  switching.
- **DiffusionTransformer with fixed pair bias: detect the pattern, then use
  `OpenFold3DiffusionTransformer` with `precompute_bias=True` by default.**
  During Phase 1 survey, check whether the source model's DiffusionTransformer
  layers project a **fixed** pair representation `z` into an attention bias via
  `LayerNorm + Linear` on every layer (i.e., `z` is passed unchanged into every
  layer's pair bias projection). If yes, this is the precomputed-bias pattern:
  all per-layer `LN(z) @ W_proj.T` calls collapse into a single mega-GEMM before
  the loop by fusing each layer's LN γ into one `W_mega [N*H, D]` matrix. In
  that case, use `OpenFold3DiffusionTransformer` (not a raw
  `DiffusionTransformerLayer` loop) with `bias_proj=True` and
  `precompute_bias=True` (the default). This turns O(N²) per-layer pair
  projections into a single constant-time GEMM — up to 14.8× speedup at large
  N_res. Skip this if `z` is updated between layers (e.g. EvoformerStack-style),
  or if the source module has no pair bias projection at all. Two
  implementation gotchas when using this path:
  1. `DiffusionTransformerConfig` has no `version` field — pass `version="v1"`
     as an extra field (allowed by `extra = "allow"` on `BaseConfig`).
  1. `OpenFold3DiffusionTransformer.__init__` replaces `proj_z[0]` with a
     `bias=False` LayerNorm when `bias_proj=True`. Use `load_state_dict` (not
     `load_weights`) and filter out `*proj_z.0.bias` keys before loading:

     ```python
     weights = {k: v for k, v in weights.items() if not k.endswith("proj_z.0.bias")}
     module.load_state_dict(weights, strict=True)
     ```

- **`load_weights` vs `load_state_dict`.** BioIR composite modules (e.g.,
  `PairformerModule`) use `load_weights(weights_dict)` which calls
  `recursive_calling_load_weights` internally. This is NOT the same as PyTorch's
  `load_state_dict` — it handles TP sharding and custom loading logic.
