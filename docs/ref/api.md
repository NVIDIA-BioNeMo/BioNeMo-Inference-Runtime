<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# BioNeMo Inference Runtime Python API

This is the public Python API for structure prediction with BioIR. It covers
the two supported ways to run a model:

1. **`build_processor`** — parse sequences and MSAs, featurize, run inference,
   and write PDB/CIF. This is the production entry point.
2. **Model constructor + `forward`** — construct an `nn.Module`, load weights,
   and call it on a feature dict you already have.

A runnable wrapper around (1) lives at
[`examples/folding/run_demo.py`](../../examples/folding/run_demo.py).
Supported models, GPUs, and fused kernels:
[`support-matrix.md`](support-matrix.md).

`import bionemo_ir` registers every model factory. Any import that
pulls in `bionemo_ir.registry` or `bionemo_ir.models.*` does this
transitively.

## When to use which API

| Goal                                                                                              | API                                                 |
| ------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| Sequences / MSAs → PDB or CIF, including Ray multi-GPU                                            | [`build_processor`](#build_processor)               |
| Inference on a feature dict you already have (custom dataloader, composing models) — not training | [Model constructor](#model-constructor-and-forward) |
| Swap a Pairformer / DiT / Evoformer in *your* architecture, or port pairwise memory optimizations | [Custom architectures](#custom-architectures)       |

Tokenizer and feature-factory objects from the registry are **pipeline specs**,
not callables. They are wired by `build_processor`. There is no
`tokenizer(request)` / `features.generate_features(...)` helper on the public
surface; going from an [`InputRequest`](#input-requests) to a feature dict is
what the processor is for.

## Input requests

The processor consumes a list of row dicts. Each row must include `record`,
an [`InputRequest`][inputrequest]:

```python
from bionemo_ir.data.schemas import InputRequest, MSARecord, Polymer

request = InputRequest(
    input_id="demo",
    polymers=[
        Polymer(
            polymer_type="protein",
            chain_id=["A"],
            sequence="GSHMSL...",
            msas=[MSARecord(path="msa.a3m", format="a3m")],
            paired_msas=[],
            templates=None,
        ),
    ],
)
```

[`Polymer`][polymer] fields:

| Field          | Type                       | Meaning                                                                                                              |
| -------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `polymer_type` | `str`                      | `"protein"`, `"rna"`, `"dna"`, `"ccd_ligand"`, or `"smiles_ligand"`                                                  |
| `chain_id`     | `str` or `list[str]`       | 1–4 alphanumeric characters per id. A list of ids on one polymer is a homo-oligomer (same sequence, several chains). |
| `sequence`     | `str`                      | 1-letter protein/NA sequence; CCD code or `_`-joined CCD list (`"ATP"`, `"ATP_FAD"`); or a SMILES string             |
| `msas`         | `list[MSARecord]`          | Unpaired a3m (path and/or inline `content`)                                                                          |
| `paired_msas`  | `list[MSARecord]`          | Paired a3m, same `MSARecord` as `msas`. One file per chain; pairing is by row index (see below)                      |
| `templates`    | `list[Template]` or `None` | Protein-only. `format` is `"cif"` or `"pdb"`. Hits you already have — BioIR does not run HHsearch / HMMsearch        |

[`MSARecord`][msarecord] / [`Template`][template] take either `path` or inline
`content`, plus `format` (`"a3m"` for MSAs; `"cif"` or `"pdb"` for templates).
Template `chain_id` selects which chain of a multi-chain CIF or PDB to use;
`None` auto-selects.

Paired MSAs are ordinary A3M (`format="a3m"`), not CSV and not a concatenated
multi-chain alignment. Each protein polymer gets its own file covering
**that chain only**. Row 0 is the query; row *k* on every chain is one pairing
group, so the files must have the same number of records (AF2 multimer
enforces this). Example (chain A; chain B has the same headers and row
count, sequences aligned to B):

```text
>query
SNAELFNLESRVEIEKSLTQMEDVLKALQMKLWEAESKLSFATCKS
>tr1
-DKELFNLESRVEIEKSLKQMEDVLKALQTKLWEVESKLSFTSCKS
```

Lowercase letters are deletions (standard A3M). Bundled files look like
[`7sfy_0_paired.a3m`](../../examples/data/samples/heterooligomers/msas/7sfy_0_paired.a3m)
and
[`7sfy_1_paired.a3m`](../../examples/data/samples/heterooligomers/msas/7sfy_1_paired.a3m)
(one paired A3M per chain, same row count).

The declarative JSON used under `examples/data/samples/` is the same shape.
A string `msas` path is accepted by `examples/folding/run_demo.py` and resolved
relative to the JSON file; the Python schema wants `list[MSARecord]`.

```json
[
  {
    "input_id": "T1031",
    "polymers": [
      {
        "polymer_type": "protein",
        "chain_id": ["A1"],
        "sequence": "ACKIENIKYKGKEVESKLGSQLIDIFNDLDRAKEEYDKLSSPEFIAKFGDWINDEVERNVNEDGEPLLIQDVRQDSSKHYFFILKNGERFDLLTR",
        "msas": "msas/T1031.a3m",
        "paired_msas": null,
        "templates": null
      }
    ]
  }
]
```

Templates are protein-only. `format` is `"cif"` or `"pdb"`. Pass hits you
already have — BioIR does not run HHsearch / HMMsearch. `chain_id` selects
which chain of a multi-chain CIF or PDB to use; omit it (or `null`) to
auto-select. Bundled sample:
[`T1047s1_with_template.json`](../../examples/data/samples/monomers/T1047s1_with_template.json)
with
[`8wle_A.cif`](../../examples/data/samples/monomers/templates/8wle_A.cif).

```python
from bionemo_ir.data.schemas import Template

templated = InputRequest(
    input_id="T1047s1_with_template",
    polymers=[
        Polymer(
            polymer_type="protein",
            chain_id=["A1"],
            sequence="MQKNAAHTYAISSLLVLSLTGCAWIPSTPLVQGATSAQPVPGPTPVANGSIFQSAQPINYGYQPLFEDRRPRNIGDTLTIVLQENVSASKSSSANASRDGKTNFGFDTVPRYLQGLFGNARADVEASGGNTFNGKGGANASNTFSGTLTVTVDQVLVNGNLHVVGEKQIAINQGTEFIRFSGVVNPRTISGSNTVPSTQVADARIEYVGNGYINEAQNMGWLQRFFLNLSPM",
            msas=[MSARecord(path="msa.a3m", format="a3m")],
            templates=[
                Template(path="templates/8wle_A.cif", format="cif", chain_id="A"),
            ],
        ),
    ],
)
```

```json
[
  {
    "input_id": "T1047s1_with_template",
    "polymers": [
      {
        "polymer_type": "protein",
        "chain_id": ["A1"],
        "sequence": "MQKNAAHTYAISSLLVLSLTGCAWIPSTPLVQGATSAQPVPGPTPVANGSIFQSAQPINYGYQPLFEDRRPRNIGDTLTIVLQENVSASKSSSANASRDGKTNFGFDTVPRYLQGLFGNARADVEASGGNTFNGKGGANASNTFSGTLTVTVDQVLVNGNLHVVGEKQIAINQGTEFIRFSGVVNPRTISGSNTVPSTQVADARIEYVGNGYINEAQNMGWLQRFFLNLSPM",
        "msas": "msas/T1047s1.a3m",
        "paired_msas": null,
        "templates": [
          {
            "path": "templates/8wle_A.cif",
            "format": "cif",
            "chain_id": "A"
          }
        ]
      }
    ]
  }
]
```

RNA, DNA, and ligands are Boltz-1/2 and OpenFold3 only (AF2 / OF2 are
protein-only). Nucleic-acid and ligand chains carry no MSA. A CCD ligand
uses `polymer_type="ccd_ligand"` and a CCD code in `sequence` (`"ATP"` or
`"ATP_FAD"`). Bundled complexes:
[`examples/data/samples/rna_dna_ligand/`](../../examples/data/samples/rna_dna_ligand/).

```python
rna = InputRequest(
    input_id="rna_demo",
    polymers=[
        Polymer(
            polymer_type="rna",
            chain_id=["A"],
            sequence="UUGGGUUCCCUCACCCCAAUCAUAAAAA",
        ),
    ],
)

dna = InputRequest(
    input_id="dna_demo",
    polymers=[
        Polymer(
            polymer_type="dna",
            chain_id=["A"],
            sequence="CGTACGATCGTA",
        ),
    ],
)

# Protein + custom SMILES ligand. Protein still needs an unpaired MSA.
smiles = InputRequest(
    input_id="smiles_demo",
    polymers=[
        Polymer(
            polymer_type="protein",
            chain_id=["A"],
            sequence="MYTVKPGDTMWKIAVKYQIGISEIIAANPQIKNPNLIYPGQKINIPNILEHHHHHH",
            msas=[MSARecord(path="msa.a3m", format="a3m")],
        ),
        Polymer(
            polymer_type="smiles_ligand",
            chain_id=["B"],
            sequence="N[C@@H](Cc1ccc(O)cc1)C(=O)O",
        ),
    ],
)
```

Same shape in JSON (`smiles_demo.json` / `R1117v2.json` in that sample
dir; there is no bundled DNA JSON — DNA is the RNA shape with ACGT):

```json
[
  {
    "input_id": "rna_demo",
    "polymers": [
      {
        "polymer_type": "rna",
        "chain_id": ["A"],
        "sequence": "UUGGGUUCCCUCACCCCAAUCAUAAAAA",
        "msas": null,
        "paired_msas": null,
        "templates": null
      }
    ]
  },
  {
    "input_id": "dna_demo",
    "polymers": [
      {
        "polymer_type": "dna",
        "chain_id": ["A"],
        "sequence": "CGTACGATCGTA",
        "msas": null,
        "paired_msas": null,
        "templates": null
      }
    ]
  },
  {
    "input_id": "smiles_demo",
    "polymers": [
      {
        "polymer_type": "protein",
        "chain_id": ["A"],
        "sequence": "MYTVKPGDTMWKIAVKYQIGISEIIAANPQIKNPNLIYPGQKINIPNILEHHHHHH",
        "msas": [{"path": "msas/T1152_0.a3m", "format": "a3m"}],
        "paired_msas": null,
        "templates": null
      },
      {
        "polymer_type": "smiles_ligand",
        "chain_id": ["B"],
        "sequence": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
        "msas": null,
        "paired_msas": null,
        "templates": null
      }
    ]
  }
]
```

Per-model coverage (monomer / MSA / templates / nucleic acids / ligands):
[support matrix — models and data pipeline](support-matrix.md#models-and-data-pipeline).

## `build_processor`

`build_processor(config)` in
`bionemo_ir.pipeline.processor.engine_proc` builds a five-stage pipeline:

```text
Parser → Tokenizer → Feature generator → Folding engine → Writer
```

It returns a `SerialProcessor` when `config.executor_backend is None`, or a
Ray `Processor` when `config.executor_backend == "ray"`.

Metadata (Boltz CCD + mols) and per-model `runtime_args` are filled in
automatically if you omit them. User-supplied keys win over registry defaults.

### Hello world (serial)

```python
from bionemo_ir.data.schemas import InputRequest, MSARecord, Polymer
from bionemo_ir.pipeline.processor.engine_proc import (
    EngineProcessorConfig,
    build_processor,
)
from bionemo_ir.pipeline.stages.configs import (
    FeatureGeneratorStageConfig,
    WriterStageConfig,
)

NAME = "boltz-2"

config = EngineProcessorConfig(
    model_source=NAME,
    executor_backend=None,  # in-process SerialProcessor
    runtime_args={
        "recycling_steps": 3,
        "num_sampling_steps": 50,
        "diffusion_samples": 1,
    },
    feature_generator_stage=FeatureGeneratorStageConfig(
        init_context={"random_seed": 42},
    ),
    writer_stage=WriterStageConfig(output_path="output", format="cif"),
)
processor = build_processor(config)

request = InputRequest(
    input_id="demo",
    polymers=[
        Polymer(
            polymer_type="protein",
            chain_id=["A"],
            sequence="GSHMSL...",
            msas=[MSARecord(path="msa.a3m", format="a3m")],
        )
    ],
)
rows = [{"record": request, "__record_id": request["input_id"]}]
outputs = processor(rows)

# outputs[i]["output_path"] → output/demo.cif
# json.loads(outputs[i]["scores"]) → pLDDT / pTM / ipTM / …
```

Each input row:

| Key           | Required    | Meaning                                                                                                |
| ------------- | ----------- | ------------------------------------------------------------------------------------------------------ |
| `record`      | yes         | `InputRequest` (or a dict with the same keys)                                                          |
| `__record_id` | recommended | Becomes the output filename stem (`output/{id}.cif`)                                                   |
| `random_seed` | no          | Not read by the tokenizer/feature `pre_init` hooks. Seed via `init_context` (below) or the process RNG |

`SerialProcessor.__call__` takes `list[dict]` and returns `list[dict]`.

### Ray (multi-GPU replicas)

Ray is the recommended executor for **large inference on a GPU cluster**.
Staged `map_batches` overlaps parser / tokenizer / featurizer / writer with
GPU forwards, so pre- and post-processing latency is hidden behind the
engine. Serial (`executor_backend=None`) is for debugging or single-process
measurements; it does not overlap those stages.

```python
import ray
from bionemo_ir.pipeline.stages.configs import (
    EngineStageConfig,
    FeatureGeneratorStageConfig,
    ParallelismMode,
    ParserStageConfig,
    TokenizerStageConfig,
    WriterStageConfig,
)

config = EngineProcessorConfig(
    model_source="boltz-2",
    executor_backend="ray",
    parser_stage=ParserStageConfig(compute=4),
    tokenizer_stage=TokenizerStageConfig(compute=4, num_cpus=2),
    feature_generator_stage=FeatureGeneratorStageConfig(
        compute=8, num_cpus=4, init_context={"random_seed": 42}
    ),
    engine_stage=EngineStageConfig(
        parallelism_mode=ParallelismMode.REPLICA,
        compute=4,          # number of engine actors
        num_gpus=1.0,       # GPUs reserved per actor
        num_cpus=4,
    ),
    writer_stage=WriterStageConfig(
        compute=4, output_path="output", format="cif"
    ),
)
processor = build_processor(config)  # calls ray.init() if needed

ds = ray.data.from_items(rows)
out_rows = list(processor(ds).materialize().iter_rows())
```

`EngineStageConfig.compute * num_gpus` must not exceed visible GPUs, or
`build_processor` raises `ValueError`.

One-replica-per-GPU helper:

```python
config = EngineProcessorConfig.create_default_replica_mode_config(
    model_source="boltz-2",
    output_dir="output",
    output_format="cif",
)
```

That sets `executor_backend="ray"` and sizes CPU stages from
`torch.cuda.device_count()`.

### `EngineProcessorConfig`

Inherits `ProcessorConfig`. Pass only documented fields.

| Field                                                                                            | Default  | Role                                                                 |
| ------------------------------------------------------------------------------------------------ | -------- | -------------------------------------------------------------------- |
| `model_source`                                                                                   | required | FoldingSupportMatrix key                                             |
| `executor_backend`                                                                               | `None`   | `None` = serial; `"ray"` = Ray Data                                  |
| `engine_kwargs`                                                                                  | `{}`     | Passed into the folding engine (see below)                           |
| `runtime_args`                                                                                   | `{}`     | Merged on top of factory defaults, then forwarded to `model.forward` |
| `metadata`                                                                                       | `None`   | `{ccd_path, mol_dir, …}`. Auto-loaded when omitted                   |
| `metadata_loader`                                                                                | `None`   | Callable used when `metadata` is omitted                             |
| `parser_stage` / `tokenizer_stage` / `feature_generator_stage` / `engine_stage` / `writer_stage` | `True`   | `bool`, `dict`, or the matching `*StageConfig`                       |
| `batch_size`                                                                                     | `1`      | Rows per `map_batches` call                                          |
| `concurrency`                                                                                    | `1`      | Default actor pool size for CPU stages                               |
| `should_continue_on_error`                                                                       | `False`  | If `True`, failed rows get `__inference_error__` instead of raising  |
| `max_concurrent_batches`                                                                         | `8`      | Ray engine-stage overlap                                             |
| `runtime_env`                                                                                    | `None`   | Ray runtime env                                                      |
| `accelerator_type`                                                                               | `None`   | Optional Ray accelerator label                                       |

`engine_kwargs` keys consumed by the folding engine:

| Key                    | Meaning                                                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `config`               | Override the pretrained `BaseConfig` (otherwise `ModelCls.get_pretrained_config(model_source)`)                               |
| `accelerated_configs`  | `dict[str, AcceleratedConfig]` applied via `model.optimize(...)` at engine construction                                       |
| `profile_inference`    | If `True`, CUDA-sync around the forward and attach `model_inference_time` (seconds) on the row. Useful on serial; skip on Ray |
| `device`               | `DeviceConfig` (default `"auto"` → CUDA if available)                                                                         |
| `postprocessor_config` | Optional post-processor Pydantic config                                                                                       |

Stage configs (`ParserStageConfig`, `TokenizerStageConfig`,
`FeatureGeneratorStageConfig`, `EngineStageConfig`, `WriterStageConfig`) all
share `compute`, `num_cpus`, `memory`, `batch_size`, `drop_keys`. Extra fields:

- **Tokenizer / feature generator:** `init_context`. Set
  `init_context={"random_seed": N}` on the **feature-generator** stage so the
  tokenizer can fall back to the same seed (RDKit ETKDG on OpenFold3 and MSA
  augmentation stay aligned). Setting it only on the tokenizer does **not**
  seed the feature stage.
- **Writer:** `output_path`, `format` (`"pdb"`, `"cif"`, or `["pdb", "cif"]`).
- **Engine:** `parallelism_mode=ParallelismMode.REPLICA`, `num_gpus` (default
  `1.0`).

All five stages always run. The `enabled` flag on a stage config is not a
public way to skip a stage.

### Runtime args

`build_processor` starts from `get_default_runtime_args(model_source)` and
overlays `config.runtime_args`. Only pass keys the model's `forward` accepts.

#### Boltz-1 / Boltz-2

```python
model(feed_dict, recycling_steps=3, num_sampling_steps=200,
      diffusion_samples=1, max_parallel_samples=None, steering_args=None)
```

#### OpenFold3

Same Boltz-style names, mapped inside `forward`:

| `runtime_args` key   | OpenFold3 meaning                     |
| -------------------- | ------------------------------------- |
| `recycling_steps`    | `num_cycles = recycling_steps + 1`    |
| `num_sampling_steps` | `no_rollout_steps` (diffusion length) |
| `diffusion_samples`  | `no_rollout_samples`                  |

You can also pin sample count at construction:
`OpenFold3(model_name="openfold3", diffusion_samples=N)`.

#### OpenFold2 / AlphaFold2

```python
model(feed_dict, recycling_steps=None)
```

If `recycling_steps` is omitted, the recycle count is the last axis of
`aatype` (sized `max_recycling_iters + 1` by the feature factory). Pass
`runtime_args={"recycling_steps": N}` to cap it. Do not pass Boltz sampling
keys to OpenFold2.

### CUDA graphs (Boltz-1/2, OpenFold3, Protenix)

On Boltz-1/2, OpenFold3, and Protenix (`protenix-v2`) the diffusion
**module** (including the token transformer) runs once per sampling step
with a fixed shape. Capturing a CUDA graph of that module and replaying it
removes per-kernel launch overhead — largest win on short sequences.
OpenFold2 / AlphaFold2 have no CUDA-graph module; the same
`accelerated_configs` entry is a no-op there. Protenix has no data pipeline;
enable graphs with [`optimize()`](#optimize-on-a-live-module) on the live
module.

Wire it through `engine_kwargs` (this is what the engine's `optimize()` call
consumes):

```python
from bionemo_ir.configs import AcceleratedConfig, BaseConfig
from bionemo_ir._torch.graph_optimization.config import (
    CUDAGraphOptimizationConfig,
    GraphOptimizationMode,
)

engine_kwargs = {
    "accelerated_configs": {
        "diffusion_module": AcceleratedConfig(
            backend="torch",
            default=BaseConfig(
                graph_optimization_config=CUDAGraphOptimizationConfig(
                    graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
                )
            ),
        ),
    }
}
```

The string form of the mode is `"cuda_graph_via_torch"`. The first few calls
for a given input shape run eager (kernel compile + allocator warmup); then
the graph is captured. A shape mismatch or capture failure falls back to
eager. `token_transformer` is nested inside `diffusion_module`; CUDA graphs
cannot nest, so requesting both keeps the parent and drops the child. See
[`optimize()`](#optimize-on-a-live-module).

### Outputs

The writer is the terminal stage (`update_row=False`). Each output row:

| Key                    | Type          | Meaning                                                                          |
| ---------------------- | ------------- | -------------------------------------------------------------------------------- |
| `output_path`          | `str \| None` | Path of the primary format                                                       |
| `output_paths`         | `str`         | JSON object mapping format → path, e.g. `'{"cif": "output/demo.cif"}'`           |
| `format`               | `str`         | Primary format                                                                   |
| `output_raw`           | `str \| None` | File contents of the primary format                                              |
| `scores`               | `str`         | JSON object. Always `json.loads(row["scores"])` before use                       |
| `__record_id`          | `str \| None` | Echo of the input id                                                             |
| `model_inference_time` | `float`       | Present when `profile_inference=True`                                            |
| `__inference_error__`  | `dict`        | `{error_msg, traceback}` when `should_continue_on_error=True` and the row failed |

`scores` always includes pLDDT / pTM / ipTM / PAE when the model produces
them. Boltz-2 adds extras such as `confidence_score`, `complex_plddt`,
`ligand_iptm`, `protein_iptm`, `pde`.

A sidecar `{id}_scores.json` is written next to the structure when
`output_path` is set.

### Errors

With the default `should_continue_on_error=False`, a failed forward raises
`FoldingPredictionError` from
`bionemo_ir.pipeline.stages.engine_stage`. The original exception is
`__cause__`.

```python
from bionemo_ir.pipeline.stages.engine_stage import FoldingPredictionError

try:
    outputs = processor(rows)
except FoldingPredictionError as exc:
    raise (exc.__cause__ or exc) from None
```

## Model constructor and `forward`

Use this at **inference** when you already have a feature dict (custom
dataloader, composing models) and want a plain `nn.Module`. This is not a
training API.

### Registry

```python
import bionemo_ir  # registers factories
from bionemo_ir.registry import (
    get_model_class,
    get_tokenizer,
    get_feature_factory,
    get_postprocessor,
    get_default_runtime_args,
    load_metadata,
)

ModelCls = get_model_class("boltz-2")
```

| Helper                                | Returns                                                           |
| ------------------------------------- | ----------------------------------------------------------------- |
| `get_model_class(name)`               | `type[nn.Module]`                                                 |
| `get_tokenizer(name)`                 | `TokenizerBase` spec (used by the processor, not called directly) |
| `get_feature_factory(name)`           | `FeatureFactoryBase` spec (same)                                  |
| `get_postprocessor(name)`             | `type[PostProcessorBase]`                                         |
| `get_default_runtime_args(name)`      | `dict`                                                            |
| `load_metadata(name, cache_dir=None)` | `{ccd_path, mol_dir, …}` or `{}`                                  |

Unknown names raise `ValueError` listing registered keys.

### Constructing a model

Import the class (`from bionemo_ir.models.boltz2 import Boltz2`) or
get it from [`get_model_class`](#registry): `get_model_class("boltz-2")` is
`Boltz2`. Then construct it.

All folding classes accept keyword arguments `config`, `model_name`, and
`include_load_weights` (OpenFold3 also accepts `diffusion_samples`). **Pass
`model_name=` explicitly** for AlphaFold2 / OpenFold2 variants:
`OpenFold2()` defaults to `openfold2_ptm_1`, not to the key you looked up.

```python
import os
from bionemo_ir.models.boltz2 import Boltz2
from bionemo_ir.models.openfold2 import OpenFold2
from bionemo_ir.models.openfold3 import OpenFold3
from bionemo_ir.models.protenix import Protenix

os.environ["ALPHAFOLD2_1_CKPT"] = "/checkpoints/alphafold2_1.pt"
af2 = OpenFold2(model_name="alphafold2_1").cuda().eval()

os.environ["ALPHAFOLD2_MULTIMER_1_CKPT"] = "/checkpoints/alphafold2_multimer_1.pt"
af2m = OpenFold2(model_name="alphafold2_multimer_1").cuda().eval()

b2 = Boltz2(model_name="boltz-2").cuda().eval()

of3 = OpenFold3(model_name="openfold3").cuda().eval()

# Protenix is not in the registry. include_load_weights defaults to False.
px = Protenix(model_name="protenix-v2", include_load_weights=True).cuda().eval()
```

`from bionemo_ir.models.boltz1 import Boltz1` follows the same pattern
as `Boltz2`.

`include_load_weights=True` (default on Boltz / OpenFold2 / OpenFold3) builds
from `ModelCls.get_pretrained_config(model_name)` and loads weights via the
hub resolver. Pass `include_load_weights=False` for an empty module you will
load yourself (`model.load_weights(state_dict)`). On `Protenix` the default
is `False`; pass `True` to load hub weights.

Pass `config=` to override dtypes, attention backends, recycle counts, and
similar. Default triangle / pairwise backends:
[support matrix — fused kernels](support-matrix.md#fused-kernels).

### Calling `forward`

```python
from bionemo_ir.registry import get_default_runtime_args, get_postprocessor

runtime_args = get_default_runtime_args("boltz-2")
# feats: dict[str, Tensor] already on CUDA, batch dim present
with torch.inference_mode():
    raw = model(feats, **runtime_args)

folding_output = get_postprocessor("boltz-2")()(feats, raw)
```

Post-processor signature is `__call__(batch, raw_output) → FoldingOutput`,
not `(raw, request, output_dir=...)`.

### `FoldingOutput`

[`FoldingOutput`][foldingoutput] (`bionemo_ir.data.schemas`) is a
`dict` the post-processor returns. Access fields as
`folding_output["atom_positions"]`. Coordinates use the 37-atom protein
layout the PDB/CIF writers expect. Confidence keys are `None` when the
model does not produce them.

| Field             | Shape                         | Required | Meaning                                                           |
| ----------------- | ----------------------------- | -------- | ----------------------------------------------------------------- |
| `atom_positions`  | `(num_res, num_atom_type, 3)` | yes      | Cartesian coordinates (Å)                                         |
| `residue_types`   | `(num_res,)`                  | yes      | Residue type as int (0–20, 20 = X)                                |
| `atom_mask`       | `(num_res, num_atom_type)`    | yes      | 1.0 if the atom is present                                        |
| `residue_indices` | `(num_res,)`                  | yes      | PDB residue numbers                                               |
| `b_factors`       | `(num_res, num_atom_type)`    | no       | Temperature factors                                               |
| `chain_indices`   | `(num_res,)`                  | no       | Chain index (multimer)                                            |
| `plddt`           | `(num_res,)`                  | no       | Per-residue confidence, 0–100                                     |
| `ptm`             | scalar                        | no       | Predicted TM-score, 0–1                                           |
| `iptm`            | scalar                        | no       | Interface pTM, 0–1 (multimer)                                     |
| `pae`             | `(num_res, num_res)`          | no       | Predicted aligned error (Å)                                       |
| `max_pae`         | scalar                        | no       | PAE cap used for normalization                                    |
| `residue_names`   | `(num_res,)` list of `str`    | no       | CCD/PDB codes (`"ALA"`, `"SAH"`, `"DA"`). Needed for ligands / NA |
| `mol_types`       | `(num_res,)`                  | no       | 0 = protein, 1 = RNA, 2 = DNA, 3 = ligand                         |

`get_scores()` returns JSON-able `plddt` / `ptm` / `iptm` / `pae` /
`max_pae` (the writer’s `scores` payload). Boltz-2 also stores extras such
as `confidence_score` and `complex_plddt` as additional dict keys; they
are not constructor arguments.

To write a file from a `FoldingOutput` without the processor:

```python
from bionemo_ir.data.utils import get_all_atom_types, get_all_residue_types
from bionemo_ir.data.writers import CIFWriter

res_types = get_all_residue_types("boltz-2")
atom_types = get_all_atom_types("boltz-2")
writer = CIFWriter(
    res_type_mapping=dict(enumerate(res_types)),
    atom_type_mapping=dict(enumerate(atom_types)),
    output_path="output/demo.cif",
)
writer.write(folding_output)
```

### `optimize()` on a live module

Same CUDA-graph config as in the processor, applied yourself:

```python
from bionemo_ir.configs import AcceleratedConfig, BaseConfig
from bionemo_ir.models.boltz2 import Boltz2
from bionemo_ir._torch.graph_optimization.config import (
    CUDAGraphOptimizationConfig,
    GraphOptimizationMode,
)

model = Boltz2(model_name="boltz-2").cuda().eval()
model.optimize({
    "diffusion_module": AcceleratedConfig(
        backend="torch",
        default=BaseConfig(
            graph_optimization_config=CUDAGraphOptimizationConfig(
                graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
            )
        ),
    ),
})
```

`optimize` mutates the module in place and returns `self`. Unknown module
names are warned and skipped. OpenFold2 has no graph-optimization modules, so
this is a no-op.

`token_transformer` lives inside `diffusion_module`. CUDA graphs cannot be
nested: if both are requested, `optimize()` keeps the parent and skips the
child (`Module 'token_transformer' is nested inside another requested
module`). Graph `token_transformer` alone if you only want that submodule
captured. Unrelated modules (for example OpenFold3 `structure_pairformer`)
are not nested and can be requested together.

## Custom architectures

If you already have a trained PyTorch model and want BioIR's optimized
Pairformer, diffusion transformer, or Evoformer in place of your module — not
the full folding pipeline — construct the layer, remap weights, and swap it
in. That path does not use `build_processor`.

The playbook is the **module-onboard** skill:
[`.agents/skills/module-onboard/SKILL.md`](../../.agents/skills/module-onboard/SKILL.md).
Worked RF3 conversions (config, adapter, weight remap, swap) live under
[`samples/`](../../.agents/skills/module-onboard/samples/).

The same custom-module path can take the **pairwise memory optimizations**
already used in BioIR (Boltz, OpenFold, Protenix): bf16 pair tensors,
shorter `[N,N,*]` lifetimes, never-materialize, and row-chunking. The
playbook is **scan-mem-opt-patterns**:
[`.agents/skills/scan-mem-opt-patterns/SKILL.md`](../../.agents/skills/scan-mem-opt-patterns/SKILL.md).
Use it when the swapped layer still OOMs at large `N` or
`diffusion_samples > 1`.

Layers (under `bionemo_ir._torch.layers.transformers`):

| Layer                                                         | Typical source module       |
| ------------------------------------------------------------- | --------------------------- |
| `PairformerModule`                                            | Pairformer / recycler stack |
| `BoltzDiffusionTransformer` / `OpenFold3DiffusionTransformer` | Diffusion token transformer |
| `EvoformerStack`                                              | Evoformer                   |

A typical conversion:

1. Map your hyperparameters onto the matching BioIR `*Config`
   (`PairformerConfig`, `DiffusionTransformerConfig`, `EvoformerStackConfig`
   from `bionemo_ir.configs`).
2. Remap `state_dict` keys into the BioIR layout (QKV / KV fusion, AdaLN
   gain+bias fusion, gate+input fusion, name renames such as
   `tri_mul_outgoing → tri_mul_out`).
3. Write a thin `nn.Module` adapter if signatures differ (mask polarity,
   extra sample/batch axes, `bool` vs `float` valid-masks).
4. Replace the original submodule on a live model.
5. Compare block-level then stack-level numerics against the original.
6. Optionally call `model.optimize(...)` for CUDA graphs on modules that
   declare graph optimization.

Fused kernels on supported SKUs:
[support matrix — fused kernels](support-matrix.md#fused-kernels).

## See also

- Config architecture (model tree vs pipeline stages):
  [`config.md`](config.md)
- Support matrix (models, GPUs, fused kernels):
  [`support-matrix.md`](support-matrix.md)
- Demo CLI: [`examples/folding/run_demo.py`](../../examples/folding/run_demo.py)
- Sample JSON / MSA: [`examples/data/samples/`](../../examples/data/samples)
- Module onboarding (swap Pairformer / DiT / Evoformer into *your* model):
  [`.agents/skills/module-onboard/SKILL.md`](../../.agents/skills/module-onboard/SKILL.md)
- Pairwise memory optimizations (port BioIR patterns onto *your* module):
  [`.agents/skills/scan-mem-opt-patterns/SKILL.md`](../../.agents/skills/scan-mem-opt-patterns/SKILL.md)

[inputrequest]: ../../bionemo_ir/data/schemas/basic.py
[polymer]: ../../bionemo_ir/data/schemas/basic.py
[msarecord]: ../../bionemo_ir/data/schemas/basic.py
[template]: ../../bionemo_ir/data/schemas/basic.py
[foldingoutput]: ../../bionemo_ir/data/schemas/basic.py
