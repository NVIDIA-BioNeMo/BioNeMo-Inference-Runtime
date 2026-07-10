<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TRT-BNM Pipeline Architecture Reference

## Directory Structure

Every model pipeline lives under `tensorrt_bionemo/pipeline/models/<model_name>/`:

```
tensorrt_bionemo/pipeline/models/<model_name>/
├── __init__.py                 # empty
├── const.py                    # Domain constants (residue types, atom types, etc.)
├── common.py                   # Shared math/helper functions
├── feature_context.py          # ContextGeneratorBase implementations
├── transforms.py               # TransformBase implementations (tokenizer transforms)
├── tokenizer.py                # TokenizerBase implementation
├── feature_generators.py       # FeatureGeneratorBase implementations
├── feature_collators.py        # FeatureCollatorBase implementations
├── feature_factory.py          # FeatureFactoryBase implementation
├── postprocessor.py            # PostProcessorBase implementation
├── msa_pairing.py              # (optional) MSA pairing for multimer
├── structure.py                # (optional) Structure manipulation (Pattern B)
├── tokenizer_logic.py          # (optional) Complex tokenization logic (Pattern B)
├── featurizer.py               # (optional) Additional featurization helpers
└── context_io.py               # (optional) Context save/load for caching
```

## Pipeline Patterns

### Pattern A — OpenFold-style (flat tensor dict)

Context generator produces a complete tensor dict. All downstream stages operate on flat `dict[str, torch.Tensor]`.

**Use when:** OSS has a single `make_features()` entry point; all intermediate data is tensors.

**Existing examples:** `openfold2/`, `boltz1/`

### Pattern B — Boltz2-style (mixed context row)

Context generator returns a row dict with **both tensor and non-tensor data** (structures, molecules, parsed MSAs). Feature generators read `context["_row"]` and produce tensors incrementally.

**Use when:** OSS has multi-step featurization with intermediate non-tensor state (structure objects, molecule dicts, per-chain MSA lists).

**Existing examples:** `boltz2/`

## Pipeline Data Flow

### Pattern A (OpenFold-style)

```
InputRequest
  → ParserStage (A3M/template parsing)
InputParsed {polymers: [PolymerParsed]}
  → TokenizerStage
    ContextGenerator(parsed) → dict[str, np.ndarray]
    → np_to_tensor_dict → dict[str, torch.Tensor]
    → TransformSpec[](batch) → context dict
  → FeatureGeneratorStage
    pre_init(context) → context with seeds
    → FeatureGeneratorSpec[](batch, context) → new feature dicts
    → features_merger(context, generated) → merged dict
    → FeatureCollatorSpec[](features, context) → final features
  → FoldingEngineStage (model inference)
  → WriterStage (PDB/CIF output)
```

### Pattern B (Boltz2-style)

```
InputRequest
  → ParserStage
InputParsed
  → TokenizerStage
    ContextGenerator(parsed) → row dict (mixed tensor/non-tensor)
    → TransformSpec[](batch) → context dict
  → FeatureGeneratorStage
    pre_init(context) → context with seeds, context["_row"] = row
    → FeatureGeneratorSpec[](batch, context) → new tensors (reads _row)
    → features_merger(context, generated) → merged dict
    → FeatureCollatorSpec[](features, context) → final features
  → FoldingEngineStage (model inference)
  → WriterStage (PDB/CIF output)
```

## Base Classes (from `tensorrt_bionemo/pipeline/base.py`)

### ContextGeneratorBase

Converts parsed input (numpy) into initial tensor dict.

```python
class ContextGeneratorBase(ABC):
    def __init__(self, config: Optional[BaseConfig] = None,
                 metadata: Optional[dict[str, Any]] = None):
        self.config = config
        self.metadata = metadata
        self._required_kwargs = []

    @abstractmethod
    def __call__(self) -> dict[str, torch.Tensor]:
        # Pattern A: returns dict[str, torch.Tensor]
        # Pattern B: returns dict[str, Any] (mixed tensor/non-tensor)
        return {}

    @property
    def required_kwargs(self) -> list[str]:
        return self._required_kwargs
```

### TransformBase

Modifies tensor dict in-place. Used in the tokenizer stage.

```python
class TransformBase(ABC):
    def __init__(self, config: Optional[BaseConfig] = None, **kwargs: Any):
        self.config = config

    @abstractmethod
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return batch

    def is_enabled(self) -> bool:
        return True
```

### FeatureGeneratorBase

Produces NEW feature tensors from existing batch + context. Returns only the new keys.

```python
class FeatureGeneratorBase(ABC):
    def __init__(self, config: Optional[BaseConfig] = None,
                 metadata: Optional[dict[str, Any]] = None, **kwargs: Any):
        self.config = config
        self._name = kwargs.get("name", self.__class__.__name__)
        self.metadata = metadata

    @abstractmethod
    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        return batch  # Should return NEW features dict

    def is_enabled(self) -> bool:
        return True
```

### FeatureCollatorBase

Modifies features dict in-place (sampling, cropping, padding). Inherits from FeatureGeneratorBase.

```python
class FeatureCollatorBase(FeatureGeneratorBase):
    pass  # Same signature as FeatureGeneratorBase
```

### TokenizerBase

Orchestrates: context generators → merger → transforms.

```python
class TokenizerBase(BaseModel):
    context_generator_specs: OrderedDict[str, ContextGeneratorSpec]
    context_merger_func: Callable
    transform_specs: list[TransformSpec]
```

### FeatureFactoryBase

Orchestrates: pre_init → generators → merger → collators.

```python
class FeatureFactoryBase(BaseModel):
    pre_init: Callable
    feature_generator_specs: list[FeatureGeneratorSpec]
    features_merger_func: Callable
    feature_collator_specs: list[FeatureCollatorSpec]
```

### PostProcessorBase

Converts model output to structured result.

```python
class PostProcessorBase:
    def __init__(self, config: Optional[BaseModel] = None, **kwargs: Any):
        self.config = config

    @abstractmethod
    def __call__(self, batch: dict[str, torch.Tensor],
                 output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return output
```

## Spec Types

```python
class ContextGeneratorSpec(BaseModel):
    name: str
    generator: Type[ContextGeneratorBase]
    required_kwargs: list[str]

class TransformSpec(BaseModel):
    name: Optional[str] = "transform"
    transform: Type[TransformBase]
    kwargs: Optional[dict[str, Any]] = {}

class FeatureGeneratorSpec(BaseModel):
    name: Optional[str] = "generator"
    functor: Type[FeatureGeneratorBase]
    kwargs: Optional[dict[str, Any]] = {}

class FeatureCollatorSpec(FeatureGeneratorSpec):
    pass
```

## Registry Pattern

```python
class ModelComponentsFactory(ABC):
    @classmethod
    def get_model_class(cls) -> Type[nn.Module]: ...
    @classmethod
    def get_tokenizer(cls) -> TokenizerBase: ...
    @classmethod
    def get_feature_factory(cls) -> FeatureFactoryBase: ...
    @classmethod
    def get_postprocessor(cls) -> Type[PostProcessorBase]: ...
    @classmethod
    def get_trt_building_modules(cls) -> Dict[str, Any]: ...
    @classmethod
    def get_supported_model_names(cls) -> list[str]: ...
```

## Config Pattern

Model-specific configs extend `BaseConfig`:

```python
from tensorrt_bionemo.configs.base import BaseConfig

class NewModelConfig(BaseConfig):
    is_multimer: bool = False
    enable_template: bool = True
    max_recycling_iters: int = 3
    max_msa_clusters: int = 512
    max_extra_msa: int = 1024
    max_templates: int = 4
    resample_msa_in_recycling: bool = True
    msa_cluster_features: bool = True
    # ... model-specific fields
```

All generators, collators, and transforms receive `config` in `__init__` and access it via `self.config`.

## SampleRepeater (Ensemble/Recycling)

The recycling loop is implemented by `SampleRepeater`, which wraps a list of collator specs and runs them N times, stacking results:

```python
class SampleRepeater(FeatureCollatorBase):
    def __init__(self, config, feature_collator_specs, get_n_iters, stack_dim=-1):
        self.n_iter = get_n_iters(config)
        self.feature_collators = [spec.functor(config=config, **spec.kwargs) for spec in feature_collator_specs]

    def __call__(self, batch, context):
        ensemble_batch = []
        for _ in range(self.n_iter):
            batch_i = batch.copy()
            for collator in self.feature_collators:
                if collator.is_enabled():
                    batch_i = collator(batch_i, context)
            ensemble_batch.append(batch_i)
        return {k: torch.stack([b[k] for b in ensemble_batch], dim=self.stack_dim) for k in ensemble_batch[0]}
```

## Helper Imports

```python
# Base classes
from tensorrt_bionemo.pipeline.base import (
    ContextGeneratorBase, ContextGeneratorSpec,
    TransformBase, TransformSpec,
    FeatureGeneratorBase, FeatureGeneratorSpec,
    FeatureCollatorBase, FeatureCollatorSpec,
    FeatureFactoryBase, TokenizerBase, PostProcessorBase,
    dict_context_merger, default_context_and_feature_merger,
)
from tensorrt_bionemo.configs.base import BaseConfig

# Data types
from tensorrt_bionemo.data.parsers import InputParsed, MSAParsed, generate_deletion_matrix
from tensorrt_bionemo.data.schemas import FoldingOutput
from tensorrt_bionemo.data.utils import sequence_to_onehot
```

## Conventions

1. **NVIDIA copyright header** on every file (Apache 2.0).
1. **`is_enabled()` for conditional logic** — never skip a spec from the list; disable it via `is_enabled()`.
1. **`context` dict carries seeds** — `ensemble_seed`, `random_seed` set in `pre_init()`.
1. **Feature keys are flat strings** — e.g., `"msa"`, `"template_aatype"`, `"extra_msa"`.
1. **Tensor device agnostic** — don't hardcode devices; use `device=batch["key"].device`.
1. **No global random state** — use `torch.Generator` seeded from context.
