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

# File Templates for BioIR Data Pipeline

Replace `<model>` with the model name (e.g., `boltz1`, `openfold3`, `protenix`).

Templates are shown for **Pattern A** (OpenFold-style, flat tensor dict) unless
noted. Where **Pattern B** (Boltz2-style, mixed context row) differs, an
alternative is shown.

Templates carry no SPDX license header — the `insert-license` hook adds it
from [`.license-header.txt`][hdr] when the file is first committed, so a
hand-copied header only risks drifting from it.

## 1. `__init__.py`

```python
# Empty file
```

## 2. `const.py`

Copy domain-specific constants from the OSS code. These are typically:

- Residue/atom type mappings
- Standard masks and lookup tables
- Physical/chemical constants

```python
# Copy or import constants from OSS residue_constants.py / chemical.py
# Examples:
# - restypes, restype_order, restype_1to3, restype_3to1
# - atom_types, atom_order, atom_type_num
# - STANDARD_ATOM_MASK
# - HHBLITS_AA_TO_ID, MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
# - chi_angles_atoms, chi_angles_mask
# - rigid_group_atom_positions
```

## 3. `common.py`

Shared helper functions used by multiple files.

```python
import torch
import numpy as np

# Port from OSS: one-hot, torsion angles, pseudo-beta, utility math
# Example:

def make_one_hot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    x_one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return x_one_hot


def pseudo_beta_fn(
    aatype: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Port from OSS: compute CB position (CA for glycine)
    ...
```

## 4. `feature_context.py`

The context generator creates the initial numpy feature dict from parsed input.

```python
from typing import Optional

import numpy as np
import torch

import bionemo_ir.pipeline.models.<model>.const as rc
from bionemo_ir.configs.base import BaseConfig
from bionemo_ir.data.parsers import InputParsed, MSAParsed, generate_deletion_matrix
from bionemo_ir.data.utils import sequence_to_onehot
from bionemo_ir.pipeline.base import ContextGeneratorBase


class FeatureContextGenerator(ContextGeneratorBase):

    def __init__(self, config: BaseConfig):
        super().__init__(config)
        # Define which features to extract at each stage
        self.unsupervised_features = [
            "aatype", "residue_index", "msa", "num_alignments",
            "seq_length", "between_segment_residues", "deletion_matrix",
            # ... model-specific features
        ]
        self.template_features = [
            "template_all_atom_positions", "template_sum_probs",
            "template_aatype", "template_all_atom_mask",
            # ... model-specific template features
        ]

    def make_sequence_features(self, sequence: str,
                               description: str) -> dict[str, np.ndarray]:
        """Port from OSS: data_pipeline.make_sequence_features()"""
        # Convert sequence to aatype, create residue_index, seq_length, etc.
        ...

    def make_msa_features(self, parsed_msa: MSAParsed) -> dict[str, np.ndarray]:
        """Port from OSS: data_pipeline.make_msa_features()"""
        # Convert MSA to int arrays, build deletion matrix
        ...

    def np_to_tensor_dict(self, np_example: dict[str, np.ndarray],
                          features: list[str]) -> dict[str, torch.Tensor]:
        """Convert numpy dict to torch tensor dict, filtering to requested features."""
        def to_tensor(t):
            if isinstance(t, torch.Tensor):
                return t.clone().detach()
            return torch.tensor(t)
        return {k: to_tensor(v) for k, v in np_example.items() if k in features}

    def __call__(self, parsed: InputParsed) -> dict[str, torch.Tensor]:
        """Main entry: parsed input → context tensors."""
        # 1. Build raw numpy features from parsed input
        # 2. Convert to tensors, filtering to relevant feature set
        # 3. Return context dict
        ...
```

### Pattern B variant: `feature_context.py`

For models with multi-step featurization and non-tensor intermediate data:

```python
from typing import Any, Optional

import numpy as np

import bionemo_ir.pipeline.models.<model>.const as rc
from bionemo_ir.configs.base import BaseConfig
from bionemo_ir.data.parsers import InputParsed
from bionemo_ir.pipeline.base import ContextGeneratorBase


class ModelContextGenerator(ContextGeneratorBase):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

    def __call__(self, parsed: InputParsed) -> dict[str, Any]:
        """Returns a context row with mixed tensor/non-tensor data.

        The row is stored as context["_row"] in the feature stage.
        Feature generators read from it to produce tensors incrementally.
        """
        row = {}
        # 1. Build structures, tokenize sequences
        # 2. Load molecules, parse MSA per chain
        # 3. Store as mix of tensors and non-tensor objects
        # row["structure"] = ...     # non-tensor
        # row["tokens"] = ...        # tensor
        # row["molecules"] = ...     # non-tensor dict
        # row["msa_parsed"] = ...    # non-tensor list
        return row
```

## 5. `transforms.py`

Non-ensembled transforms that modify the tensor dict.

```python
from typing import Optional

import torch

import bionemo_ir.pipeline.models.<model>.const as rc
from bionemo_ir.configs.base import BaseConfig
from bionemo_ir.pipeline.base import TransformBase


class CastTo64BitInts(TransformBase):
    """Port from OSS: cast_to_64bit_ints"""

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for k, v in batch.items():
            if v.dtype == torch.int32:
                batch[k] = v.type(torch.int64)
        return batch


class ExampleConditionalTransform(TransformBase):
    """Example of a conditional transform."""

    def is_enabled(self) -> bool:
        return self.config.some_flag  # Only runs when flag is True

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Modify batch in-place
        ...
        return batch


class ExampleParameterizedTransform(TransformBase):
    """Example of a transform with constructor parameters (from OSS curried function)."""

    def __init__(self, config: Optional[BaseConfig] = None, some_param: float = 0.0):
        super().__init__(config)
        self.some_param = some_param

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Use self.some_param
        ...
        return batch
```

## 6. `tokenizer.py`

Wires context generators and transforms into a declarative pipeline.

```python
from collections import OrderedDict
from typing import Callable

from bionemo_ir.pipeline.base import (
    ContextGeneratorSpec, TokenizerBase, TransformSpec, dict_context_merger,
)
from .feature_context import FeatureContextGenerator
from .transforms import CastTo64BitInts, OtherTransform  # import your transforms


class Tokenizer(TokenizerBase):
    context_generator_specs: OrderedDict[str, ContextGeneratorSpec] = OrderedDict({
        'primary': ContextGeneratorSpec(
            name='primary',
            generator=FeatureContextGenerator,
            required_kwargs=['parsed']
        )
    })

    context_merger_func: Callable = dict_context_merger

    transform_specs: list[TransformSpec] = [
        TransformSpec(name='cast_to_64_bit_ints', transform=CastTo64BitInts),
        # Add all non-ensembled transforms in OSS order
        # TransformSpec(name='...', transform=..., kwargs={...}),
    ]


# Optional: multimer variant with different transforms
class MultimerTokenizer(TokenizerBase):
    context_generator_specs: OrderedDict[str, ContextGeneratorSpec] = OrderedDict({
        'primary': ContextGeneratorSpec(
            name='primary',
            generator=FeatureContextGenerator,
            required_kwargs=['parsed']
        )
    })

    context_merger_func: Callable = dict_context_merger

    transform_specs: list[TransformSpec] = [
        TransformSpec(name='cast_to_64_bit_ints', transform=CastTo64BitInts),
    ]
```

## 7. `feature_generators.py`

Classes that produce NEW feature tensors from the batch.

**Pattern A:** generators read from `batch` (tensor dict) and optionally
`context` (seeds, metadata). **Pattern B:** generators also read
`context["_row"]` to access non-tensor data from the context generator.

```python
from typing import Any, Optional

import torch

import bionemo_ir.pipeline.models.<model>.const as rc
from bionemo_ir.configs.base import BaseConfig
from bionemo_ir.pipeline.base import FeatureGeneratorBase
from .common import some_helper  # import model-specific helpers


class MakeSequenceMask(FeatureGeneratorBase):
    """Port from OSS: make_seq_mask (Pattern A example)"""

    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        feats = {}
        feats["seq_mask"] = torch.ones(batch["aatype"].shape, dtype=torch.float32)
        return feats


class ConditionalGenerator(FeatureGeneratorBase):
    """Example: only enabled when config flag is set."""

    def is_enabled(self) -> bool:
        return self.config.enable_feature_x

    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        feats = {}
        feats["feature_x"] = ...  # compute from batch
        return feats


class ParameterizedGenerator(FeatureGeneratorBase):
    """Example: takes constructor params from kwargs in spec."""

    def __init__(self, config: Optional[BaseConfig] = None, prefix: str = ""):
        super().__init__(config)
        self.prefix = prefix

    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        # Use self.prefix to read/write prefixed keys
        ...


class ContextRowGenerator(FeatureGeneratorBase):
    """Pattern B example: reads non-tensor data from context row."""

    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        row = context["_row"]  # Mixed tensor/non-tensor data from context generator
        feats = {}
        # Convert non-tensor row data into tensors
        # feats["atom_coords"] = torch.tensor(row["structure"].get_coords())
        return feats
```

## 8. `feature_collators.py`

Classes that modify/sample/crop/pad the feature dict. Run per ensemble
iteration.

```python
from typing import Any, Optional

import torch

from bionemo_ir.configs.base import BaseConfig
from bionemo_ir.pipeline.base import FeatureCollatorBase
from .common import some_helper


MSA_FEATURE_NAMES = ["msa", "deletion_matrix", "msa_mask", "msa_row_mask", "bert_mask", "true_msa"]


class SampleMsa(FeatureCollatorBase):
    """Port from OSS: sample_msa — subsample MSA sequences."""

    def __init__(self, config: Optional[BaseConfig] = None, keep_extra: bool = True):
        super().__init__(config)
        self.keep_extra = keep_extra

    def __call__(self, features: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        seed = None
        if not self.config.resample_msa_in_recycling:
            seed = context.get("ensemble_seed", None)
        max_seq = self.config.max_msa_clusters
        num_seq = features["msa"].shape[0]

        g = None
        if seed is not None:
            g = torch.Generator(device=features["msa"].device)
            g.manual_seed(seed)

        shuffled = torch.randperm(num_seq - 1, generator=g) + 1
        index_order = torch.cat((torch.tensor([0], device=shuffled.device), shuffled), dim=0)
        num_sel = min(max_seq, num_seq)
        sel_seq, not_sel_seq = torch.split(index_order, [num_sel, num_seq - num_sel])

        for k in MSA_FEATURE_NAMES:
            if k in features:
                if self.keep_extra:
                    features["extra_" + k] = torch.index_select(features[k], 0, not_sel_seq)
                features[k] = torch.index_select(features[k], 0, sel_seq)

        return features


class SelectFeat(FeatureCollatorBase):
    """Filter feature dict to only include specified keys."""

    def __init__(self, config: Optional[BaseConfig] = None,
                 include_feats: list[str] = None):
        super().__init__(config)
        self.include_feats = include_feats

    def __call__(self, features: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        remove_keys = {k for k in features if k not in self.include_feats}
        for key in remove_keys:
            del features[key]
        return features


class MakeFixedSize(FeatureCollatorBase):
    """Pad features to fixed sizes for batching."""

    def __init__(self, config: Optional[BaseConfig] = None):
        super().__init__(config)
        # Define shape schema: which dims are dynamic
        # None = keep original size, named = pad to config value
        self._shape_schema = {
            # "feature_name": [dim_spec, dim_spec, ...]
        }
        self._pad_size_map = {
            # "dim_name": config.max_value
        }

    def __call__(self, features: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        # Pad each feature according to schema
        ...
        return features
```

## 9. `feature_factory.py`

Wires generators and collators into the complete feature pipeline.

```python
import random
from typing import Any, Callable

import numpy as np
import torch

from bionemo_ir.pipeline.base import (
    FeatureCollatorSpec, FeatureFactoryBase, FeatureGeneratorSpec,
    default_context_and_feature_merger,
)
from .feature_collators import (SampleMsa, MakeFixedSize, SelectFeat, ...)
from .feature_generators import (MakeSequenceMask, MakeMsaMask, ...)

# Import SampleRepeater from openfold2 or reimplement
from bionemo_ir.pipeline.models.openfold2.feature_factory import SampleRepeater

_FEATURE_KEYS = [
    # List all feature keys the model expects as input
    "aatype", "residue_index", "msa_feat", "target_feat", ...
]


def pre_init(context: dict[str, Any]) -> dict[str, Any]:
    """Setup random seeds for reproducible feature generation."""
    random_seed = context.get("random_seed", 0)
    if random_seed is None:
        random_seed = random.randrange(2**32)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)
    context["ensemble_seed"] = random.randint(0, torch.iinfo(torch.int32).max)
    return context


def create_ensemble_feature_collator() -> list[FeatureCollatorSpec]:
    """Build the ensembled (per-recycling-iter) collator pipeline."""
    feature_collator_specs = [
        FeatureCollatorSpec(name="sample_msa", functor=SampleMsa, kwargs={}),
        # ... add collators matching OSS ensembled_transform_fns order
        FeatureCollatorSpec(name="select_feat", functor=SelectFeat,
                            kwargs={"include_feats": _FEATURE_KEYS}),
        FeatureCollatorSpec(name="make_fixed_size", functor=MakeFixedSize, kwargs={}),
    ]
    return [
        FeatureCollatorSpec(
            name="repeater",
            functor=SampleRepeater,
            kwargs={
                "feature_collator_specs": feature_collator_specs,
                "get_n_iters": lambda config: config.max_recycling_iters + 1,
            }
        )
    ]


class FeatureFactory(FeatureFactoryBase):
    pre_init: Callable = pre_init
    feature_generator_specs: list[FeatureGeneratorSpec] = [
        # Add generators matching OSS nonensembled generators order
        FeatureGeneratorSpec(name="make_sequence_mask", functor=MakeSequenceMask, kwargs={}),
        FeatureGeneratorSpec(name="make_msa_mask", functor=MakeMsaMask, kwargs={}),
        # ...
    ]
    features_merger_func: Callable = default_context_and_feature_merger
    feature_collator_specs: list[FeatureCollatorSpec] = create_ensemble_feature_collator()
```

## 10. `postprocessor.py`

Converts model output to structured result.

```python
from typing import Any, Optional

import numpy as np
import torch
from pydantic import BaseModel

import bionemo_ir.pipeline.models.<model>.const as rc
from bionemo_ir.data.schemas import FoldingOutput
from bionemo_ir.pipeline.base import PostProcessorBase


class PostProcessorConfig(BaseModel):
    # Model-specific postprocessor settings
    subtract_plddt: bool = False


class PostProcessor(PostProcessorBase):

    def __init__(self, config: Optional[BaseModel] = None) -> None:
        super().__init__(config)
        if self.config is None:
            self.config = PostProcessorConfig()

    def __call__(self, batch: dict[str, Any],
                 output: dict[str, Any]) -> FoldingOutput:
        # 1. Extract relevant tensors from batch and output
        # 2. Compute confidence scores (pLDDT, pTM, iPTM, PAE)
        # 3. Extract atom positions and masks
        # 4. Determine chain indices
        # 5. Return FoldingOutput
        return FoldingOutput(
            residue_types=...,
            atom_positions=...,
            atom_mask=...,
            residue_indices=...,
            b_factors=...,
            chain_indices=...,
            plddt=...,
            ptm=...,
            iptm=...,
            pae=...,
            max_pae=...,
        )
```

## File Creation Order

Create files in dependency order:

1. `__init__.py` (empty)
1. `const.py` (no internal dependencies)
1. `common.py` (depends on const)
1. `feature_context.py` (depends on const, common, base)
1. `transforms.py` (depends on const, base)
1. `feature_generators.py` (depends on const, common, base)
1. `feature_collators.py` (depends on common, base)
1. `tokenizer.py` (depends on feature_context, transforms)
1. `feature_factory.py` (depends on feature_generators, feature_collators)
1. `postprocessor.py` (depends on const, base)

[hdr]: ../../../.license-header.txt
