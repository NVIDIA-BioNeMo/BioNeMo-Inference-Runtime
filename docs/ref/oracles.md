---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# Custom structure oracles

BioIR exposes a dependency-light mechanism for calling a user-supplied
structure oracle. An oracle provider is independent of beam search: callers
can use the same provider for search rewards, best-of-N selection, post-hoc
scoring, benchmarks, or dataset curation.

BioIR does not yet ship a pipeline-backed oracle provider. Users can adapt an
existing BioIR folding pipeline to the `OracleProvider` contract; a built-in
integration is planned for future work. Custom providers own their model,
service, container, files, retries, and resource allocation. No OSS oracle
code is imported or modified.

## Architecture

The public mechanism has four parts:

- `OracleProvider` is the provider base class and lifecycle contract.
- `OracleRequest` and `OracleResponse` are the batch boundary.
- residue and atom resolvers translate source-model integer vocabularies into
  canonical identities without guessing from vocabulary size;
- adapters translate a model output into requests and an oracle response into
  a consumer contract.

Oracle data contracts live beside BioIR's existing molecular schemas. Runtime
providers and adapters remain separate from `_torch`, model implementations,
and the data pipeline:

```text
bionemo_ir/data/schemas/
  basic.py
  oracle.py
bionemo_ir/oracles/
  resolvers.py
  providers/
    base.py
    callable.py
  adapters/
    rfproteina.py
    search.py
```

Import schemas from `bionemo_ir.data.schemas`. `bionemo_ir.oracles` also
re-exports them for compatibility alongside the resolvers and
custom-callable provider. Import adapters explicitly from
`bionemo_ir.oracles.adapters`.

## Provider contract

Subclass `OracleProvider` and implement `name` and `run`:

```python
from collections.abc import Sequence

from bionemo_ir.data.schemas import (
    OracleRequest,
    OracleResponse,
    OracleStatus,
)
from bionemo_ir.oracles import OracleProvider


class MyOracleProvider(OracleProvider):
    @property
    def name(self) -> str:
        return "my_structure_oracle"

    def run(
        self,
        requests: Sequence[OracleRequest],
    ) -> Sequence[OracleResponse]:
        responses = []
        for request in requests:
            try:
                result = my_oracle_client.fold(
                    sequence=request.structure.sequence,
                    request_id=request.request_id,
                )
            except TimeoutError as error:
                responses.append(
                    OracleResponse(
                        request_id=request.request_id,
                        status=OracleStatus.TIMEOUT,
                        error=str(error),
                    )
                )
                continue

            responses.append(
                OracleResponse(
                    request_id=request.request_id,
                    status=OracleStatus.OK,
                    metrics={
                        "my_oracle.plddt": result.plddt,
                        "my_oracle.ptm": result.ptm,
                    },
                    output=result,
                )
            )
        return responses
```

Consumers call `provider.evaluate(requests)`, not `run` directly.
`evaluate` requires unique request IDs, requires exactly one response per
request, rejects unexpected IDs, and restores request order. A provider may
return responses in any order.

`setup()` and `teardown()` are optional lifecycle hooks. The provider is also
a context manager:

```python
with MyOracleProvider() as provider:
    responses = provider.evaluate(requests)
```

For a function that already accepts and returns batches, use
`CallableOracleProvider`:

```python
from bionemo_ir.oracles import CallableOracleProvider

provider = CallableOracleProvider("my_structure_oracle", my_oracle_function)
```

The function has the same batch contract as `run`.

Search may split candidates according to `scoring_chunk_size`. A provider used
for search should therefore make each response depend on its request, not on
batch order or batch size.

## Request and response data

`OracleRequest.structure` contains both the source encoding and resolved
identities:

- `encoded.coordinates`: `[tokens, atom_slots, 3]` NumPy array in ångströms;
- `encoded.atom_mask`: `[tokens, atom_slots]`;
- `encoded.residue_type_indices`: the source residue IDs;
- `encoded.atom_type_indices`: the source atom IDs;
- `encoded.source_token_indices`: positions in the original model output;
- optional token-type and chain indices;
- `residues`: one `ResidueIdentity` per included token;
- `atoms`: one `AtomIdentity` for every occupied atom slot.

Resolved identities reuse the shared `ResType`, `AtomType`, and
`PolymerType` schemas from `bionemo_ir.data.schemas`. `component_id` and
`atom_name` are derived from those canonical types instead of being a second,
oracle-specific vocabulary.

`request.structure.sequence` constructs a one-letter sequence. It raises when
any resolved residue lacks `one_letter_code`; providers supporting ligands or
other non-sequence entities should consume the resolved records directly.

`OracleResponse.metrics` contains scalar metric names and values. Search uses
only metrics from `OracleStatus.OK` responses. `input_request` optionally
carries the shared `InputRequest` schema on a request, and `folding_output`
optionally carries the shared `FoldingOutput` schema on a response. These
fields are the typed bridge for a future BioIR pipeline-backed provider.
`output` remains available for a custom provider-native prediction, while
`artifacts` maps names to paths or URIs for post-hoc consumers.

Providers must return an explicit status for every request:

- `OK`: metrics may participate in a reward;
- `UNSUPPORTED`: the provider or resolver cannot represent the input;
- `FAILED`: evaluation failed;
- `TIMEOUT`: evaluation exceeded its time limit;
- `SKIPPED`: an external policy deliberately skipped it.

## Residue and atom resolvers

Source integer vocabularies are not inferred from their width. In particular,
RFProteína's 20-class and unified 33-class outputs have different index
origins, and a custom checkpoint may use another ordering.

Every RFProteína request adapter therefore requires both a
`ResidueTypeResolver` and `AtomTypeResolver`:

```python
from bionemo_ir.data.schemas import (
    AtomIdentity,
    AtomTypes,
    PolymerType,
    ResidueIdentity,
    ResTypes,
)
from bionemo_ir.oracles import MappingAtomTypeResolver, MappingResidueTypeResolver

residue_resolver = MappingResidueTypeResolver(
    {
        1: ResidueIdentity(1, ResTypes.A, "A", PolymerType.PROTEIN),
        2: ResidueIdentity(2, ResTypes.R, "R", PolymerType.PROTEIN),
        # Supply the complete vocabulary used by this model.
    }
)
atom_resolver = MappingAtomTypeResolver(
    {
        0: AtomIdentity(0, AtomTypes.N, "N"),
        1: AtomIdentity(1, AtomTypes.CA, "C"),
        # Supply every occupied source atom index.
    }
)
```

Mapping resolvers support more specific keys:

- `(token_type_index, residue_type_index)` overrides a plain residue index;
- `(component_id, atom_type_index)` overrides a plain atom index.

For contextual mappings, use `CallableResidueTypeResolver` and
`CallableAtomTypeResolver`, or implement the resolver protocols directly.
Resolvers receive the complete `EncodedStructure`, token position, and, for
atoms, atom slot and resolved residue. This is required for atomized ligands:
all ligand tokens may occupy atom slot 1 while their identities come from
metadata such as `element`, `charge`, or `molecule_iid`.

A resolver should raise `OracleResolutionError` for an unsupported source
index. The RFProteína adapter converts that row into an `UNSUPPORTED`
response and does not call the provider for it. Other exceptions propagate as
implementation errors.

## RFProteína search

Compose a provider, source adapter, and search metric adapter:

```python
from bionemo_ir.models.rfproteina import (
    RFProteinaBeamSearchConfig,
    RFProteinaLocalMetricEvaluator,
    RFProteinaSearchRewardConfig,
)
from bionemo_ir.oracles.adapters import (
    OracleMetricEvaluator,
    RFProteinaOracleRequestAdapter,
)

request_adapter = RFProteinaOracleRequestAdapter(
    residue_type_resolver=residue_resolver,
    atom_type_resolver=atom_resolver,
)
oracle_metrics = OracleMetricEvaluator(
    provider,
    request_adapter,
    required_tags_any={"refolding"},
)

search_config = RFProteinaBeamSearchConfig(
    step_checkpoints=[0, 25, 50, 75, 100],
    n_branch=4,
    beam_width=4,
    rewards=RFProteinaSearchRewardConfig(
        weights={"clash_metrics.frac_atoms_clashing_1p5A": -1.0},
        reward_metrics_tags=["local"],
        checkpoint_overrides={
            100: {
                "weights": {"my_oracle.plddt": 1.0},
                "reward_metrics_tags": ["refolding"],
            },
        },
    ),
)

with provider:
    result = model.search(
        static_features,
        search_config,
        evaluators=[
            RFProteinaLocalMetricEvaluator(),
            oracle_metrics,
        ],
        seed=1234,
    )
```

This example uses inexpensive local metrics at intermediate checkpoints and
runs the custom oracle only at step 100. The generic composite evaluator
merges each evaluator's disjoint metric keys row by row. Passing an explicit
`evaluators` list replaces the default local evaluator, so this example
includes `RFProteinaLocalMetricEvaluator` in the list. `reward_metrics_tags`
are delivered through `MetricEvaluationContext`; the search adapter checks tag
conditions before it builds requests or calls the provider.

The RFProteína adapter:

- uses the prediction's final coordinate, residue, and atom-mask views;
- removes padding tokens;
- converts coordinates from nanometres to ångströms;
- provides stable candidate IDs from the search context;
- preserves selected static features as CPU NumPy arrays in
  `EncodedStructure.metadata`.

By default it preserves identity and topology fields such as `element`,
`charge`, `chain_iid`, `molecule_iid`, and bond features. Pass
`metadata_keys=None` to preserve every static feature, or provide an explicit
key collection. Metadata remains aligned to the original source row; use
`encoded.source_token_indices` to address it.

Atom slots are the source atom indices by default. If the input carries a
different atom vocabulary, set `atom_type_indices_key` to a static feature
with shape `[batch, tokens, atom_slots]`.

A failed, timed-out, skipped, or unsupported response contributes no oracle
metric keys. If the active weighted objective requires one of those keys,
that candidate receives `-inf`. Search raises if every candidate for a
request receives `-inf`.

## Use outside search

Search is only a consumer adapter. Other tasks can build `OracleRequest`
objects and call the same provider directly:

```python
responses = provider.evaluate(requests)
successful = [
    response
    for response in responses
    if response.status is OracleStatus.OK
]
```

Caching, retries, structure-comparison metrics, and a provider backed by the
BioIR folding pipeline are not part of the current mechanism. A future
pipeline adapter can populate `OracleRequest.input_request` and return its
`FoldingOutput` as `OracleResponse.folding_output` without adding another
molecular schema.

## Approximate options

Approximate speed options, such as Protenix [frame-aware
prediction reuse](sampling.md#frame-aware-prediction-reuse-protenix-opt-in),
are judged on paired lDDT and DockQ against the exact rollout of the same
input and seed, not on bit equality.
