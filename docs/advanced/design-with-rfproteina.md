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

# RFProteína numerical engine

BioIR runs RFProteína's encoder, velocity network, numerical integration and
decoder. The host application owns generation workflows, search, scoring,
sequence constraints, atom selection, motif assignment and structure output,
and runs the same application code on every backend. Both execution boundaries
are public:

- `RFProteina.partial_simulation(...)` advances an existing state with the
  optimized feature builder and integrator. A host's BioIR backend uses this
  API for generation, partial diffusion and search.
- `RFProteinaTrunk.forward_features(...)` runs the numerical trunk on
  caller-built representations. Use it to change feature preparation,
  integration, autoencoders or readouts independently of BioIR.

## Public API

```python
from bionemo_ir.models.rfproteina import RFProteina, SamplingContext

model = RFProteina.from_reference_state_dicts(
    flow_weights,
    ae_weights,
    simulation_config=simulation_config,
    reference_config=training_nn_config,
    device="cuda",
)
model.configure_graphs()

with SamplingContext.graph_safe(features["padding_mask"].device, seed=42) as rng:
    result = model.partial_simulation(
        features,
        state,
        start_step=0,
        end_step=20,
        ts=ts,
        gt=gt,
        recycle_depths=recycle_depths,
        context=rng,
    )
    resumed = model.partial_simulation(
        features,
        result.state,
        start_step=20,
        end_step=60,
        ts=ts,
        gt=gt,
        recycle_depths=recycle_depths,
        initial_clean=result.clean_prediction,
        context=rng,
    )

decoded = model.decode(resumed.state, features["padding_mask"], can_be_encoded=emit_mask)
```

`features` is the already-tensorized batch. `state` contains `bb_ca` [B, N, 3]
in nanometres and `local_latents` [B, N, latent_dim]. The caller initializes
that state and supplies one time grid `ts` and noise-rate array `gt` per
modality, plus explicit `recycle_depths`. Interval indices are absolute. The
same context continues the random stream across segments. Inputs are not
modified, so branches can share their starting tensors.

Pass the preceding clean prediction to continue self-conditioning. Set
`carry_self_condition=False` when the application intentionally restarts it.
For inverse folding, `target_backbone` supplies the clean backbone whose
forward marginal is resampled during integration.

`partial_simulation` integrates a supplied state, as the host's own simulation
method does. It does not initialize partial diffusion. For that workflow the
application takes the clean state from
`encode(features, generator=rng.generator)`, corrupts it, and chooses the
schedule, then calls this numerical operation. The full interval is the
default; `start_step` and `end_step` allow resuming.

BioIR has no `sample()` or `sample_partial()` workflow, schedule builder,
recycle strategy, or automatic sample-and-decode call. `model(...)` evaluates
the velocity network from prepared representations.

## Sampler settings

`RFProteinaSimulationConfig` contains numerical parameters only, and the host
supplies it: `from_reference_state_dicts` requires `simulation_config` and has
no fallback. The built-in defaults serve tests and toy examples. They are not
the released sampler settings (`sc_scale_noise`, `t_lim_ode` and `tsr_sigma`
differ, and `tsr_scaled_score_drift` is off), so a rollout that relies on them
follows a different trajectory. Modality and step-parameter names follow the
host's configuration (`bb_ca`, `local_latents`, `sc_scale_noise`, `t_lim_ode`,
`t_lim_ode_below`), so build the config from the host's own mappings:

```python
from bionemo_ir.models.rfproteina.config import (
    RFProteinaModalitySimulationConfig,
    RFProteinaReferenceConfig,
    RFProteinaSimulationConfig,
    RFProteinaVFStepConfig,
)

simulation_config = RFProteinaSimulationConfig(
    **{
        modality: RFProteinaModalitySimulationConfig(
            step=RFProteinaVFStepConfig(**step_params[modality]),
            reference=RFProteinaReferenceConfig(**reference_params[modality]),
        )
        for modality in ("bb_ca", "local_latents")
    }
)
```

`step_params[modality]` holds that modality's step parameters: `sampling_mode`,
`sc_scale_noise`, `t_lim_ode`, `t_lim_ode_below`, `tsr_k`, `tsr_sigma` and, for
the stochastic VF-TSR update, `tsr_scaled_score_drift`.
`reference_params[modality]` holds its reference-noise settings. Keys the
configs do not define raise an error.

## Decoder contract

`decode()` returns `coords` [B, N, 37, 3] in nanometres, `sequence_logits`
[B, N, classes], their `residue_type` argmax [B, N], and, when the checkpoint
has the head, raw `atom_presence_logits` [B, N, 37]. Undecoded tokens have
zero coordinates, residue ID -1 and presence logits of negative infinity.

Sequence conditions remain inputs to the decoder network. The application
applies residue overrides, legal-atom tables, presence thresholds and required
atoms. It also restores conditioned coordinates and formats ligand and motif
outputs. BioIR does not return alternative "complex" or "final" structure views.

## Numerical example

```bash
python examples/design/rfproteina.py \
  --flow /path/to/flow.ckpt --autoencoder /path/to/autoencoder.ckpt \
  --length 128 --seed 42
```

This supplies a toy state and explicit ODE grids, then prints tensor shapes. A
host application provides assembled structures, partial redesign, inverse
folding and search.

## Graphs and reproducibility

`configure_graphs()`, `graph_status()` and `reset_graphs()` control and inspect
the numerical trunk cache. Recycle steps share captured graphs; workflow and
search logic stay outside capture.

Paired backend comparisons require the same seed, input shape, schedule and
recycle depths. Every backend receives the same unbucketed sampling shapes:
compiler length buckets pad inside the native trunk, after random draws and
features, and BioIR pads inside its trunk only for kernel alignment.
Floating-point kernels can still produce different trajectories and search
selections after identical random draws.

## Migration

The removed `redesign()` and `inverse_fold()` workflows and structure-output
helpers belong to the host application. Replace `RFProteinaPredictionOutput`
and its materialized views with simulation results and raw decoder tensors.
Replace `advance_segment` with `partial_simulation` and `encode_clean_state`
with `encode`; supply `ts`, `gt` and `recycle_depths` explicitly. Replace
imports from the internal RFProteina package root with imports from the
module that defines each symbol. The public entry points remain under
`bionemo_ir.models.rfproteina`; detailed configuration and weight conversion
remain in its `config` and `convert` modules.

## Frontend-owned heads and features

Pass a PyTorch `heads` module to `RFProteina.from_reference_state_dicts` to keep
readout computation in the frontend. It implements
`forward(single, pair, padding_mask, emit_masks)` and declares `requires_pair`.
A rollout passes the masks it integrates with as `emit_masks`: `bb_ca` marks the
generated tokens and `local_latents` the tokens whose latents it emits, so the
readout and the sampler cannot disagree. The module receives unpadded features;
a requested pair is the final trunk pair, before recycle projection. Heads run
outside the captured trunk. Single-only heads do not retain or copy the final
pair. The engine owns and loads the supplied module in the trunk dtype,
preserving integer and boolean buffers. Its source weights live under
`nn.heads.`. Caller-owned readouts can replace every released head; trunk shape
inference does not require released head parameter names.

Pass `reference_config` with the training network configuration when using the
optimized feature builder. Conversion rejects source parameters consumed by none
of the responsible components, and checks supported feature order and numerical
settings. Arbitrary code changes with identical weights/configuration still
require numerical parity tests.

### Trunk-only API

A frontend can supply representations directly, without constructing the full
engine. Construct the trunk once, then call it at each denoising step:

```python
import torch
from bionemo_ir.models.rfproteina import RFProteinaTrunk
from bionemo_ir.models.rfproteina.config import trunk_graph_config

trunk = RFProteinaTrunk.from_reference_state_dict(flow_weights, heads=heads, device="cuda")
trunk.trunk_graph.enable(trunk_graph_config())
with torch.inference_mode():
    single_out, pair_out = trunk.forward_features(
        single.to(trunk.dtype), pair.to(trunk.dtype), condition.to(trunk.dtype),
        padding_mask, recycle_depth=3, return_pair=trunk.heads.requires_pair,
    )
    output = trunk.heads(single_out, pair_out, padding_mask, emit_masks)
```

`flow_weights` holds merged native `nn.*` weights; `heads` is the caller's
readout module. Inputs are CUDA tensors: `single` [B, N, token_dim], `pair`
[B, N, N, pair_dim], and boolean `padding_mask` [B, N]. The widths are in
`trunk.config`. `condition` is the raw factory output before the two condition
transitions: [B, N, condition_dim], or [recycle_depth, B, N, condition_dim]
for different conditions at each recycle.
Outputs remain valid after later graph replays. Use `return_pair=False` when the
frontend consumes only single features. Pair output needs two graph variants for
multiple recycles; the standard single-only path uses one per shape.

Feature semantics, integration and readouts are caller-owned here. The tensor
contract and supported trunk architecture still apply. Rebuild the trunk after
changing its weights or architecture.

### Why a host uses partial simulation

The full engine prepares static feature contributions and attention masks once
per interval, then reuses them across steps. Its feature builder writes pairs
directly into the trunk's input buffers. A trunk-only frontend must perform that
work itself. Both APIs expose the same numerical trunk; partial simulation adds
these optimizations without taking ownership of the host's workflow.

In the released-checkpoint H200 comparison, partial simulation reduced warm
request time by 15.7–35.3% at batch 1 and 1.9–7.8% at batch 8 versus a
trunk-only frontend with compiled native features. Six cases covered 100–250
tokens, protein-only, PPI, DNA and enzyme inputs, each with 60 steps and 180
trunk passes. Each pair used the same GPU and inputs, two warmups and four timed
requests, repeated in reverse order. Benefits depend on the frontend and shape.
