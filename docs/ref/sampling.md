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

# Sampling architecture

BioNeMo Inference Runtime (BioIR) uses a Torch-native runtime for iterative
generative sampling. Model samplers own model-specific conditioning and
prediction calls, while `_torch/sampling` owns reusable rollout contracts,
iteration, random state, hooks, and integration algorithms.

## Overview

```mermaid
flowchart LR
    M["Model<br/>Boltz / OpenFold3 / Protenix / RFProteína"]
    S["Model-specific sampler<br/>conditioning, reference draw, schedule, predict()"]
    R["GenerativeRunner<br/>seeded step loop"]
    I["Concrete integrator<br/>AF3 / Boltz EDM<br/>default / RFProteína flow matching"]
    T["DenoiseIntegratorTemplate.step()<br/>fixed skeleton:<br/>prepare → hooks → predict → hooks → complete"]
    D["Denoiser / velocity network<br/>model-owned nn.Module"]
    O["Sampled coordinates or product-space state"]

    M --> S --> R --> I --> O
    I -.->|"extends fixed skeleton"| T
    I -->|"sampler predict()"| D
```

The Template Method separates the invariant `step()` skeleton from
algorithm- and model-specific operations. Each concrete integrator implements
those operations directly. OpenFold3 and Protenix use `AF3EDMIntegrator` with
tensor state, Boltz uses `BoltzEDMIntegrator` with structured state,
`FlowMatchingIntegrator` provides the default product-space flow, and
RFProteína specializes it with `RFProteinaFlowMatchingIntegrator`.

## Ownership boundaries

### Model

The top-level model owns trunk execution, model configuration, checkpoint
loading, and the denoising neural network. The denoiser remains an `nn.Module`
and is injected into the model-specific sampler:

```python
diffusion_module = DiffusionModule(config=...)
diffusion_sampler = ModelDiffusionSampler(
    config=...,
    diffusion_module=diffusion_module,
)
```

This keeps neural weights separate from rollout orchestration and allows
optimized submodules to be replaced through the model registry.

### Model-specific sampler

The sampler adapts the model to the shared runtime. It:

- prepares model-specific conditioning and masks;
- draws the initial state the rollout starts from;
- creates the rollout plan and `SamplingContext`;
- binds invariant tensors into a `predict(model_input, time)` closure;
- selects the concrete integrator and hooks;
- decodes the final runtime state into the model's public output.

The expensive denoiser call remains model-specific. The step loop does not.

### `GenerativeRunner`

`GenerativeRunner` owns algorithm-independent rollout mechanics:

1. initialize state through the integrator;
1. iterate for `integrator.num_steps(plan)`;
1. update `context.step_index`;
1. honor cancellation;
1. return `SamplingResult` with the final state and seed.

The runner does not know whether a prediction is denoised coordinates,
velocity, token logits, or a model-specific structured object.

### Integrator

`DenoiseIntegratorTemplate` fixes the denoiser-driven step order:

```text
prepare step
  -> before-denoise hooks
  -> extract model input and time
  -> predict
  -> attach prediction
  -> after-denoise hooks
  -> complete mathematical update
```

Concrete integrators implement this contract directly:

- `AF3EDMIntegrator` implements the AF3-style first-order EDM algorithm over
  coordinate tensors for OpenFold3 and Protenix.
- `BoltzEDMIntegrator` implements the same template for Boltz's structured
  state, including atom masks, previous denoised coordinates, and token
  representations.
- `FlowMatchingIntegrator` implements deterministic Euler over a dict of
  per-modality tensors with linear-interpolant clean endpoints.
- `RFProteinaFlowMatchingIntegrator` subclasses `FlowMatchingIntegrator` to
  supply mask-aware endpoints, the VF-TSR update, and rollout diagnostics.

None reimplements `step()`. Each supplies the template's typed operations:
initialization, step preparation, model inputs, prediction attachment, and the
mathematical update.

### Hooks

`DenoiseHookPipeline` applies ordered extensions around each prediction call.
Boltz uses hooks for:

- random rigid augmentation;
- churn-noise injection;
- potential guidance and particle resampling;
- token-representation accumulation;
- optional rigid alignment.

Hook constructors own rollout-local configuration and mutable state. Random
operations receive the explicit generator from `SamplingContext`.

## EDM step

The EDM implementation follows the first-order portion of Algorithm 2 from
Karras et al., [Elucidating the Design Space of Diffusion-Based Generative
Models](https://arxiv.org/pdf/2206.00364):

The initial coordinates are sampled at the largest noise level:

$$
\boldsymbol{x}_0 \sim
\mathcal{N}\!\left(\boldsymbol{0}, \sigma_0^2 \boldsymbol{I}\right).
$$

Each sampling step temporarily adds noise, invokes the denoiser, and advances
toward the next noise level:

$$
\begin{aligned}
\hat{\sigma}_i
    &= \sigma_i \left(1 + \gamma_i\right), \\
\boldsymbol{\epsilon}_i
    &\sim \mathcal{N}\!\left(\boldsymbol{0}, \boldsymbol{I}\right), \\
\hat{\boldsymbol{x}}_i
    &= \boldsymbol{x}_i
       + s_{\mathrm{noise}}
         \sqrt{\hat{\sigma}_i^2 - \sigma_i^2}\,
         \boldsymbol{\epsilon}_i, \\
\boldsymbol{x}^{\mathrm{denoised}}_i
    &= \mathrm{predict}
       \left(\hat{\boldsymbol{x}}_i, \hat{\sigma}_i\right), \\
\boldsymbol{d}_i
    &= \frac{
         \hat{\boldsymbol{x}}_i
         - \boldsymbol{x}^{\mathrm{denoised}}_i
       }{\hat{\sigma}_i}, \\
\boldsymbol{x}_{i+1}
    &= \hat{\boldsymbol{x}}_i
       + s_{\mathrm{step}}
         \left(\sigma_{i+1} - \hat{\sigma}_i\right)
         \boldsymbol{d}_i .
\end{aligned}
$$

In the implementation, `noise_scale` supplies the noise multiplier and
`step_scale` supplies the Euler-step multiplier.

This is the AF3/Boltz first-order adaptation, not the paper's full Heun
sampler. It intentionally omits Algorithm 2's second denoiser evaluation and
second-order correction. Churn gating, `step_scale`, rigid augmentation, and
model hooks follow model-specific inference algorithms.

## Model specializations

### OpenFold3 and Protenix

OpenFold3 and Protenix use:

- `EDMRolloutPlan`;
- `AF3EDMIntegrator`;
- a model-specific `predict` closure around their `DiffusionModule`.

Their differences remain in denoiser signatures, conditioning, schedule
termination, and optional caches.

### Frame-aware prediction reuse

Protenix, OpenFold3 and Boltz-2 can skip some denoiser calls near the end of
the rollout. This is an **approximation of the denoiser**, not of the
integration: every EDM update, rigid augmentation and churn draw still happens,
so the random streams are unchanged. It is **on by default** for Protenix and
OpenFold3 at the step counts it was evaluated with, and **opt-in** for Boltz-2.

The window is derived from the rollout plan, not configured. It starts at the
first step whose churn rate is zero, because a reused step must not inject
churn noise (step $0.8\,n$ with the default schedule), and it ends two steps
before the last, so the final two steps always evaluate the network. It was
**only evaluated with 100, 150, 200 and 400 sampling steps**; any other count
runs exact. 50 steps is left out on purpose: OpenFold3 lost 0.003 lDDT there.
For 200 steps the window is steps 160 to 197.

The config field `frame_reuse_period` takes three values:

- `"auto"`, the Protenix and OpenFold3 default, reuses with period 4 when the
  rollout has an evaluated step count and the sampler can reuse in it, and runs
  exact otherwise. A sampler cannot reuse in training, with coordinate
  augmentation off, or, for Boltz-2, with steering. Nor can it reuse when the
  churn gate leaves no churn-free step before the two fresh final steps, as with
  `gamma_min = 0`.
- `0` always runs exact, and is the Boltz-2 default. Use it to compare against
  an upstream rollout or to reproduce outputs from before reuse became the
  default.
- `4` forces reuse and raises a `ValueError` wherever `"auto"` would run
  exact.

```python
config = Protenix.get_pretrained_config("protenix-v2")
config.edm_sampling_config.frame_reuse_period = 0  # exact; "auto" (default) reuses at 100, 150, 200 and 400 steps
model = Protenix(model_name="protenix-v2", config=config, include_load_weights=True)
```

For OpenFold3 the field is the same name on its `EDMSamplingConfig`
(`models/openfold3/config.py`), with the same values. The configs check the
field when it is assigned, not only when they are built, so an unsupported
value fails on the line that sets it.

With `frame_reuse_period = p` and a window $[s, e]$, steps $s \le i \le e$
evaluate the network when $(i - s) \bmod p = 0$ and reuse the previous
prediction otherwise. Only $p = 4$ was evaluated, so `frame_reuse_period`
accepts `"auto"`, 0 or 4 and raises a `ValueError` for any other value. At 200
steps that is 172 network calls and 28 reused predictions. With 4 forced, any
other step count raises a `ValueError` ("frame_reuse_period was only evaluated
with (100, 150, 200, 400) sampling steps; got N") before any model work: each
model checks its call's step count before the trunk runs, and Protenix also
checks `n_step` when its config is built or assigned. A plan with coordinate
augmentation off raises when the rollout starts.

Both models were compared with the exact rollout on 99 PDB entries, 5 samples
each. Every sample's rms displacement stayed below the bound computed from the
reused steps, and the mean lDDT and DockQ change was within 0.002 and 0.005.
An atom moved by more than 1 Å in 24 to 35 percent of the Protenix samples and
33 to 51 percent of the OpenFold3 samples. For comparison, two exact rollouts
whose inputs differ by 0.0005 Å differ that much in 0 to 3 percent. Both models
were run at 100, 150, 200 and 400 steps.

Ending the window at step 198 instead of 197 gave the same accuracy on 99 PDB
entries with Protenix at 200 steps (median rms 0.0293 against 0.0292 Å versus
the exact rollout), so step 198 stays fresh only by convention. Step 199
differs: its next noise level is 0 for Protenix, which gives it an error gain
six times larger than its neighbours, so it stays fresh.

**What is reused.** Each step first applies a fresh rigid augmentation
$T_i(\boldsymbol{x}) = R_i(\boldsymbol{x} - \boldsymbol{c}_i) + \boldsymbol{t}_i$
to the persistent coordinates. The denoiser is equivariant, so its previous
output, given in the previous step's frame, is mapped through the current step's
$T_i$ instead of being recomputed. Only Cartesian coordinates are mapped; no
learned feature channel is reused.

```python
for i in range(n_steps):
    x, T_i = augment(x)                       # same random draws as without reuse
    x_hat = x + noise_i                       # churn is zero at every reused step
    if reuses(i):                             # start <= i <= end and (i - start) % p != 0
        D = T_i(D_prev)                       # map the previous prediction, no network call
    else:
        D = network(x_hat, sigma_hat_i)
    d = (x_hat - D) / sigma_hat_i
    x = x_hat + eta * (sigma_next_i - sigma_hat_i) * d
    D_prev = D                                # a mapped prediction is cached again
```

**Why the error stays small.** A prediction error $\varepsilon_i$ at step $i$
moves the next state by $a_i \varepsilon_i$ with

$$
a_i = \eta \left(1 - \frac{\sigma_{i+1}}{\hat{\sigma}_i}\right),
$$

and late steps have $\sigma_{i+1} \approx \hat{\sigma}_i$, so $a_i$ is small.
The staleness error $\varepsilon_i$ is the difference between the mapped
previous prediction and the one the network would give. If the remaining steps
do not amplify a perturbation (gain at most 1), the final coordinates differ
from the exact rollout by at most

$$
B = \sqrt{f^2 + \Big(\sum_{i \in \text{reused}} a_i \, \varepsilon_i\Big)^2},
$$

where $f$ is the numerical floor: any perturbation at steps $\ge 160$ re-rolls
TF32 rounding and moves the output by about 0.01 to 0.016 Å rms. The
$\varepsilon_i$ are measured per sample, so $B$ is a per-sample check, not a
guarantee; on the benchmark inputs and on 99 further PDB entries it held for
every sample. It bounds the rms displacement; it says nothing about single
atoms, a few of which can move by more than 1 Å (noise alone does this
for a fraction of samples too).

**Expected effect.** At 200 steps, Protenix `model.forward` is 3 to 12 percent
faster on the benchmark inputs, less on large inputs, because 28 denoiser calls
are saved whatever the input size. The OpenFold3 diffusion sampler is 15 to 18
percent faster at 200 and 400 steps. Output is approximate: it is judged on
paired lDDT / DockQ against the exact rollout, not on bit equality
([oracles.md](oracles.md#approximate-options)). With `frame_reuse_period = 0`
the output is bit-identical to the exact sampler.

### Boltz

Boltz uses:

- `EDMRolloutPlan` with an appended terminal zero;
- `BoltzEDMIntegrator`;
- a model-specific ordered hook pipeline;
- a chunked `predict` closure around its injected `DiffusionModule`.

This preserves steering, resampling, alignment, token accumulation, and sample
chunking without duplicating the EDM loop.

**Frame reuse (Boltz-2).** Boltz-2 takes the same
[frame-aware prediction reuse](#frame-aware-prediction-reuse) through
`atom_diffusion.frame_reuse_period`, with the same values, but the field
defaults to 0, so Boltz-2 runs exact unless asked. Boltz's augmentation
hook already maps the previous denoised coordinates through each step's rigid
transform, so `BoltzFrameReuseEDMIntegrator` only skips the network call at a
reused step and keeps that mapped prediction. The window comes from the same
rule. With Boltz's schedule, which appends its terminal zero, churn stops
earlier: at 200 steps the window is steps 154 to 197, so 167 steps evaluate the
network and 33 reuse.

```python
config = Boltz2.get_pretrained_config("boltz-2")
config.structure_module.atom_diffusion.frame_reuse_period = "auto"  # opt in: reuses at 200 and 400 steps
```

Boltz-2 kept its accuracy only at 200 and 400 steps. On 17 benchmark entries
with 5 samples each, the mean lDDT change against the exact rollout was -0.0009
± 0.0056 at 200 steps and -0.0017 ± 0.0028 at 400 steps (95% confidence
interval). At 100 and 150 steps it dropped by 0.023 and 0.013, so `"auto"` runs
those counts exact and a forced 4 rejects them. The 200-step interval cannot
rule out a loss of about 0.006 lDDT, so the option stays off by default until a
larger comparison narrows it. `model.forward` was 7 and 10 percent faster at 200
and 400 steps on inputs of 600 residues or more. Steering and Boltz-1's
`accumulate_token_repr` read or edit every fresh prediction, so `"auto"` runs
exact with steering, a forced 4 rejects it, and Boltz-1 has no field.

## Flow-matching step

Flow matching transports a reference sample to a data sample along a
time-parameterized path. For the linear (conditional optimal-transport)
interpolant used here,

$$
\boldsymbol{x}_t = t\,\boldsymbol{x}_1 + (1 - t)\,\boldsymbol{x}_0 ,
$$

the network predicts the velocity `v = dx_t/dt`. One interval is an Euler step,
and the clean endpoint implied by a velocity follows from differentiating the
interpolant:

$$
\begin{aligned}
\boldsymbol{v}_i
    &= \mathrm{predict}\left(\boldsymbol{x}_i, t_i\right), \\
\boldsymbol{x}^{\mathrm{clean}}_i
    &= \boldsymbol{x}_i + \left(1 - t_i\right) \boldsymbol{v}_i, \\
\boldsymbol{x}_{i+1}
    &= \boldsymbol{x}_i
       + s_{\mathrm{step}} \left(t_{i+1} - t_i\right) \boldsymbol{v}_i .
\end{aligned}
$$

Time *increases* toward data at `t = 1`, the opposite direction to EDM's
decreasing noise levels. The clean endpoint is both a rollout output and, when
`self_condition` is enabled, the next step's self-conditioning input; it is
zero-filled before the first prediction, so self-conditioning is inactive on
step 0.

`GenerativeRunner.run_segment()` resumes any global interval range with an
explicit state. Continuation setup restores the caller's generator after
integrator initialization, and the first resumed prediction receives either
the parent's clean endpoint or explicit zeros according to
`carry_self_condition`. A caller that checkpoints a search can therefore resume
any branch from its saved state.

Sampling is *product-space*: a model may evolve several modalities on their own
time grids through one shared network evaluation, so states, times, and
velocities are keyed by modality name. `FlowMatchingRolloutPlan` requires every
modality to use the same number of intervals.

### Reference distribution

The rollout needs a sample at `t = 0` before the first interval. The shared
runtime does not draw it. Shapes, padding, and centering are model-specific in
a way generic code cannot guess, so `FlowMatchingIntegrator` accepts an
`initial_value` supplied by the model.

This keeps the two ends of the path in different layers: `_torch/sampling` owns
the transport from `t = 0` to `t = 1`, and the model owns what `t = 0` means.

### RFProteína

The host application owns initialization and workflow policy.
`RFProteina.partial_simulation` uses:

- Caller-supplied initial state, time/noise grids, and recycle depths;
- `FlowMatchingRolloutPlan` with one `FlowMatchingSchedule` per modality
  (`bb_ca` and `local_latents`), each carrying its own stochastic forcing rate;
- `RFProteinaFlowMatchingIntegrator`;
- a model-specific `predict` closure that builds representations from the
  current product-space state before calling the velocity network.

#### Product-space reference distribution

`RFProteinaProductSpaceFlowMatcher` draws the reference state one modality at a
time, in a fixed order, so a seed reproduces the same starting point exactly.

The draw is more than a Gaussian, because RFProteína centers it. The backbone is
projected onto the zero-center-of-mass subspace, since a translated backbone is
the same structure and the network should never have to learn that. *Stochastic
centering* then offsets tokens by a small random translation, which is what
teaches the model that absolute position carries no meaning. A scope decides who
moves together:

- `all` applies one offset to every token;
- `without_absolute_coords` leaves coordinate-conditioned tokens at the origin,
  since moving them would break the conditioning they carry;
- `per_group` gives each chain, molecule, or coordinate group its own offset, so
  the components of a complex are placed independently.

The `per_group` scope reads token groupings that `groups.py` extracts from the
static features, keyed by `chain_iid`, `pn_unit_iid`, `molecule_iid`, or
`coord_group`.

Each modality configures its own reference block, so the released checkpoint
centers the backbone and leaves the latents as plain masked noise. Latent width
comes from the checkpoint rather than the config, so the two cannot disagree.

#### Interval update

The simulator's prediction closure converts
`{modality: {"v": velocity}}` into the integrator's
`{modality: velocity}` contract and captures the optional insertion head.
`RFProteinaFlowMatchingIntegrator` specializes mask-aware clean endpoints, the
released Euler or VF-TSR update, latent masking, and optional trajectory
snapshots. Interval bookkeeping, self-conditioning handoff, iteration, and
random state remain shared.

The VF-TSR update is a stochastic extension of the Euler step above: it
converts velocity to the interpolant score, rescales the score by a
temperature ratio, and adds Brownian forcing on intervals where the time falls
inside the configured stochastic window.

## Source layout

```text
bionemo_ir/_torch/sampling/
  contracts.py             shared types, plans, context, and protocols
  runner.py                algorithm-independent rollout loop
  denoise_integrator.py    invariant denoiser-step ordering
  hooks.py                 ordered before/after-denoise hooks
  edm.py                   EDM schedule, math, and tensor integrator
  frame_reuse.py           frame-aware prediction reuse
  flow_matching.py         flow schedule, math, and product-space integrator
```

Model-specific rollout code lives directly beside its model. For RFProteína:

```text
bionemo_ir/_torch/modules/rfproteina/
  simulator.py             RFProteinaProductSpaceSimulator
  flow_matcher.py          product-space reference distribution and centering
  groups.py                token groupings for per-group centering
  vf_tsr.py                velocity/score conversions and the VF-TSR step
  rollout.py               RFProteinaFlowMatchingIntegrator
```
