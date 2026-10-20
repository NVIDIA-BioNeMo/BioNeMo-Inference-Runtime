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

# Design proteins with RFProteína

RFProteína (RFD4) addresses a long catalogue of design tasks with one
conditional sampler. This guide starts from how such a model is controlled,
derives the task catalogue from it, then shows the runnable commands. For the
networks and kernels underneath, see
[RFProteína architecture](../ref/rfproteina.md).

## How a design model is controlled

The task catalogue looks long, but every task is a choice of position along
three orthogonal axes. Internalizing this reduction is the point: it is why
BioIR exposes a handful of entry points rather than one function per
biological task.

**Axis A — conditioning features.** Tensors consumed by the representation
builder and folded into the single, pair, and condition representations before
the trunk runs. They *bias the velocity field*; they never constrain the
state. A hotspot mask makes contact more likely; it does not pin an atom.
Because they are static across the trajectory, all purely feature-derived work
is loop-invariant and can be hoisted.

**Axis B — state clamping and partial noising.** Choices about the initial
condition and about what is overwritten at each step. Starting from an
intermediate time $t_0 > 0$ with part of the state initialized from a
reference structure gives partial redesign; holding the backbone at its clean
value while the latents run the full trajectory gives inverse folding. This
axis constrains the state directly, so it is strictly stronger than Axis A.

**Axis C — trajectory search.** An outer loop that branches the trajectory at
chosen checkpoints, scores partial rollouts with a metric ensemble, and keeps
the best beam. It changes neither the velocity field nor the state update,
only *which* sampled trajectories survive.

Axis A is soft and cheap, Axis B is hard and cheap, Axis C is soft and
expensive — it multiplies network evaluations by the branching factor. When an
objective can be expressed on Axis A or B, expressing it on Axis C instead is
the most common way to waste a GPU-hour.

## Design workflows

Each task below is a position on those axes.

**Unconditional de novo generation.** Supply only a length and a chain layout.
Both modalities initialize from the reference distribution and integrate to
$t = 1$. This is the base case against which every conditional task is a
restriction.

**Fold-class and secondary-structure conditioning.** Supply a CATH label or a
per-residue secondary-structure assignment. Pure Axis A: the state is still
fully sampled, but the field is steered toward a topology class or an
$\alpha/\beta$ pattern. This is the cheapest way to control global topology
and should be preferred over search whenever it suffices.

**Motif scaffolding.** Supply a functional motif and ask for a protein that
presents it. The motif enters as atom37 coordinates and their mask, internal
coordinates, and optionally a fixed sequence. *Indexed* scaffolding fixes
which generated residue indices carry the motif; *unindexed* scaffolding
supplies the motif as a set and lets the model place it, with
`condition_index-group_1_token` grouping residues that must stay contiguous
and correctly ordered. Unindexed is the harder and more useful mode, and it is
why motif RMSD must be evaluated under an index assignment rather than
positionally.

**Protein–protein binder design.** Supply a target chain as motif-style
coordinate conditioning plus a hotspot mask marking target residues the binder
should engage, then generate a new chain. The target is conditioning, not
state: it biases the field and is not itself re-generated.
`mask_not-cysteine_1_residue` is commonly set on the binder to suppress
unpaired cysteines.

**Small-molecule binder and enzyme design.** Supply ligand atoms in the
atomized token representation together with interaction conditioning —
hydrogen bonds, relative accessible surface area, and, in the tensor schema,
$\pi$-stacking and $\pi$-cation channels. Enzyme design is this task with a
catalytic-site motif added on Axis A, so the two share an implementation path
and differ only in which tensors are populated. Treat the $\pi$-stacking and
$\pi$-cation channels as plumbed, not as validated controls: they are consumed
by the representation builder but are not exercised as active constraints in
the released validation set.

**Nucleic-acid complex design.** Protein binders for DNA or RNA targets. The
residue vocabulary is unified across protein, RNA, and DNA, and nucleic-acid
secondary structure enters through per-token state (`na_ss_state`) and a
pairwise channel (`na_ss_pair`) that expresses base pairing, which is
intrinsically a pair-level rather than token-level constraint.

**Partial redesign.** Supply an existing structure and ask for a variation.
The clean state is encoded, noised to an intermediate time $t_0$, and
integrated over the schedule tail $[t_0, 1]$. This is Axis B, and $t_0$ is a
continuous similarity dial: small $1 - t_0$ keeps the input nearly intact,
large $1 - t_0$ approaches unconditional generation.

**Inverse folding.** Supply a backbone and ask for a sequence that folds to
it. This is not simply partial redesign with the backbone frozen. Both
modalities start at full noise and the backbone is not integrated at all: at
every step it is *resampled* from the forward marginal of the supplied clean
backbone at the current time, so the latents always see a backbone noised
consistently with where they are on the trajectory. The clean backbone is
restored at the end. Freezing the backbone at its clean value instead would
present the latents with an out-of-distribution input at every early step.

### Task-to-API mapping

BioIR exposes the tasks through a small tensor-only surface on `RFProteina`.
The boundary is deliberate: inputs are already-tensorized dictionaries, and
structure parsing, tokenization, and file writing live outside the model
package.

- `forward` — unconditional and conditioned design: `sample` plus decode and
  materialization.
- `sample` — product-space trajectory only, no decoding.
- `redesign` — encodes the input, noises to $t_0$, runs the tail.
- `sample_partial` — redesign from an already-encoded clean state.
- `inverse_fold` — backbone resampled from its forward marginal each step.
- `search` — beam search over any of the above.
- `encode_clean_state` and `decode` — the autoencoder boundary.
- `velocity` — a single trunk evaluation, for parity checks and profiling.

Because that boundary is tensor-only, RFProteína has no pipeline factory, so
it cannot yet be driven through `build_processor` the way the folding models
are. Callers assemble the feature dictionary themselves, as the example below
does, and own structure parsing and file writing. Running a design workflow
under `build_processor` — with its batching, Ray executor, and writer stages —
is future work.

### Conditioning tensor contract

These are the Axis A tensors as they appear in the serialized batch contract.
Atom-level entries use the dense atom37 layout. BioIR consumes these keys
directly and performs no parsing, tokenization, or annotation — those stay
upstream of the model boundary. Every one of them is constant over the
trajectory.

- `condition_coordinate_1_atom` (atom37) — motif or target geometry.
- `condition_InternalCoordinate_1_atom` (atom37) — motif internal geometry.
- `condition_sequence_1_residue` (residue) — fixed or preferred identities.
- `condition_index-group_1_token` (token) — contiguity groups for unindexed
  motifs.
- `condition_cath-label_1_residue` (residue) — fold-class steering.
- `condition_secondary-structure_1_residue` (residue) — per-residue
  $\alpha$/$\beta$/loop.
- `mask_hotspot_1_atom` (atom) — interface residues to engage.
- `mask_not-cysteine_1_residue` (residue) — suppress cysteine.
- `condition_hbond_1_atom` (atom) — hydrogen-bond intent.
- `condition_rasa-binned_1_atom` (atom) — burial and exposure.
- `condition_ChainCentroid_1_atom` (atom) — coarse chain placement.
- `na_ss_state` (token) and `na_ss_pair` (pair) — nucleic-acid secondary
  structure.
- `flow_backbone_mask` (token) — the tokens whose backbone the flow generates.
  A token outside it, such as an unindexed motif residue, only conditions: the
  trunk attends to it, but it holds zero state, takes no step, and is left out
  of centering. It defaults to every real token.
- `chirality` (token) with `chiral_frame_atoms` and `chiral_slot_owner` —
  stereochemical vectors for atomized ligands. A checkpoint trained with
  noise-aligned chirality needs all three, because the vectors are re-expressed
  in the noisy state's frame at every step. The weights do not record that
  training choice, so the host that holds the checkpoint's configuration sets
  `noise_align_chirality` when it builds the model.

## Oracles and metric-weighted search

Search is Axis C. At configured step checkpoints the sampler branches each
surviving trajectory, scores the branches with a weighted objective, and
prunes back to the beam width. Scoring can use cheap local metrics — clashes,
motif RMSD, sequence adherence, ligand contact — or an external structure
oracle you supply.

Local metrics need no external service:

```python
from bionemo_ir._torch.modules.rfproteina.metrics import evaluate_local_metrics
```

For an oracle-backed reward, implement the `OracleProvider` contract and wrap
it with the search adapter. A provider owns its model, service, container,
retries, and resource allocation; BioIR imports and modifies no upstream
oracle code:

```python
from bionemo_ir.oracles.adapters.search import OracleMetricEvaluator
```

Configure the beam with `RFProteinaBeamSearchConfig` and pass it to
`RFProteina.search`:

```python
from bionemo_ir.models.rfproteina import RFProteinaBeamSearchConfig
```

Budget search deliberately. It multiplies loop cost by the branching factor at
every checkpoint, and an external oracle can cost more than the entire
sampling budget. Search also cannot create quality that the field does not
already produce — it re-ranks trajectories, it does not redirect them. See
[Custom structure oracles](../ref/oracles.md) for the full provider contract,
the request and response schemas, and the residue and atom resolvers.

## Run the example

`examples/design/rfproteina.py` runs one design end to end and prints the
designed sequence with the elapsed time. It needs a CUDA device and the two
released checkpoints — the flow (LDM) checkpoint and the autoencoder
checkpoint.

The script builds its configuration from the checkpoints themselves rather
than hard-coding shapes, using `RFProteina.get_pretrained_config` for the
optimized defaults (BF16 trunk, auto-selected attention backend) and the
`rfproteina_*_config_from_weights` helpers for the architecture:

```python
from bionemo_ir.models.rfproteina import (
    RFProteina,
    RFProteinaConfig,
    rfproteina_decoder_config_from_weights,
    rfproteina_representation_config_from_weights,
    rfproteina_trunk_config_from_weights,
)
```

Each subcommand corresponds to one workflow above:

```bash
# Unconditional de novo generation.
python examples/design/rfproteina.py \
  --flow "$RFPROTEINA_LDM_CKPT" --autoencoder "$RFPROTEINA_AE_CKPT" \
  --length 128 unconditional

# Motif scaffolding around a fixed helical CA motif.
python examples/design/rfproteina.py \
  --flow "$RFPROTEINA_LDM_CKPT" --autoencoder "$RFPROTEINA_AE_CKPT" \
  --length 128 motif --motif-length 16

# Protein-protein binder against a fixed target, with hotspots.
python examples/design/rfproteina.py \
  --flow "$RFPROTEINA_LDM_CKPT" --autoencoder "$RFPROTEINA_AE_CKPT" \
  --length 96 ppi --target-length 48 --hotspot-residues 8

# Small-molecule binder around a fixed atomized ligand pose.
python examples/design/rfproteina.py \
  --flow "$RFPROTEINA_LDM_CKPT" --autoencoder "$RFPROTEINA_AE_CKPT" \
  --length 96 small-molecule --ligand-atoms 8

# Nucleic-acid binder against a fixed DNA sequence.
python examples/design/rfproteina.py \
  --flow "$RFPROTEINA_LDM_CKPT" --autoencoder "$RFPROTEINA_AE_CKPT" \
  --length 96 nucleic-acid --dna-length 16 --structure

# Partial redesign of an existing backbone.
python examples/design/rfproteina.py \
  --flow "$RFPROTEINA_LDM_CKPT" --autoencoder "$RFPROTEINA_AE_CKPT" \
  --length 128 partial-redesign --partial-t 0.6
```

`--steps` overrides the sampling-step count; omit it to use the model's
released schedule. `--seed` fixes the draw.

Two flags map directly onto the axes. `--hotspot-residues` is Axis A: it marks
target residues without pinning anything, so raising it biases the interface
rather than enforcing it. `--partial-t` is Axis B: it is the similarity dial,
where values near 1 keep the input backbone nearly intact and values near 0
approach unconditional generation.

### Run beam search

`examples/design/search.py` is the Axis C counterpart. It reuses the feature
builders from `rfproteina.py`, wraps them in `RFProteina.search`, and scores
with BioIR's built-in `RFProteinaLocalMetricEvaluator` — no external oracle
and no extra dependency:

```bash
python examples/design/search.py \
  --flow "$RFPROTEINA_LDM_CKPT" --autoencoder "$RFPROTEINA_AE_CKPT" \
  --length 64 --steps 6 --beam-width 2 --n-branch 2 \
  motif --motif-length 8
```

```text
motif: beam 2 x branch 2 over 3 checkpoints -> 8 rollouts, 36 velocity steps in 62.45s
weights: {'clash_metrics.frac_atoms_clashing_1p5A': -1.0, 'design_motif_allatom_rmsd.rmsd': -0.1}
  slot 0  reward -0.5688  frac_atoms_clashing_1p5A=0.1801 rmsd=3.8866
  slot 1  reward -0.6115  frac_atoms_clashing_1p5A=0.1917 rmsd=4.1974
best (slot 0, seed 6978676964798891550):
SYWWWTWYPSNDTSIVITFRSCWWQEAKWIQPDDYRQQEQSDFNPLYINAPWGEYFESMWSQKY
```

That run uses six sampling steps so it finishes quickly; the clash fraction is
correspondingly poor and is not representative of a real design budget.

The four subcommands — `unconditional`, `motif`, `ppi`, and `small-molecule` —
differ only in which metrics the features make available, and the script
weights exactly those. This matters because a weighted key that the features
cannot produce scores minus infinity and rejects every candidate, so motif
RMSD is weighted only when coordinates are pinned and ligand contact only when
an atomized ligand is present. Clash applies everywhere.

Two fields of the printed beam are easy to misread. `reward` is the weighted
sum, where negative weights mean "minimize", so a less negative reward is
better. `slot` is the beam position, not a quality rank — the script sorts by
reward and reports the best first.

## Related pages

- [RFProteína architecture](../ref/rfproteina.md) — networks, shapes, kernels,
  and measured results.
- [Custom structure oracles](../ref/oracles.md) — the oracle provider
  contract.
- [Sampling architecture](../ref/sampling.md) — the shared rollout runtime.
