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

"""Run one numerical RFProteina rollout; a host application assembles structures."""

import argparse
from pathlib import Path

import torch

from bionemo_ir._torch.modules.rfproteina.representation_builder import SEQUENCE_PROBABILITY_TOKENS
from bionemo_ir.models.rfproteina import RFProteina, RFProteinaSimulationConfig
from bionemo_ir.models.rfproteina.config import RFProteinaModalitySimulationConfig, RFProteinaVFStepConfig
from bionemo_ir.models.rfproteina.convert import load_rfproteina_checkpoint_state

_CA_SLOT = 1
_ATOMS = 37
_BOND_PATH_OVERFLOW = 63


def toy_simulation_config() -> RFProteinaSimulationConfig:
    """Plain flow matching for both modalities, to match the toy ODE grids below.

    A host application passes the sampler settings of its own configuration instead.
    """
    return RFProteinaSimulationConfig(
        bb_ca=RFProteinaModalitySimulationConfig(step=RFProteinaVFStepConfig(sampling_mode="vf")),
        local_latents=RFProteinaModalitySimulationConfig(step=RFProteinaVFStepConfig(sampling_mode="vf")),
    )


def load_model(flow_checkpoint: Path, autoencoder_checkpoint: Path) -> RFProteina:
    """Convert the supplied weights into a numerical engine."""
    return RFProteina.from_reference_state_dicts(
        load_rfproteina_checkpoint_state(flow_checkpoint),
        load_rfproteina_checkpoint_state(autoencoder_checkpoint),
        simulation_config=toy_simulation_config(),
        device="cuda",
    )


def _features(
    token_type: torch.Tensor,
    chain_iid: torch.Tensor,
    residue_type: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """One batch from per-token kind, chain, and residue-type vectors."""
    length = token_type.shape[0]
    device = token_type.device
    index = torch.arange(length, device=device)
    separation = (index[:, None] - index[None, :]).abs().clamp(max=_BOND_PATH_OVERFLOW)
    same_chain = chain_iid[:, None] == chain_iid[None, :]
    separation = torch.where(same_chain, separation, torch.full_like(separation, _BOND_PATH_OVERFLOW))
    atom_mask = torch.zeros(1, length, _ATOMS, dtype=torch.bool, device=device)
    atom_mask[..., _CA_SLOT] = True
    features = {
        "padding_mask": torch.ones(1, length, dtype=torch.bool, device=device),
        "token_type": token_type.view(1, length),
        "chain_iid": chain_iid.view(1, length),
        "molecule_iid": chain_iid.view(1, length),
        "residue_type": residue_type.view(1, length),
        "atom37_mask": atom_mask,
        "condition_index-group_1_token": torch.zeros(1, length, dtype=torch.long, device=device),
        "condition_cath-label_1_residue": torch.full((1, length, 3), -1, dtype=torch.long, device=device),
        "condition_CoordinateGroup_1_atom": torch.full((1, length, _ATOMS), -1, dtype=torch.long, device=device),
        "token_bond_path_length": separation.view(1, length, length),
    }
    unspecified = torch.full((1, length), torch.nan, device=device)
    for token in SEQUENCE_PROBABILITY_TOKENS:
        features[f"condition_sequence-probability-{token}_1_residue"] = unspecified.clone()
    return features


def chain_batch(length: int, device: torch.device) -> dict[str, torch.Tensor]:
    """One unconditioned protein chain.

    Sequence probabilities are unspecified (NaN). Bond-path length is the
    sequence separation, so the chain is one polymer.
    """
    zeros = torch.zeros(length, dtype=torch.long, device=device)
    return _features(zeros, zeros, zeros)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--autoencoder", type=Path, required=True)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.length < 1:
        parser.error("--length must be positive")
    model = load_model(args.flow, args.autoencoder)
    features = chain_batch(args.length, torch.device("cuda"))
    if args.steps < 1:
        parser.error("--steps must be positive")
    # An explicit toy ODE problem; production generation policy belongs to the host application.
    torch.manual_seed(args.seed)
    state = {
        key: torch.randn(1, args.length, width, device="cuda")
        for key, width in (("bb_ca", 3), ("local_latents", model.config.trunk.latent_dim))
    }
    ts = {key: torch.linspace(0.01, 1.0, args.steps + 1, device="cuda") for key in state}
    gt = {key: torch.zeros(args.steps, device="cuda") for key in state}
    with torch.inference_mode():
        result = model.partial_simulation(
            features,
            state,
            start_step=0,
            end_step=args.steps,
            ts=ts,
            gt=gt,
            recycle_depths=[1] * args.steps,
            seed=args.seed,
        )
        decoded = model.decode(result.state, features["padding_mask"])
    print({name: tuple(value.shape) for name, value in decoded.items()})


if __name__ == "__main__":
    main()
