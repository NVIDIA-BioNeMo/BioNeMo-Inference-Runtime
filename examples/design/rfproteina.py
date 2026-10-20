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

# Design one protein with RFProteína.

import argparse
import math
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from bionemo_ir._torch.modules.rfproteina.atomized import (
    ATOMIZED_COORDINATE_SLOT,
    ATOMIZED_RESIDUE_TYPE,
    ATOMIZED_TOKEN_TYPE,
)
from bionemo_ir._torch.modules.rfproteina.nucleic_acid import DNA_TOKEN_TYPE
from bionemo_ir._torch.modules.rfproteina.representation_builder import SEQUENCE_PROBABILITY_TOKENS
from bionemo_ir.models.rfproteina import (
    RFProteina,
    RFProteinaConfig,
    RFProteinaPartialSimulationConfig,
    RFProteinaPredictionOutput,
    RFProteinaSamplingConfig,
    rfproteina_decoder_config_from_weights,
    rfproteina_representation_config_from_weights,
    rfproteina_trunk_config_from_weights,
)
from bionemo_ir.models.rfproteina import convert as rfproteina_convert

# Unified residue indices 0 and 1-20. Anything else prints as X.
_LETTERS = "XARNDCQEGHILKMFPSTWYV"
_ALA = 1
_CA_SLOT = 1
# Unified DNA bases A, C, G, T. 31 is the unknown DNA token.
_DNA_BASES = (27, 28, 29, 30)
_BOND_PATH_OVERFLOW = 63
_ATOMS = 37


def _read_state(path: Path) -> dict[str, torch.Tensor]:
    """Read a checkpoint's tensors.

    The released files store a config object beside the state dict, so this
    does not use weights-only loading.
    """
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = saved["state_dict"] if isinstance(saved, dict) and "state_dict" in saved else saved
    if not isinstance(state, Mapping):
        raise TypeError(f"{path} does not contain a state dict")
    return state


def load_model(flow_checkpoint: Path, autoencoder_checkpoint: Path, steps: int | None) -> RFProteina:
    """Build an RFProteína model from its flow and autoencoder checkpoints."""
    flow = flow_checkpoint.resolve()
    autoencoder = autoencoder_checkpoint.resolve()
    weights = {flow: _read_state(flow), autoencoder: _read_state(autoencoder)}
    rfproteina_convert.load_rfproteina_checkpoint_state = lambda path: weights[Path(path).resolve()]

    pretrained = RFProteina.get_pretrained_config()
    trunk = rfproteina_trunk_config_from_weights(
        weights[flow],
        dtype="bfloat16",
        pairwise_attention_backend=pretrained.trunk.pairwise_attention_backend,
        enable_cudnn_graph=pretrained.trunk.enable_cudnn_graph,
    )
    representation = rfproteina_representation_config_from_weights(
        weights[flow],
        latent_dim=trunk.latent_dim,
        dtype="float32",
        pair_adaln_dtype="bfloat16",
    )
    decoder = rfproteina_decoder_config_from_weights(
        weights[autoencoder],
        dtype="float32",
        transformer_dtype="bfloat16",
    )
    sampling = pretrained.sampling
    if steps is not None:
        sampling = RFProteinaSamplingConfig(num_steps=steps, n_recycle_schedule=[(steps, 1)])
    config = RFProteinaConfig(
        dtype="bfloat16",
        trunk=trunk,
        representation=representation,
        decoder=decoder,
        sampling=sampling,
    )
    model = RFProteina(
        config,
        checkpoint_path=flow,
        autoencoder_checkpoint_path=autoencoder,
        include_load_weights=True,
    )
    return model.cuda().eval()


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


def helical_ca(count: int) -> torch.Tensor:
    """CA coordinates, in angstroms, of a short alpha helix."""
    index = torch.arange(count, dtype=torch.float32)
    angle = index * (100.0 * math.pi / 180.0)
    return torch.stack((2.3 * torch.cos(angle), 2.3 * torch.sin(angle), 1.5 * index), dim=-1)


def _coordinate_buffers(features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = features.get("condition_coordinate_1_atom")
    mask = features.get("mask_coordinate_1_atom")
    if coordinates is None or mask is None:
        length = features["padding_mask"].shape[-1]
        device = features["padding_mask"].device
        coordinates = torch.full((1, length, _ATOMS, 3), torch.nan, device=device)
        mask = torch.zeros(1, length, _ATOMS, dtype=torch.bool, device=device)
        features["condition_coordinate_1_atom"] = coordinates
        features["mask_coordinate_1_atom"] = mask
    return coordinates, mask


def pin_ca(features: dict[str, torch.Tensor], start: int, ca_angstrom: torch.Tensor) -> None:
    """Fix CA coordinates on ``features`` over ``[start, start + count)``."""
    length = features["padding_mask"].shape[-1]
    count = ca_angstrom.shape[0]
    if start < 0 or start + count > length:
        raise ValueError(f"span [{start}, {start + count}) is outside a chain of {length}")
    coordinates, mask = _coordinate_buffers(features)
    span = slice(start, start + count)
    coordinates[0, span, _CA_SLOT] = ca_angstrom.to(coordinates.device)
    mask[0, span, _CA_SLOT] = True


def pin_motif(features: dict[str, torch.Tensor], start: int, ca_angstrom: torch.Tensor) -> None:
    """Fix a helical CA motif. ``pin_ca`` is the general form."""
    pin_ca(features, start, ca_angstrom)


def pin_sequence(features: dict[str, torch.Tensor], start: int, residue_types: torch.Tensor) -> None:
    """Fix unified residue types over ``[start, start + count)``."""
    length = features["padding_mask"].shape[-1]
    count = residue_types.shape[0]
    if start < 0 or start + count > length:
        raise ValueError(f"sequence [{start}, {start + count}) is outside a chain of {length}")
    device = features["padding_mask"].device
    sequence = features.get("condition_sequence_1_residue")
    if sequence is None:
        sequence = torch.zeros(1, length, dtype=torch.long, device=device)
        features["condition_sequence_1_residue"] = sequence
        features["mask_sequence_1_residue"] = torch.zeros(1, length, dtype=torch.bool, device=device)
    span = slice(start, start + count)
    sequence[0, span] = residue_types.to(device)
    features["mask_sequence_1_residue"][0, span] = True


def pin_hotspots(features: dict[str, torch.Tensor], start: int, count: int) -> None:
    """Mark ``count`` target CA atoms, starting at ``start``, as hotspots."""
    length = features["padding_mask"].shape[-1]
    if count < 1 or start < 0 or start + count > length:
        raise ValueError(f"hotspot span [{start}, {start + count}) is outside a chain of {length}")
    device = features["padding_mask"].device
    hotspot = torch.zeros(1, length, _ATOMS, device=device)
    hotspot[0, start : start + count, _CA_SLOT] = 1
    features["condition_hotspot_1_atom"] = hotspot


def ppi_batch(binder: int, target: int, hotspots: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Design ``binder`` residues against a fixed helical target."""
    if hotspots > target:
        raise ValueError(f"--hotspot-residues {hotspots} exceeds the target length {target}")
    length = binder + target
    token_type = torch.zeros(length, dtype=torch.long, device=device)
    chain = torch.zeros(length, dtype=torch.long, device=device)
    chain[binder:] = 1
    features = _features(token_type, chain, torch.zeros(length, dtype=torch.long, device=device))
    target_ca = helical_ca(target)
    target_ca[:, 0] += 25
    pin_ca(features, binder, target_ca)
    pin_hotspots(features, binder, hotspots)
    encoded = torch.ones(1, length, dtype=torch.bool, device=device)
    encoded[:, binder:] = False
    features["can_be_encoded"] = encoded
    return features


def ligand_batch(protein: int, ligand_atoms: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Design ``protein`` residues around a fixed atomized ligand pose."""
    length = protein + ligand_atoms
    token_type = torch.zeros(length, dtype=torch.long, device=device)
    token_type[protein:] = ATOMIZED_TOKEN_TYPE
    chain = torch.zeros(length, dtype=torch.long, device=device)
    chain[protein:] = 1
    residue_type = torch.zeros(length, dtype=torch.long, device=device)
    residue_type[protein:] = ATOMIZED_RESIDUE_TYPE
    features = _features(token_type, chain, residue_type)
    # Atomized tokens occupy slot 1 and no other atom37 slot.
    features["atom37_mask"][0, protein:, :] = False
    features["atom37_mask"][0, protein:, ATOMIZED_COORDINATE_SLOT] = True
    index = torch.arange(ligand_atoms, dtype=torch.float32)
    pose = torch.stack((1.5 * index, torch.zeros(ligand_atoms), torch.zeros(ligand_atoms)), dim=-1)
    pin_ca(features, protein, pose)
    return features


def nucleic_batch(protein: int, dna: int, structure: bool, device: torch.device) -> dict[str, torch.Tensor]:
    """Design ``protein`` residues against a fixed DNA sequence.

    ``structure`` also fixes the DNA C1' coordinates. Otherwise the DNA
    sequence is fixed and its coordinates are designed.
    """
    length = protein + dna
    token_type = torch.zeros(length, dtype=torch.long, device=device)
    token_type[protein:] = DNA_TOKEN_TYPE
    chain = torch.zeros(length, dtype=torch.long, device=device)
    chain[protein:] = 1
    residue_type = torch.zeros(length, dtype=torch.long, device=device)
    bases = torch.tensor([_DNA_BASES[index % len(_DNA_BASES)] for index in range(dna)], dtype=torch.long)
    residue_type[protein:] = bases.to(device)
    features = _features(token_type, chain, residue_type)
    pin_sequence(features, protein, bases)
    if structure:
        dna_ca = helical_ca(dna)
        dna_ca[:, 0] += 20
        pin_ca(features, protein, dna_ca)
        encoded = torch.ones(1, length, dtype=torch.bool, device=device)
        encoded[:, protein:] = False
        features["can_be_encoded"] = encoded
    return features


def redesign_inputs(length: int, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """An all-alanine helix, in nanometres, for an existing-structure redesign."""
    token_type = torch.zeros(length, dtype=torch.long, device=device)
    residue_type = torch.full((length,), _ALA, dtype=torch.long, device=device)
    features = _features(token_type, token_type, residue_type)
    coordinates = torch.zeros(1, length, _ATOMS, 3, device=device)
    coordinates[0, :, _CA_SLOT] = helical_ca(length).to(device) * 0.1
    return features, coordinates


def design(model: RFProteina, features: dict[str, torch.Tensor], seed: int) -> tuple[RFProteinaPredictionOutput, float]:
    """Run one design and return the prediction with its elapsed seconds."""
    if not torch.cuda.is_available():
        raise RuntimeError("RFProteína design requires a CUDA device")
    start = time.perf_counter()
    with torch.inference_mode():
        prediction = model(features, seed=seed)
    torch.cuda.synchronize()
    return prediction, time.perf_counter() - start


def redesign(
    model: RFProteina,
    features: dict[str, torch.Tensor],
    coordinates_nm: torch.Tensor,
    partial_t: float,
    seed: int,
) -> tuple[RFProteinaPredictionOutput, float]:
    """Redesign an existing backbone from partial time ``partial_t``."""
    if not torch.cuda.is_available():
        raise RuntimeError("RFProteína design requires a CUDA device")
    partial = RFProteinaPartialSimulationConfig(partial_t_backbone=partial_t, partial_t_latent=partial_t)
    start = time.perf_counter()
    with torch.inference_mode():
        prediction = model.redesign(features, partial, coordinates=coordinates_nm, seed=seed)
    torch.cuda.synchronize()
    return prediction, time.perf_counter() - start


def sequence_of(prediction: RFProteinaPredictionOutput, count: int) -> str:
    residues = prediction.canonical_residue_types[0, :count].tolist()
    return "".join(_LETTERS[residue] if 0 <= residue < len(_LETTERS) else "X" for residue in residues)


def _positive(name: str, value: int) -> None:
    if value < 1:
        raise SystemExit(f"{name} must be positive")


def main() -> None:
    parser = argparse.ArgumentParser(description="Design one protein with RFProteína.")
    parser.add_argument("--flow", type=Path, required=True, help="Flow checkpoint.")
    parser.add_argument("--autoencoder", type=Path, required=True, help="Autoencoder checkpoint.")
    parser.add_argument("--length", type=int, default=128, help="Designed protein length in residues.")
    parser.add_argument("--steps", type=int, default=None, help="Sampling steps. Default: the model schedule.")
    parser.add_argument("--seed", type=int, default=0)
    subparsers = parser.add_subparsers(dest="workflow", required=True)

    subparsers.add_parser("unconditional", help="Design a chain with no coordinate or sequence conditions.")
    motif = subparsers.add_parser("motif", help="Design a chain around a fixed helical CA motif.")
    motif.add_argument("--motif-start", type=int, default=None, help="First motif residue. Default: length // 3.")
    motif.add_argument("--motif-length", type=int, default=16)

    ppi = subparsers.add_parser("ppi", help="Design a binder against a fixed helical target with hotspots.")
    ppi.add_argument("--target-length", type=int, default=48)
    ppi.add_argument("--hotspot-residues", type=int, default=8)

    ligand = subparsers.add_parser("small-molecule", help="Design a binder around a fixed atomized ligand pose.")
    ligand.add_argument("--ligand-atoms", type=int, default=8)

    nucleic = subparsers.add_parser("nucleic-acid", help="Design a binder against a fixed DNA sequence.")
    nucleic.add_argument("--dna-length", type=int, default=16)
    nucleic.add_argument("--structure", action="store_true", help="Also fix the DNA coordinates.")

    partial = subparsers.add_parser("partial-redesign", help="Redesign an existing all-alanine helix.")
    partial.add_argument("--partial-t", type=float, default=0.6, help="Start time in [0, 1) for backbone and latents.")

    args = parser.parse_args()
    _positive("--length", args.length)
    device = torch.device("cuda")
    model = load_model(args.flow, args.autoencoder, args.steps)
    coordinates_nm = None
    designed = args.length
    if args.workflow == "unconditional":
        features = chain_batch(args.length, device)
    elif args.workflow == "motif":
        _positive("--motif-length", args.motif_length)
        features = chain_batch(args.length, device)
        start = args.length // 3 if args.motif_start is None else args.motif_start
        pin_motif(features, start, helical_ca(args.motif_length))
    elif args.workflow == "ppi":
        _positive("--target-length", args.target_length)
        _positive("--hotspot-residues", args.hotspot_residues)
        features = ppi_batch(args.length, args.target_length, args.hotspot_residues, device)
    elif args.workflow == "small-molecule":
        _positive("--ligand-atoms", args.ligand_atoms)
        features = ligand_batch(args.length, args.ligand_atoms, device)
    elif args.workflow == "nucleic-acid":
        _positive("--dna-length", args.dna_length)
        features = nucleic_batch(args.length, args.dna_length, args.structure, device)
    else:
        if not 0.0 <= args.partial_t < 1.0:
            raise SystemExit("--partial-t must be in [0, 1)")
        features, coordinates_nm = redesign_inputs(args.length, device)
    if coordinates_nm is None:
        prediction, seconds = design(model, features, args.seed)
    else:
        prediction, seconds = redesign(model, features, coordinates_nm, args.partial_t, args.seed)
    print(f"{args.workflow}: {designed} residues in {seconds:.2f}s")
    print(sequence_of(prediction, designed))


if __name__ == "__main__":
    main()
