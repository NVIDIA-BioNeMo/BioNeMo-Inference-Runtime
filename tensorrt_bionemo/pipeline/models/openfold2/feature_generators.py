from typing import Any, Optional

import torch

import tensorrt_bionemo.pipeline.models.openfold2.const as rc
from tensorrt_bionemo.configs.base import BaseConfig
from tensorrt_bionemo.pipeline.base import FeatureGeneratorBase

from .common import atom37_to_torsion_angles, make_one_hot, pseudo_beta_fn


class UseClampedFape(FeatureGeneratorBase):

    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        feats = {}
        feats["use_clamped_fape"] = torch.full(
            size=[self.config.max_recycling_iters + 1],
            fill_value=0.0,
            dtype=torch.float32,
        )
        return feats


class MakeSequenceMask(FeatureGeneratorBase):

    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        feats = {}
        feats["seq_mask"] = torch.ones(batch["aatype"].shape,
                                       dtype=torch.float32)
        return feats


class MakeMsaMask(FeatureGeneratorBase):

    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        feats = {}
        feats["msa_mask"] = torch.ones(batch["msa"].shape, dtype=torch.float32)
        feats["msa_row_mask"] = torch.ones((batch["msa"].shape[0]),
                                           dtype=torch.float32)
        return feats


class MakeTemplateMask(FeatureGeneratorBase):

    def is_enabled(self) -> bool:
        return self.config.enable_template

    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        feats = {}
        feats["template_mask"] = torch.ones(batch["template_aatype"].shape[0],
                                            dtype=torch.float32)
        return feats


class MakeTemplatePseudoBeta(FeatureGeneratorBase):

    def is_enabled(self) -> bool:
        return self.config.enable_template

    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Create pseudo-beta (alpha for glycine) position and mask."""
        feats = {}
        feats["template_pseudo_beta"], feats[
            "template_pseudo_beta_mask"] = pseudo_beta_fn(
                batch["template_aatype"],
                batch["template_all_atom_positions"],
                batch["template_all_atom_mask"],
            )
        return feats


class Atom37ToTorsionAngles(FeatureGeneratorBase):

    def __init__(self, config: Optional[BaseConfig] = None, prefix: str = ""):
        super().__init__(config)
        self.prefix = prefix

    def is_enabled(self) -> bool:
        return self.config.enable_template and self.config.use_template_torsion_angles

    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        return atom37_to_torsion_angles(batch, prefix=self.prefix)


class MakeAtom14Masks(FeatureGeneratorBase):

    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Construct denser atom positions (14 dimensions instead of 37)."""
        feats = {}
        restype_atom14_to_atom37 = []
        restype_atom37_to_atom14 = []
        restype_atom14_mask = []

        for rt in rc.restypes:
            atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
            restype_atom14_to_atom37.append([
                (rc.atom_order[name] if name else 0) for name in atom_names
            ])
            atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
            restype_atom37_to_atom14.append([
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in rc.atom_types
            ])

            restype_atom14_mask.append([(1.0 if name else 0.0)
                                        for name in atom_names])

        # Add dummy mapping for restype 'UNK'
        restype_atom14_to_atom37.append([0] * 14)
        restype_atom37_to_atom14.append([0] * 37)
        restype_atom14_mask.append([0.0] * 14)

        restype_atom14_to_atom37 = torch.tensor(
            restype_atom14_to_atom37,
            dtype=torch.int32,
            device=batch["aatype"].device,
        )
        restype_atom37_to_atom14 = torch.tensor(
            restype_atom37_to_atom14,
            dtype=torch.int32,
            device=batch["aatype"].device,
        )
        restype_atom14_mask = torch.tensor(
            restype_atom14_mask,
            dtype=torch.float32,
            device=batch["aatype"].device,
        )
        protein_aatype = batch['aatype'].to(torch.long)

        # create the mapping for (residx, atom14) --> atom37, i.e. an array
        # with shape (num_res, 14) containing the atom37 indices for this protein
        residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
        residx_atom14_mask = restype_atom14_mask[protein_aatype]

        feats["atom14_atom_exists"] = residx_atom14_mask
        feats["residx_atom14_to_atom37"] = residx_atom14_to_atom37.long()

        # create the gather indices for mapping back
        residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype]
        feats["residx_atom37_to_atom14"] = residx_atom37_to_atom14.long()

        # create the corresponding mask
        restype_atom37_mask = torch.zeros([21, 37],
                                          dtype=torch.float32,
                                          device=batch["aatype"].device)
        for restype, restype_letter in enumerate(rc.restypes):
            restype_name = rc.restype_1to3[restype_letter]
            atom_names = rc.residue_atoms[restype_name]
            for atom_name in atom_names:
                atom_type = rc.atom_order[atom_name]
                restype_atom37_mask[restype, atom_type] = 1

        residx_atom37_mask = restype_atom37_mask[protein_aatype]
        feats["atom37_atom_exists"] = residx_atom37_mask

        return feats


class MakeHhblitsProfile(FeatureGeneratorBase):

    def __call__(self, batch: dict[str, torch.Tensor],
                 context: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Compute the HHblits MSA profile if not already present."""
        feats = {}
        if "hhblits_profile" in batch:
            return feats

        # Compute the profile for every residue (over all MSA sequences).
        msa_one_hot = make_one_hot(batch["msa"], 22)

        feats["hhblits_profile"] = torch.mean(msa_one_hot, dim=0)
        return feats
