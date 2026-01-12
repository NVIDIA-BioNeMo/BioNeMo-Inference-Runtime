from typing import Optional

import numpy as np
import torch

import tensorrt_bionemo.pipeline.models.openfold2.const as rc
from tensorrt_bionemo.configs.base import BaseConfig
from tensorrt_bionemo.pipeline.base import TransformBase


class CastTo64BitInts(TransformBase):

    def __call__(self, batch: dict[str,
                                   torch.Tensor]) -> dict[str, torch.Tensor]:
        # We keep all ints as int64
        for k, v in batch.items():
            if v.dtype == torch.int32:
                batch[k] = v.type(torch.int64)

        return batch


class CorrectMsaRestypes(TransformBase):

    def __call__(self, batch: dict[str,
                                   torch.Tensor]) -> dict[str, torch.Tensor]:
        """Correct MSA restype to have the same order as rc."""
        new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
        new_order = torch.tensor(
            [new_order_list] * batch["msa"].shape[1],
            device=batch["msa"].device,
        ).transpose(0, 1)
        batch["msa"] = torch.gather(new_order, 0, batch["msa"])

        perm_matrix = np.zeros((22, 22), dtype=np.float32)
        perm_matrix[range(len(new_order_list)), new_order_list] = 1.0

        for k in batch.keys():
            if "profile" in k:
                num_dim = batch[k].shape.as_list()[-1]
                assert num_dim in [
                    20,
                    21,
                    22,
                ], "num_dim for %s out of expected range: %s" % (k, num_dim)
                batch[k] = torch.dot(batch[k], perm_matrix[:num_dim, :num_dim])

        return batch


class SqueezeFeatures(TransformBase):

    def __call__(self, batch: dict[str,
                                   torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remove singleton and repeated dimensions in features."""
        batch["aatype"] = torch.argmax(batch["aatype"], dim=-1)
        for k in [
                "domain_name",
                "msa",
                "num_alignments",
                "seq_length",
                "sequence",
                "superfamily",
                "deletion_matrix",
                "resolution",
                "between_segment_residues",
                "residue_index",
                "template_all_atom_mask",
        ]:
            if k in batch:
                final_dim = batch[k].shape[-1]
                if isinstance(final_dim, int) and final_dim == 1:
                    if torch.is_tensor(batch[k]):
                        batch[k] = torch.squeeze(batch[k], dim=-1)
                    else:
                        batch[k] = np.squeeze(batch[k], axis=-1)

        for k in ["seq_length", "num_alignments"]:
            if k in batch:
                batch[k] = batch[k][0]

        return batch


class RandomlyReplaceMsaWithUnknown(TransformBase):

    def __init__(self,
                 config: Optional[BaseConfig] = None,
                 replace_proportion: float = 0.):
        super().__init__(config)
        self.replace_proportion = replace_proportion

    def __call__(self, batch: dict[str,
                                   torch.Tensor]) -> dict[str, torch.Tensor]:
        """Replace a portion of the MSA with 'X'."""
        msa_mask = torch.rand(batch["msa"].shape) < self.replace_proportion
        x_idx = 20
        gap_idx = 21
        msa_mask = torch.logical_and(msa_mask, batch["msa"] != gap_idx)
        batch["msa"] = torch.where(msa_mask,
                                   torch.ones_like(batch["msa"]) * x_idx,
                                   batch["msa"])
        aatype_mask = torch.rand(
            batch["aatype"].shape) < self.replace_proportion

        batch["aatype"] = torch.where(
            aatype_mask,
            torch.ones_like(batch["aatype"]) * x_idx,
            batch["aatype"],
        )
        return batch


class FixTemplatesAatype(TransformBase):

    def is_enabled(self) -> bool:
        return self.config.enable_template

    def __call__(self, batch: dict[str,
                                   torch.Tensor]) -> dict[str, torch.Tensor]:
        # Map one-hot to indices
        num_templates = batch["template_aatype"].shape[0]
        batch["template_aatype"] = torch.argmax(batch["template_aatype"],
                                                dim=-1)
        # Map hhsearch-aatype to our aatype.
        new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
        new_order = torch.tensor(
            new_order_list,
            dtype=torch.int64,
            device=batch["template_aatype"].device,
        ).expand(num_templates, -1)
        batch["template_aatype"] = torch.gather(new_order,
                                                1,
                                                index=batch["template_aatype"])

        return batch
