from typing import Any, Optional

import numpy as np
from pydantic import BaseModel

import tensorrt_bionemo.pipeline.models.openfold2.const as rc
from tensorrt_bionemo.data.schemas import FoldingOutput
from tensorrt_bionemo.pipeline.base import PostProcessorBase


class PostProcessorConfig(BaseModel):
    subtract_plddt: bool = False
    multimer_ri_gap: int = 200


class PostProcessor(PostProcessorBase):

    def __init__(self, config: Optional[BaseModel] = None) -> None:
        super().__init__(config)
        if self.config is None:
            self.config = PostProcessorConfig()

    def __call__(self, batch: dict[str, Any],
                 output: dict[str, Any]) -> FoldingOutput:
        np_batch = {}
        # Get only the last element of the tensor, we don't need all the recycling steps
        for k in [
                "residue_index", "aatype", "asym_id", "final_atom_positions",
                "final_atom_mask"
        ]:
            if k in batch:
                np_batch[k] = np.array(batch[k][..., -1].cpu())
        plddt = output["plddt"].cpu().numpy()
        plddt_b_factors = np.repeat(plddt[..., None],
                                    rc.atom_type_num,
                                    axis=-1)

        if self.config.subtract_plddt:
            plddt_b_factors = 100 - plddt_b_factors

        # For multi-chain FASTAs
        ri = np_batch["residue_index"]
        chain_indices = (ri -
                         np.arange(ri.shape[0])) / self.config.multimer_ri_gap
        chain_indices = chain_indices.astype(np.int64)
        cur_chain = 0
        prev_chain_max = 0
        for i, c in enumerate(chain_indices):
            if c != cur_chain:
                cur_chain = c
                prev_chain_max = i + cur_chain * self.config.multimer_ri_gap

            np_batch["residue_index"][i] -= prev_chain_max

        if 'asym_id' in np_batch:
            chain_indices = np_batch["asym_id"] - 1
        else:
            chain_indices = np.zeros_like(np_batch["aatype"])

        return FoldingOutput(
            residue_types=np_batch["aatype"],
            atom_positions=output["final_atom_positions"].cpu().numpy(),
            atom_mask=output["final_atom_mask"].cpu().numpy(),
            residue_indices=np_batch["residue_index"] + 1,
            b_factors=plddt_b_factors,
            chain_indices=chain_indices)
