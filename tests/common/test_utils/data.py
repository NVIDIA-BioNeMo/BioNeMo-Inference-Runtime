import os

import numpy as np

from tensorrt_bionemo.data.schemas.basic import FoldingOutput

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_sample_folding_output() -> FoldingOutput:
    sample = np.load(os.path.join(_CURRENT_DIR, "sample_folding_output.npy"),
                     allow_pickle=True).item()

    return FoldingOutput(atom_positions=sample["atom_positions"],
                         residue_types=sample["residue_types"],
                         atom_mask=sample["atom_mask"],
                         residue_indices=sample["residue_indices"],
                         b_factors=sample["b_factors"],
                         chain_indices=sample["chain_indices"])
