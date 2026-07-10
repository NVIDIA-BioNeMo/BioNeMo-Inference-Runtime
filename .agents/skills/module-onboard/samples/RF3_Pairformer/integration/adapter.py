"""
Drop-in adapter: wraps TRT-BNM PairformerModule to match BakerLab PairformerBlock forward signature.

Customer signature:  forward(S_I, Z_II, is_padding_I) -> (S_I, Z_II)
TRT-BNM signature:   forward(s, z, mask, pair_mask, ...) -> (s, z)
"""

import torch
import torch.nn as nn

from tensorrt_bionemo._torch.layers.transformers.pairformer import PairformerModule


class BakerLabPairformerAdapter(nn.Module):
    """Drop-in replacement for BakerLab's pairformer_stack (list of PairformerBlocks)."""

    def __init__(self, trtbnm_module: PairformerModule):
        super().__init__()
        self.module = trtbnm_module

    def forward(self, S_I, Z_II, is_padding_I):
        # Convert mask: customer is_padding_I (bool, True=pad) -> TRT-BNM mask (float, 1.0=valid)
        mask = (~is_padding_I).float()
        pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        s, z = self.module(S_I, Z_II, mask, pair_mask)
        return s, z
