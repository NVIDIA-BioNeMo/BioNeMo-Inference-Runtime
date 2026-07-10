"""
Module swap: replace BakerLab Recycler's pairformer_stack with TRT-BNM PairformerModule.
"""

import torch

from tensorrt_bionemo._torch.layers.transformers.pairformer import PairformerModule

from .adapter import BakerLabPairformerAdapter
from .config import make_bakerlab_pairformer_config


def swap_pairformer_stack(
    recycler,
    customer_state_dict=None,
    num_blocks: int = 48,
    c_s: int = 384,
    c_z: int = 128,
    dtype: str = "bfloat16",
    device: str = "cuda",
    triangle_attention_backend: str = "CUEQUIV",
    pairwise_attention_backend: str = "SDPA",
):
    """Replace recycler.pairformer_stack with a TRT-BNM PairformerModule.

    Args:
        recycler: BakerLab Recycler nn.Module instance.
        customer_state_dict: Optional state_dict from the customer's pairformer_stack.
            If None, TRT-BNM module uses default (random) initialization.
        num_blocks: Number of pairformer blocks.
        c_s: Single representation dimension.
        c_z: Pair representation dimension.
        dtype: Weight dtype string.
        device: Target device.
        triangle_attention_backend: Triangle attention backend for Torch path.
        pairwise_attention_backend: Pairwise attention backend for Torch path.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from convert.convert_weights import convert_pairformer_stack_weights

    config = make_bakerlab_pairformer_config(
        num_blocks=num_blocks, c_s=c_s, c_z=c_z, dtype=dtype,
        triangle_attention_backend=triangle_attention_backend,
        pairwise_attention_backend=pairwise_attention_backend,
    )
    trtbnm_module = PairformerModule(config)

    if customer_state_dict is not None:
        converted = convert_pairformer_stack_weights(
            customer_state_dict, num_blocks=num_blocks)
        trtbnm_module.load_weights(converted)

    trtbnm_module = trtbnm_module.to(device)
    adapter = BakerLabPairformerAdapter(trtbnm_module)
    recycler.pairformer_stack = adapter
    return recycler
