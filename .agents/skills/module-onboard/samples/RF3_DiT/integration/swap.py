"""
Module swap: replace BakerLab DiffusionModule's diffusion_transformer
with TRT-BNM DiffusionTransformerLayer stack.
"""

import sys
import os

import torch
import torch.nn as nn

from tensorrt_bionemo._torch.layers.transformers.diffusion_transformer import (
    DiffusionTransformerLayer,
)

from .adapter import BakerLabDiTStackAdapter
from .config import make_bakerlab_dit_config

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from convert.convert_weights import convert_dit_block_weights


def swap_diffusion_transformer(
    diffusion_module,
    customer_state_dict=None,
    num_blocks: int = 24,
    c_token: int = 384,
    c_s: int = 384,
    c_tokenpair: int = 128,
    num_heads: int = 16,
    dtype: str = "bfloat16",
    device: str = "cuda",
    pairwise_attention_backend: str = "SDPA",
):
    """Replace diffusion_module.diffusion_transformer with TRT-BNM equivalent.

    Args:
        diffusion_module: BakerLab DiffusionModule instance.
        customer_state_dict: Optional state_dict from customer's DiffusionTransformer.
            If None, TRT-BNM modules use default initialization.
    """
    config = make_bakerlab_dit_config(
        num_blocks=num_blocks, c_token=c_token, c_s=c_s,
        c_tokenpair=c_tokenpair, num_heads=num_heads, dtype=dtype,
        pairwise_attention_backend=pairwise_attention_backend,
    )
    torch_dtype = config.torch_dtype

    layers = nn.ModuleList()
    for i in range(num_blocks):
        layer = DiffusionTransformerLayer(
            layer_idx=i, num_heads=num_heads, dim=c_token,
            dim_single_cond=c_s, dim_pairwise=c_tokenpair,
            dtype=torch_dtype, attn_backend=pairwise_attention_backend,
            initial_norm=True, bias_proj=True, pair_norm=True,
            attn_output_gate=False, conditioned_transition_using_silu=True,
        )
        if customer_state_dict is not None:
            converted = convert_dit_block_weights(
                customer_state_dict, f"blocks.{i}", "", c_token, num_heads)
            layer.load_state_dict({k: v.to(device) for k, v in converted.items()})
        layers.append(layer)

    layers = layers.to(device).eval()
    adapter = BakerLabDiTStackAdapter(layers)
    diffusion_module.diffusion_transformer = adapter
    return diffusion_module
