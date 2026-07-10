"""TRT-BNM DiffusionTransformerConfig for BakerLab RF3 DiffusionTransformer hyperparameters."""

from tensorrt_bionemo.configs.modules import DiffusionTransformerConfig


def make_bakerlab_dit_config(
    num_blocks: int = 24,
    c_token: int = 384,
    c_s: int = 384,
    c_tokenpair: int = 128,
    num_heads: int = 16,
    dtype: str = "bfloat16",
    # Torch backend defaults
    pairwise_attention_backend: str = "CuTeDSL",
) -> DiffusionTransformerConfig:
    return DiffusionTransformerConfig(
        num_blocks=num_blocks,
        num_heads=num_heads,
        dim=c_token,
        dim_single_cond=c_s,
        dim_pairwise=c_tokenpair,
        dtype=dtype,
        bias_proj=True,
        attention_initial_norm=False,
        post_layer_norm=False,
        conditioned_transition_using_silu=True,
        attn_output_gate=False,
        pairwise_attention_backend=pairwise_attention_backend,
    )
