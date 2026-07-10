"""TRT-BNM PairformerConfig for BakerLab RF3 Pairformer hyperparameters."""

from tensorrt_bionemo.configs.modules import PairformerConfig


def make_bakerlab_pairformer_config(
    num_blocks: int = 48,
    c_s: int = 384,
    c_z: int = 128,
    num_heads: int = 16,
    pairwise_head_width: int = 32,
    pairwise_num_heads: int = 4,
    dtype: str = "bfloat16",
    # Torch backend defaults
    triangle_attention_backend: str = "CuTeDSL",
    pairwise_attention_backend: str = "CuTeDSL",
) -> PairformerConfig:
    return PairformerConfig(
        num_blocks=num_blocks,
        token_s=c_s,
        token_z=c_z,
        num_heads=num_heads,
        pairwise_head_width=pairwise_head_width,
        pairwise_num_heads=pairwise_num_heads,
        dtype=dtype,
        triangle_attention_backend=triangle_attention_backend,
        pairwise_attention_backend=pairwise_attention_backend,
        attention_initial_norm=True,
        version="v1",
    )
