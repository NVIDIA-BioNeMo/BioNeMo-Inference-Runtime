import pytest
import torch
from test_utils._plain_attn import (
    plain_pairwise_attention,
    plain_triangle_attention,
)

from tensorrt_bionemo._torch.attention_backend.interface import (
    AttentionMetadata,
    PredefinedAttentionBiases,
)
from tensorrt_bionemo._torch.attention_backend.vanilla import VanillaAttention


@pytest.mark.parametrize("seq_len", [16, 128])
@pytest.mark.parametrize("chunk_size", [None, 16, 32, 64, 96])
@pytest.mark.parametrize("has_biases", [False, True])
def test_triangle_attention(seq_len, chunk_size, has_biases):
    num_heads = 8
    head_dim = 32
    layer_idx = 0

    q = torch.randn(seq_len, seq_len, num_heads, head_dim)
    k = torch.randn(seq_len, seq_len, num_heads, head_dim)
    v = torch.randn(seq_len, seq_len, num_heads, head_dim)

    vanilla_attn = VanillaAttention(layer_idx,
                                    num_heads,
                                    head_dim,
                                    num_kv_heads=num_heads)
    biases = None
    if has_biases:
        biases = [
            torch.randn(seq_len, 1, 1, seq_len),
            torch.randn(1, num_heads, seq_len, seq_len)
        ]
    metadata = AttentionMetadata(chunk_size=chunk_size, chunk_dim=0)
    vanilla_out = vanilla_attn.forward(
        q,
        k,
        v,
        biases=biases,
        biases_type=PredefinedAttentionBiases.TRIANGLE,
        metadata=metadata)
    assert vanilla_out.shape == (seq_len, seq_len, num_heads * head_dim)
    plain_out = plain_triangle_attention(q, k, v, num_heads, head_dim, biases)
    assert vanilla_out.shape == plain_out.shape

    torch.testing.assert_close(vanilla_out, plain_out)


@pytest.mark.parametrize("batch_size", [16, 128])
@pytest.mark.parametrize("chunk_size", [None, 16, 32, 64, 96])
@pytest.mark.parametrize("has_biases", [False, True])
def test_pairwise_attention(batch_size, chunk_size, has_biases):
    num_heads = 8
    head_dim = 32
    layer_idx = 0
    q_size = 32
    kv_size = 128

    q = torch.randn(batch_size, q_size, num_heads, head_dim)
    k = torch.randn(batch_size, kv_size, num_heads, head_dim)
    v = torch.randn(batch_size, kv_size, num_heads, head_dim)

    vanilla_attn = VanillaAttention(layer_idx,
                                    num_heads,
                                    head_dim,
                                    num_kv_heads=num_heads)
    biases = None
    if has_biases:
        biases = [
            torch.randn(batch_size, 1, 1, kv_size),
            torch.randn(batch_size, num_heads, q_size, kv_size)
        ]
    metadata = AttentionMetadata(chunk_size=chunk_size, chunk_dim=0)
    vanilla_out = vanilla_attn.forward(
        q,
        k,
        v,
        biases=biases,
        biases_type=PredefinedAttentionBiases.PAIRWISE,
        metadata=metadata)
    assert vanilla_out.shape == (batch_size, q_size, num_heads * head_dim)
    plain_out = plain_pairwise_attention(q, k, v, num_heads, head_dim, biases)
    assert vanilla_out.shape == plain_out.shape
    torch.testing.assert_close(vanilla_out, plain_out)
