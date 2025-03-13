from copy import deepcopy
from dataclasses import dataclass

import pytest
import torch
import transformers
from test_utils._plain_attn import RefTriangleAttention

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.model_config import ModelConfig
from tensorrt_bionemo._torch.modules.attention import TriangleAttention

_MOCK_MODEL_CONFIG = {
    "architectures": ["triangle-attention"],
    "torch_dtype": "float32",
}


@dataclass(kw_only=True, frozen=True)
class Scenario:
    backend: str
    seq_len: int = 16
    hidden_size: int = 128
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    gating: bool = True
    bias: bool = False
    chunk_size: int = None
    chunk_dim: int = None
    torch_dtype: str = "float32"


@pytest.mark.parametrize("s", [
    Scenario(backend="VANILLA"),
    Scenario(backend="VANILLA", torch_dtype="bfloat16"),
    Scenario(backend="VANILLA", torch_dtype="float16"),
])
def test_triangle_attention_backend(s: Scenario):
    metadata_cls = get_attention_backend(s.backend).Metadata
    config_dict = deepcopy(_MOCK_MODEL_CONFIG)
    config_dict["torch_dtype"] = s.torch_dtype
    model_config = ModelConfig(
        pretrained_config=transformers.PretrainedConfig.from_dict(config_dict),
        attn_backend=s.backend,
        skip_create_weights=False,
    )
    dtype = model_config.pretrained_config.torch_dtype
    device = torch.device('cuda')

    ref_attn = RefTriangleAttention.load_weights(no_heads=s.num_attention_heads)
    ref_attn.to(device)
    ref_attn = ref_attn

    qkv_weights = [
        {
            "weight": ref_attn.linear_q.weight.data.to(dtype),
            "bias": ref_attn.linear_q.bias.data.to(dtype) if s.bias else None
        },
        {
            "weight": ref_attn.linear_k.weight.data.to(dtype),
            "bias": ref_attn.linear_k.bias.data.to(dtype) if s.bias else None
        },
        {
            "weight": ref_attn.linear_v.weight.data.to(dtype),
            "bias": ref_attn.linear_v.bias.data.to(dtype) if s.bias else None
        },
    ]
    o_proj_weights = [{
        "weight":
        ref_attn.linear_o.weight.data.to(dtype),
        "bias":
        ref_attn.linear_o.bias.data.to(dtype) if s.bias else None
    }]
    g_proj_weights = [{
        "weight":
        ref_attn.linear_g.weight.data.to(dtype),
        "bias":
        ref_attn.linear_g.bias.data.to(dtype) if s.bias else None
    }]

    attn = TriangleAttention(
        layer_idx=0,
        hidden_size=s.hidden_size,
        num_attention_heads=s.num_attention_heads,
        num_key_value_heads=s.num_key_value_heads,
        gating=s.gating,
        bias=s.bias,
        dtype=dtype,
        config=model_config,
    )
    attn.qkv_proj.load_weights(qkv_weights)
    attn.o_proj.load_weights(o_proj_weights)
    if s.gating:
        attn.g_proj.load_weights(g_proj_weights)
    attn.to(device)
    attn_metadata = metadata_cls(chunk_size=s.chunk_size, chunk_dim=s.chunk_dim)
    hidden_states = torch.randn(s.seq_len,
                                s.seq_len,
                                s.hidden_size,
                                dtype=torch.float32,
                                device=device)
    biases = [
        torch.randn(s.seq_len,
                    1,
                    1,
                    s.seq_len,
                    dtype=torch.float32,
                    device=device),
        torch.randn(1,
                    s.num_attention_heads,
                    s.seq_len,
                    s.seq_len,
                    dtype=torch.float32,
                    device=device)
    ]

    with torch.inference_mode():
        ref_output_float = ref_attn(hidden_states, hidden_states, biases=biases)
        hidden_states = hidden_states.to(dtype)
        biases = [bias.to(dtype) for bias in biases]
        ref_attn = ref_attn.to(dtype)
        ref_output = ref_attn(hidden_states, hidden_states, biases=biases)
        output = attn(hidden_states, biases=biases, attn_metadata=attn_metadata)

    assert output.shape == ref_output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-3)
    else:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() -
                                          ref_output_float))

        assert diff0_max <= (diff1_max + 0.5)
        assert diff0_mean <= (diff1_mean + 0.01)
