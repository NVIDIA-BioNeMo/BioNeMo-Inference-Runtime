from typing import Optional, Type

from .interface import AttentionBackend
from .vanilla import VanillaAttention


def get_attention_backend(backend_name: str) -> Type[AttentionBackend]:
    """Get the attention backend class based on the backend name."""
    if backend_name == "vanilla":
        return VanillaAttention
    else:
        raise ValueError(f"Invalid backend name: {backend_name}")


def create_attention(backend_name: str,
                     layer_idx: int,
                     num_heads: int,
                     head_dim: int,
                     num_kv_heads: Optional[int] = None) -> AttentionBackend:
    """Create an attention backend based on the backend name."""
    attn_cls = get_attention_backend(backend_name)
    return attn_cls(layer_idx, num_heads, head_dim, num_kv_heads)
