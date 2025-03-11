import enum
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, Union

import torch
from tensorrt_llm.mapping import Mapping


@dataclass(kw_only=True)
class AttentionMetadata:
    """
    Metadata for multi-head attention layer.
    """

    max_num_tokens: int
    mapping: Optional[Mapping] = None
    chunk_dim: Optional[int] = None
    chunk_size: Optional[int] = None


TMetadata = TypeVar("TMetadata", bound=AttentionMetadata)


class PredefinedAttentionBiases(str, enum.Enum):
    """
    Predefined attention mask types

    Attributes:
        TRIANGLE: Use bias for triangular attention.
        PAIRWISE:  Use bias for pairwise attention.
    """

    TRIANGLE = "triangle"
    PAIRWISE = "pairwise"


# May extend to custom attention mask type
AttentionBiases = Union[PredefinedAttentionBiases]


class AttentionBackend(Generic[TMetadata]):

    def __init__(self,
                 layer_idx: int,
                 num_heads: int,
                 head_dim: int,
                 num_kv_heads: Optional[int] = None):
        """
        Initialize the attention backend.
        Args:
            layer_idx (int): The index of the attention layer.
            num_heads (int): The number of attention heads.
            head_dim (int): The dimension of each attention head.
            num_kv_heads (Optional[int]): The number of key-value heads.
        """
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: Optional[list[torch.Tensor]] = None,
        metadata: TMetadata = None,
        attention_biases: Optional[AttentionBiases] = PredefinedAttentionBiases.
        TRIANGLE,
        **kwargs,
    ) -> torch.Tensor:
        """
        Perform the attention operation.
        Args:
            q (torch.Tensor): The query tensor. Shape [I, s_q, h_q*d]
            k (torch.Tensor): The key tensor. Shape [I, s_kv, h_kv*d]
            v (torch.Tensor): The value tensor. Shape [I, s_kv, h_kv*d]
            biases (Optional[list[torch.Tensor]]): The biases for the attention layer.
            metadata (AttentionMetadata): The metadata for the attention layer.
            attention_biases (Optional[AttentionBiases]): The type of attention biases to use.
            **kwargs: Additional keyword arguments.
        Returns:
            torch.Tensor: The output tensor.
        """
        raise NotImplementedError("Subclasses must implement this method.")
