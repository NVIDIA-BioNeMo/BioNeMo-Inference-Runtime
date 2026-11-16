# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RFProteína partial-autoencoder decoder.

Turns the sampler's product-space state -- backbone CA positions plus per-token
local latents -- back into an all-atom structure and a sequence. Mirrors OSS
``rfproteina.partial_autoencoder.decoder.DecoderTransformer``.

The decoder needs no atomworks featurization: its whole input is
``ca_coors_nm`` (3) and ``z_latent`` (8) on the sequence track, and binned
relative sequence separation plus binned CA pair distances on the pair track,
all derived here from the state itself.
"""

from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from bionemo_ir._torch.attention_backend.interface import AttentionMetadata
from bionemo_ir._torch.attention_backend.utils import precompute_single_masks
from bionemo_ir._torch.layers.transformers.diffusion_transformer import (
    build_bias_mega_offset,
    build_bias_mega_weight,
    precompute_pair_biases,
)
from bionemo_ir._torch.layers.transformers.local_latents_transformer import RFProteinaTrunkLayer
from bionemo_ir._torch.layers.transition import Transition
from bionemo_ir.configs import BaseConfig
from bionemo_ir.runtime.buffers import PreallocatedBuffers
from bionemo_ir.utils import str_dtype_to_torch

#: Index of the CA atom in the atom37 layout.
ATOM37_CA_INDEX = 1
#: Number of atom slots in the atom37 layout.
ATOM37_SIZE = 37

#: Largest fused sequence-separation x distance lookup table to build, in
#: bytes. The released decoder needs 127 * 30 * 256 * 4 B ~= 3.9 MiB; the
#: budget only exists so an unusual bin count falls back instead of allocating
#: something enormous.
_MAX_FUSED_TABLE_BYTES = 128 * 1024**2

#: Residue classes of the protein-only decoder and encoder, which have no nucleic-acid classes.
LEGACY_RESIDUE_TYPES = 20


@cache
def _allowed_from_classes(
    device: torch.device | None,
    n_residue_types: int,
    allowed_classes: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    """Build the per-token-kind table from the classes each kind may decode as."""
    allowed = torch.zeros((len(allowed_classes), n_residue_types), dtype=torch.bool, device=device)
    for kind, classes in enumerate(allowed_classes):
        if classes:
            allowed[kind, list(classes)] = True
    return allowed


def get_allowed_residue_type_mask(
    device: torch.device | None = None,
    n_residue_types: int = 20,
    allowed_classes: tuple[tuple[int, ...], ...] | None = None,
) -> torch.Tensor | None:
    """Return decoder classes allowed for each token type.

    Token types are protein (0), RNA (1), DNA (2), atomized (3), and unknown
    (4); a wider vocabulary adds glycan (5) and NCAA (6) and supplies its own
    table through ``allowed_classes``, which weight conversion resolves from
    :mod:`bionemo_ir.data.schemas.vocabulary`. The legacy 20-class protein
    decoder predates this contract and returns ``None`` so its logits remain
    unchanged.

    Restricting the logits matters for more than tidiness: a class the token
    kind never represents can carry an empty atom table, and a token that
    decodes to one emits no atoms at all.
    """
    if allowed_classes is not None:
        return _allowed_from_classes(device, n_residue_types, allowed_classes)
    if n_residue_types != 33:
        return None

    allowed = torch.zeros((5, n_residue_types), dtype=torch.bool, device=device)
    allowed[0, 1:22] = True
    allowed[1, 22:27] = True
    allowed[2, 27:32] = True
    allowed[3, 32] = True
    allowed[4, 1:33] = True
    return allowed


def pair_distance_bin_indices(coords: torch.Tensor, min_dist: float, max_dist: float, num_bins: int) -> torch.Tensor:
    """Bin indices of pairwise distances of ``coords`` ``[B, N, 3]``.

    Returns ``[B, N, N]`` in ``[0, num_bins)``; the outer bins are open, so
    they capture distances beyond the limits.
    """
    distances = torch.norm(coords.unsqueeze(2) - coords.unsqueeze(1), dim=-1)
    bin_limits = torch.linspace(min_dist, max_dist, num_bins - 1, device=coords.device, dtype=coords.dtype)
    return torch.bucketize(distances, bin_limits)


def sequence_separation_bin_indices(token_index: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Bin indices of signed separation ``i - j`` of ``token_index`` ``[B, N]``.

    ``num_bins`` must be odd so the bin limits straddle zero symmetrically.
    Returns ``[B, N, N]`` in ``[0, num_bins)``.
    """
    if num_bins % 2 == 0:
        raise ValueError(f"relative sequence separation needs an odd bin count, got {num_bins}")
    separation = token_index.unsqueeze(-1) - token_index.unsqueeze(-2)
    limit = num_bins / 2.0 - 1
    bin_limits = torch.linspace(-limit, limit, num_bins - 1, device=token_index.device, dtype=torch.float32)
    return torch.bucketize(separation.to(torch.float32), bin_limits)


class RFProteinaLatentDecoder(nn.Module):
    """Decode ``(bb_ca, local_latents)`` into atom37 coordinates and sequence.

    The trunk reuses :class:`RFProteinaTrunkLayer` in its ``layer_norm``
    variant; unlike the velocity trunk there are no pair updates and no
    triangle multiplication, and the conditioning track is a learned constant
    (the released autoencoder is unconditional, so its OSS condition factory
    emits zeros and carries no weights).

    Because the pair representation never changes, every layer's pair bias is
    a projection of the same tensor, so they are all computed up front with a
    single normalization and one mega-GEMM (as in the OpenFold3 and Protenix
    token transformers). Set ``config.precompute_bias=False`` to project per
    layer instead.
    """

    def __init__(self, config: BaseConfig) -> None:
        """
        Args:
            config: a :class:`~bionemo_ir.configs.BaseConfig` carrying the
                decoder fields (see
                ``bionemo_ir.models.rfproteina.config.RFProteinaDecoderConfig``).
        """
        super().__init__()
        self.config = config
        self.dtype = config.torch_dtype
        self.transformer_dtype = str_dtype_to_torch(config.transformer_dtype)
        self.latent_dim = config.latent_dim
        self.num_residue_types = config.num_residue_types
        self.pair_distance_bins = config.pair_distance_bins
        self.pair_distance_min = config.pair_distance_min
        self.pair_distance_max = config.pair_distance_max
        self.sequence_separation_bins = config.sequence_separation_bins
        self.absolute_coordinates = config.absolute_coordinates

        self.init_repr_factory = _LinearFactory(config.single_input_dim, config.token_dim, self.dtype)
        if config.input_schema == "unified":
            self.token_type_embedding = nn.Embedding(
                config.num_token_kinds, config.token_type_embedding_dim, dtype=self.dtype
            )
            self.conditional_residue_embedding = nn.Embedding(
                config.num_conditional_residue_types,
                config.num_conditional_residue_types,
                dtype=self.dtype,
            )
        else:
            self.token_type_embedding = None
            self.conditional_residue_embedding = None
        self.pair_rep_factory = (
            _LinearFactory(config.pair_input_dim, config.pair_dim, self.dtype) if config.use_pair_bias else None
        )

        cond_hidden = int(config.condition_dim * config.condition_transition_expansion)
        self.transition_c_1 = Transition(
            config.condition_dim,
            cond_hidden,
            dtype=self.dtype,
            normalize=False,
        )
        self.transition_c_2 = Transition(
            config.condition_dim,
            cond_hidden,
            dtype=self.dtype,
            normalize=False,
        )

        self.transformer_layers = nn.ModuleList(
            [
                RFProteinaTrunkLayer(
                    layer_idx=i,
                    token_dim=config.token_dim,
                    pair_dim=config.pair_dim,
                    condition_dim=config.condition_dim,
                    num_heads=config.num_heads,
                    transition_expansion=config.transition_expansion,
                    backend=config.pairwise_attention_backend,
                    mask_inf=config.mask_inf,
                    use_qk_norm=config.use_qk_norm,
                    eps=config.norm_epsilon,
                    dtype=self.transformer_dtype,
                    norm_type="layer_norm",
                    cond_norm_bias=True,
                    use_pair_bias=config.use_pair_bias,
                    mask_left_aligned=True,
                    sanitize_masked_inputs=False,
                    require_inference_mode=False,
                )
                for i in range(config.num_layers)
            ]
        )

        self.logit_linear = nn.Sequential(
            nn.LayerNorm(config.token_dim, eps=config.norm_epsilon, dtype=self.dtype),
            nn.Linear(config.token_dim, config.num_residue_types, bias=False, dtype=self.dtype),
        )
        self.struct_linear = nn.Sequential(
            nn.LayerNorm(config.token_dim, eps=config.norm_epsilon, dtype=self.dtype),
            nn.Linear(config.token_dim, ATOM37_SIZE * 3, bias=False, dtype=self.dtype),
        )
        self.atom_presence_linear = (
            nn.Sequential(
                nn.LayerNorm(config.token_dim, eps=config.norm_epsilon, dtype=self.dtype),
                nn.Linear(config.token_dim, ATOM37_SIZE, bias=False, dtype=self.dtype),
            )
            if config.use_atom_presence
            else None
        )
        self._allowed_residue_classes = config.allowed_residue_classes

        self.precompute_bias = config.precompute_bias
        self._norm_epsilon = config.norm_epsilon
        self._num_heads = config.num_heads
        self._bias_pad_multiple = 8 if config.pairwise_attention_backend == "CuTeDSL" else -1

    def _precompute_all_biases(self, pair: torch.Tensor) -> list[torch.Tensor]:
        """Project every layer's pair bias from the shared representation.

        Rebuilt per call so weight loading and in-place edits are observed;
        the fold is a few small matmuls against the projection weights.
        """
        return precompute_pair_biases(
            pair.to(self.transformer_dtype),
            build_bias_mega_weight(self.transformer_layers, attn_attr="mha"),
            len(self.transformer_layers),
            self._num_heads,
            self._norm_epsilon,
            self._bias_pad_multiple,
            bias_offset=build_bias_mega_offset(self.transformer_layers, attn_attr="mha"),
        )

    def build_representations(
        self,
        backbone_ca: torch.Tensor,
        local_latents: torch.Tensor,
        mask: torch.Tensor,
        token_index: torch.Tensor | None = None,
        token_type: torch.Tensor | None = None,
        conditional_residue_type: torch.Tensor | None = None,
        conditional_residue_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build the single and pair representations from the state.

        Args:
            backbone_ca: CA coordinates in nanometres ``[B, N, 3]``.
            local_latents: Per-token latents ``[B, N, latent_dim]``.
            mask: Boolean valid-token mask ``[B, N]``.
            token_index: Optional 1-based token positions ``[B, N]`` used for
                the sequence-separation feature. Defaults to ``1..N``.

        Returns:
            ``(single, pair)`` of shape ``[B, N, token_dim]`` and
            optional ``[B, N, N, pair_dim]``.
        """
        return self._build_representations(
            backbone_ca,
            local_latents,
            mask,
            mask.unsqueeze(-1).to(self.dtype),
            token_index,
            token_type,
            conditional_residue_type,
            conditional_residue_mask,
        )

    def _build_representations(
        self,
        backbone_ca: torch.Tensor,
        local_latents: torch.Tensor,
        mask: torch.Tensor,
        token_valid: torch.Tensor,
        token_index: torch.Tensor | None,
        token_type: torch.Tensor | None,
        conditional_residue_type: torch.Tensor | None,
        conditional_residue_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Shared body taking the already-computed ``[B, N, 1]`` token mask."""
        batch, num_tokens = mask.shape
        if token_index is None:
            token_index = torch.arange(1, num_tokens + 1, device=mask.device, dtype=torch.long)
            token_index = token_index.unsqueeze(0).expand(batch, -1)

        single_features = [backbone_ca, local_latents]
        if self.config.input_schema == "unified":
            if token_type is None:
                token_type = torch.zeros_like(mask, dtype=torch.long)
            if token_type.shape != mask.shape:
                raise ValueError(f"token_type must have shape {tuple(mask.shape)}, got {tuple(token_type.shape)}")
            token_type = token_type.to(device=mask.device, dtype=torch.long)
            invalid_token_type = (token_type < 0) | (token_type >= self.token_type_embedding.num_embeddings)
            if torch.any(invalid_token_type):
                values = torch.unique(token_type[invalid_token_type]).detach().cpu().tolist()
                raise ValueError(f"token_type contains invalid values {values}")
            single_features.append(self.token_type_embedding(token_type))

            if conditional_residue_type is None or conditional_residue_mask is None:
                conditioned_sequence = torch.zeros(
                    *mask.shape,
                    self.config.num_conditional_residue_types,
                    device=mask.device,
                    dtype=self.dtype,
                )
            else:
                if conditional_residue_type.shape != mask.shape or conditional_residue_mask.shape != mask.shape:
                    raise ValueError("conditional residue tensors must have shape [B, N]")
                conditional_residue_mask = conditional_residue_mask.to(device=mask.device, dtype=torch.bool)
                conditional_residue_type = conditional_residue_type.to(device=mask.device, dtype=torch.long)
                conditional_residue_type = conditional_residue_type * conditional_residue_mask
                conditioned_sequence = self.conditional_residue_embedding(conditional_residue_type)
                conditioned_sequence = conditioned_sequence * conditional_residue_mask.unsqueeze(-1).to(self.dtype)
            single_features.append(conditioned_sequence)

        single_features = torch.cat(single_features, dim=-1)
        single = self.init_repr_factory(single_features.to(self.dtype), token_valid)
        pair = self._build_pair_representation(backbone_ca, mask, token_index) if self.config.use_pair_bias else None
        return single, pair

    def _build_pair_representation(
        self, backbone_ca: torch.Tensor, mask: torch.Tensor, token_index: torch.Tensor
    ) -> torch.Tensor:
        """Gather the pair representation from its two binned features.

        Both pair features are one-hot, so rather than building them (two
        ``[B, N, N, bins]`` one-hots, their concatenation, and a masked copy,
        all before the projection) this looks the projection up directly.
        Masked pairs select a zero row, which replaces the output masking too.
        """
        if self.pair_rep_factory is None:
            raise RuntimeError("pair feature projection requires use_pair_bias=True")
        widths = (self.sequence_separation_bins, self.pair_distance_bins)
        separation = sequence_separation_bin_indices(token_index, widths[0])
        distance = pair_distance_bin_indices(backbone_ca, self.pair_distance_min, self.pair_distance_max, widths[1])
        pair_valid = mask.unsqueeze(-1) & mask.unsqueeze(-2)

        fused = self.pair_rep_factory.fused_one_hot_table(widths)
        if fused is not None:
            index = separation * widths[1] + distance
            return F.embedding(index.where(pair_valid, widths[0] * widths[1]), fused)

        separation_table, distance_table = self.pair_rep_factory.one_hot_tables(widths)
        pair = F.embedding(separation.where(pair_valid, widths[0]), separation_table)
        pair += F.embedding(distance.where(pair_valid, widths[1]), distance_table)
        return pair

    def allowed_residue_types(self, device: torch.device | None = None) -> torch.Tensor | None:
        """Return the classes each token kind may decode as, ``[kinds, num_residue_types]``.

        ``None`` when the decoder does not restrict them.
        """
        return get_allowed_residue_type_mask(device, self.num_residue_types, self._allowed_residue_classes)

    def forward(
        self,
        backbone_ca: torch.Tensor,
        local_latents: torch.Tensor,
        mask: torch.Tensor,
        token_index: torch.Tensor | None = None,
        metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
        token_type: torch.Tensor | None = None,
        conditional_residue_type: torch.Tensor | None = None,
        conditional_residue_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode a product-space state.

        Args:
            backbone_ca: CA coordinates in nanometres ``[B, N, 3]``.
            local_latents: Per-token latents ``[B, N, latent_dim]``.
            mask: Boolean valid-token mask ``[B, N]``.
            token_index: Optional 1-based token positions ``[B, N]``.
            metadata: Optional pairwise-attention metadata.
            buffers: Optional preallocated Torch buffers.
            token_type: Optional token-kind IDs ``[B, N]``. For the unified
                33-class decoder, this constrains each token to the matching
                protein, RNA, DNA, atomized, or unknown output classes.
            conditional_residue_type: Optional conditioned residue classes
                ``[B, N]`` of the unified decoder.
            conditional_residue_mask: Which tokens ``conditional_residue_type``
                holds a condition for, ``[B, N]``.
        Returns:
            Coordinates [B, N, 37, 3] in nm, sequence logits and their argmax,
            and raw presence logits [B, N, 37] when the checkpoint has that head.
            Sequence conditions affect network inputs only. The caller owns
            residue overrides, atom selection and structure assembly.
        """
        mask = mask.to(torch.bool)
        valid = mask.unsqueeze(-1).to(self.dtype)
        single, pair = self._build_representations(
            backbone_ca,
            local_latents,
            mask,
            valid,
            token_index,
            token_type,
            conditional_residue_type,
            conditional_residue_mask,
        )

        precomputed_single_masks = precompute_single_masks(
            self.config.pairwise_attention_backend,
            mask,
            inf=self.config.mask_inf,
        )
        condition = torch.zeros(*mask.shape, self.config.condition_dim, device=mask.device, dtype=self.dtype)
        condition = self.transition_c_1(condition, mask)
        condition = self.transition_c_2(condition, mask)
        # Only the transformer stack runs in reduced precision. The factories,
        # condition transitions, and output heads stay in the decoder dtype.
        single = single.to(self.transformer_dtype)
        if pair is not None:
            pair = pair.to(self.transformer_dtype)
        condition = condition.to(self.transformer_dtype)

        if self.precompute_bias:
            if pair is None:
                raise RuntimeError("precompute_bias requires a pair representation")
            biases = self._precompute_all_biases(pair)
            for layer in self.transformer_layers:
                layer.mha.bias_proj = False
            try:
                for i, layer in enumerate(self.transformer_layers):
                    single = layer(
                        single,
                        biases[i],
                        condition,
                        mask,
                        metadata=metadata,
                        buffers=buffers,
                        precomputed_single_masks=precomputed_single_masks,
                    )
            finally:
                for layer in self.transformer_layers:
                    layer.mha.bias_proj = True
        else:
            for layer in self.transformer_layers:
                single = layer(
                    single,
                    pair,
                    condition,
                    mask,
                    metadata=metadata,
                    buffers=buffers,
                    precomputed_single_masks=precomputed_single_masks,
                )

        single = single.to(self.dtype)
        sequence_logits = self.logit_linear(single) * valid
        if token_type is not None:
            allowed_residue_types = self.allowed_residue_types(sequence_logits.device)
            if allowed_residue_types is not None:
                if token_type.shape != mask.shape:
                    raise ValueError(f"token_type must have shape {tuple(mask.shape)}, got {tuple(token_type.shape)}")
                token_type = token_type.to(device=sequence_logits.device, dtype=torch.long)
                max_token_type = allowed_residue_types.shape[0] - 1
                invalid = (token_type < 0) | (token_type > max_token_type)
                if torch.any(invalid):
                    bad_values = torch.unique(token_type[invalid]).detach().cpu().tolist()
                    raise ValueError(
                        f"token_type must be in [0, {max_token_type}] for "
                        "unified AE decoding; got invalid values "
                        f"{bad_values}"
                    )
                allowed_per_token = allowed_residue_types[token_type] | ~mask.unsqueeze(-1)
                sequence_logits = sequence_logits.masked_fill(
                    ~allowed_per_token, torch.finfo(sequence_logits.dtype).min
                )
        coords = (self.struct_linear(single) * valid).unflatten(-1, (ATOM37_SIZE, 3))

        # The generated CA is authoritative: OSS zeroes the predicted CA slot
        # and (when abs_coors is False) treats the rest as offsets from it.
        coords = coords.clone()
        coords[..., ATOM37_CA_INDEX, :] = 0.0
        if self.absolute_coordinates:
            coords[..., ATOM37_CA_INDEX, :] = backbone_ca.to(coords.dtype)
        else:
            coords = coords + backbone_ca.to(coords.dtype).unsqueeze(-2)

        residue_type = torch.argmax(sequence_logits, dim=-1) * mask
        output = {"coords": coords, "sequence_logits": sequence_logits, "residue_type": residue_type}
        if self.atom_presence_linear is not None:
            output["atom_presence_logits"] = self.atom_presence_linear(single)
        return output


class _LinearFactory(nn.Module):
    """Bias-free projection of concatenated raw features, masked either side.

    Mirrors OSS ``FeatureFactory`` with ``use_ln_out=False``: mask the
    concatenated features, project, then mask again. The caller supplies
    ``valid`` so the same mask serves several projections and the output heads.
    """

    def __init__(self, in_dim: int, out_dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.linear_out = nn.Linear(in_dim, out_dim, bias=False, dtype=dtype)

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Project ``features``, masked by ``valid`` broadcast over channels."""
        return self.linear_out(features * valid) * valid

    def one_hot_tables(self, widths: tuple[int, ...]) -> list[torch.Tensor]:
        """Lookup tables equivalent to projecting concatenated one-hot blocks.

        For a bias-free projection, ``Linear(one_hot(k)) == W[:, k]``, so each
        block of ``widths`` contiguous input columns becomes a gather table of
        its transposed weight columns. Every table carries one extra zero row
        at the end, so a masked position can select it and come out zero
        without a separate masking pass.

        Slicing the live weight keeps checkpoint keys unchanged.
        """
        weight = self.linear_out.weight  # [out_dim, sum(widths)]
        tables, start = [], 0
        for width in widths:
            block = weight[:, start : start + width].t()  # [width, out_dim]
            tables.append(torch.cat([block, block.new_zeros(1, block.shape[1])]).contiguous())
            start += width
        return tables

    def fused_one_hot_table(self, widths: tuple[int, int]) -> torch.Tensor | None:
        """Fuse two one-hot blocks into a single ``[w0 * w1 + 1, out_dim]``.

        Projecting ``cat(one_hot(a), one_hot(b))`` sums one row from each
        block's table, so pre-adding every combination turns the pair into one
        gather and avoids ever holding two ``[B, N, N, out_dim]`` tensors.
        The trailing row is the zero row for masked positions.

        Returns ``None`` when the product of the widths would make the table
        exceed :data:`_MAX_FUSED_TABLE_BYTES`.
        """
        first, second = widths
        weight = self.linear_out.weight
        out_dim = weight.shape[0]
        if (first * second + 1) * out_dim * weight.element_size() > _MAX_FUSED_TABLE_BYTES:
            return None
        a, b = self.one_hot_tables(widths)
        fused = a[:first].unsqueeze(1) + b[:second].unsqueeze(0)
        return torch.cat([fused.reshape(first * second, out_dim), fused.new_zeros(1, out_dim)]).contiguous()
