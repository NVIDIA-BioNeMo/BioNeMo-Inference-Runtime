# Copyright 2025 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from bionemo_ir._torch.layers.linear import Linear
from bionemo_ir._torch.layers.transformers.pairformer import PairformerModule
from bionemo_ir._torch.modules.openfold3.utils.atomize_utils import (
    broadcast_token_feat_to_atoms,
    get_token_frame_mask,
    get_token_representative_atoms,
    max_atom_per_token_masked_select,
)
from bionemo_ir._torch.utils import (
    CHUNK_REGISTRY,
    CONFIDENCE_PAIR_EMBEDDING,
    CONFIDENCE_PAIR_PROJECTION,
    CONFIDENCE_TRIANGLE_ATTENTION,
    ChunkPolicy,
    iter_chunks,
    recursive_calling_load_weights,
)

if TYPE_CHECKING:
    from bionemo_ir.models.openfold3.config import PairformerConfig


def _project_pair_logits(
    zij: torch.Tensor,
    layer_norm: nn.Module,
    linear: Linear,
    c_out: int,
    policy: ChunkPolicy | None,
) -> torch.Tensor:
    """Project pair logits in bounded token-row slices when the policy engages."""

    def project(zij_slice: torch.Tensor) -> torch.Tensor:
        return linear(layer_norm(zij_slice))

    num_rows = zij.shape[-3]
    device = zij.device if zij.is_cuda else None
    if policy is None or not policy.should_chunk_size(num_rows, device):
        return project(zij)

    row_dim = zij.ndim - 3
    logits = zij.new_empty((*zij.shape[:-1], c_out))
    for start, length in iter_chunks(num_rows, policy.chunk_size):
        projected = project(zij.narrow(row_dim, start, length))
        logits.narrow(row_dim, start, length).copy_(projected)
        del projected
    return logits


def _symmetrize_pair_logits_(
    logits: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Symmetrize fresh pair logits in place with one bounded source block."""
    chunks = tuple(iter_chunks(logits.shape[-3], block_size))
    for row_index, (row_start, row_length) in enumerate(chunks):
        for column_start, column_length in chunks[row_index:]:
            upper = logits.narrow(-3, row_start, row_length).narrow(-2, column_start, column_length)
            upper_original = upper.clone()
            if row_start == column_start:
                upper.add_(upper_original.transpose(-2, -3))
                continue

            lower = logits.narrow(-3, column_start, column_length).narrow(-2, row_start, row_length)
            upper.add_(lower.transpose(-2, -3))
            lower.add_(upper_original.transpose(-2, -3))
    return logits


def _finalize_aux_output(
    name: str,
    value: torch.Tensor,
    *,
    out_device: torch.device,
    out_dtype: torch.dtype,
    keep_pair_logits_on_cpu: bool,
) -> torch.Tensor:
    """Cast one auxiliary output while preserving opted-in pair-logit offload."""
    keep_on_cpu = keep_pair_logits_on_cpu and name in ("pae_logits", "pde_logits")
    target_device = torch.device("cpu") if keep_on_cpu else out_device
    return value.to(device=target_device, dtype=out_dtype)


def _finalize_aux_outputs_(
    aux_out: dict[str, torch.Tensor],
    *,
    out_device: torch.device,
    out_dtype: torch.dtype,
    keep_pair_logits_on_cpu: bool,
    reclaim_cuda_cache: bool = False,
) -> dict[str, torch.Tensor]:
    """Finalize auxiliary outputs and reclaim large last-use CUDA allocations."""
    can_reclaim = (
        reclaim_cuda_cache
        and out_device.type == "cuda"
        and not keep_pair_logits_on_cpu
        and not torch.cuda.is_current_stream_capturing()
    )
    for name in tuple(aux_out):
        value = aux_out[name]
        if (
            can_reclaim
            and name in ("pae_logits", "pde_logits")
            and (value.device != out_device or value.dtype != out_dtype)
        ):
            torch.cuda.empty_cache()
        aux_out[name] = _finalize_aux_output(
            name,
            value,
            out_device=out_device,
            out_dtype=out_dtype,
            keep_pair_logits_on_cpu=keep_pair_logits_on_cpu,
        )
        del value
    return aux_out


class PairformerEmbedding(nn.Module):
    """
    Implements AF3 Algorithm 31, line 1 - 6
    """

    def __init__(
        self,
        pairformer: PairformerConfig,
        c_s_input: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bin: int,
        inf: float,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
    ):
        """
        Args:
            pairformer:
                Config for PairFormerStack used
            c_s_input:
                Single (input) embedding dimension
            c_z:
                Pair embedding dimension
            min_bin:
                Minimum value for bin (3.25). The value is slightly
                different from SI. Previous AF2 implementation utilized these values
                for bins.
            max_bin:
                Maximum value for bin (20.75). ibid
            no_bin:
                Number of bins (15). ibid
        """
        super().__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bin = no_bin
        self.inf = inf
        self.dtype = dtype
        self.skip_create_weights = skip_create_weights

        self.linear_i = Linear(
            c_s_input, c_z, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
        )

        self.linear_j = Linear(
            c_s_input, c_z, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
        )

        self.linear_distance = Linear(
            self.no_bin, c_z, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
        )

        bins = torch.linspace(min_bin, max_bin, no_bin)
        squared_bins = bins**2
        upper = torch.cat([squared_bins[1:], squared_bins.new_tensor([inf])], dim=-1)
        self.register_buffer("bins", bins, persistent=False)
        self.register_buffer("squared_bins", squared_bins, persistent=False)
        self.register_buffer("upper", upper, persistent=False)
        self.pair_embedding_chunk_policy = CHUNK_REGISTRY.get(CONFIDENCE_PAIR_EMBEDDING)
        self.pairformer_stack = PairformerModule(config=pairformer)
        triangle_attention_chunk_policy = CHUNK_REGISTRY[CONFIDENCE_TRIANGLE_ATTENTION]
        for layer in self.pairformer_stack.layers:
            # Confidence retains several O(N²) tensors here. Bound the fused
            # QKV transient without changing earlier pairformer stacks.
            layer.tri_attn_start.chunk_policy = triangle_attention_chunk_policy
            layer.tri_attn_end.chunk_policy = triangle_attention_chunk_policy

    def _embed_zij_dense(
        self,
        si_input: torch.Tensor,
        zij: torch.Tensor,
        x_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Build the complete confidence pair through the autograd-safe dense path."""
        orig_dtype = zij.dtype
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            # si projection to zij
            zij = zij + self.linear_i(si_input.unsqueeze(-2)) + self.linear_j(si_input.unsqueeze(-3))

            # Embed pair distances of representative atoms
            dij = torch.sum(
                (x_pred[..., None, :] - x_pred[..., None, :, :]) ** 2,
                dim=-1,
                keepdims=True,
            )
            dij = ((dij > self.squared_bins) * (dij < self.upper)).type(x_pred.dtype)
            zij = zij + self.linear_distance(dij)

        return zij.to(dtype=orig_dtype)

    def _embed_zij_chunked(
        self,
        si_input: torch.Tensor,
        zij: torch.Tensor,
        x_pred: torch.Tensor,
        policy: ChunkPolicy,
    ) -> torch.Tensor:
        """Build an owned final-dtype confidence pair from bounded FP32 row updates."""
        num_rows = zij.shape[-3]
        row_dim = zij.ndim - 3
        si_row_dim = si_input.ndim - 2
        x_row_dim = x_pred.ndim - 2
        leading_shape = torch.broadcast_shapes(zij.shape[:-3], si_input.shape[:-2], x_pred.shape[:-2])
        output = zij.new_empty((*leading_shape, num_rows, zij.shape[-2], zij.shape[-1]))

        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            projected_i = self.linear_i(si_input.unsqueeze(-2))
            projected_j = self.linear_j(si_input.unsqueeze(-3))
            for start, length in iter_chunks(num_rows, policy.chunk_size):
                pair_rows = zij.narrow(row_dim, start, length)
                projected_i_rows = projected_i.narrow(si_row_dim, start, length)
                embedded_rows = pair_rows + projected_i_rows + projected_j

                x_pred_rows = x_pred.narrow(x_row_dim, start, length)
                dij = torch.sum(
                    (x_pred_rows[..., None, :] - x_pred[..., None, :, :]) ** 2,
                    dim=-1,
                    keepdims=True,
                )
                dij = ((dij > self.squared_bins) * (dij < self.upper)).type(x_pred.dtype)
                embedded_rows = embedded_rows + self.linear_distance(dij)
                output.narrow(output.ndim - 3, start, length).copy_(embedded_rows)
                del dij, embedded_rows, pair_rows, projected_i_rows, x_pred_rows

        return output

    def embed_zij(
        self,
        si_input: torch.Tensor,
        zij: torch.Tensor,
        x_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Build the confidence pair, bounding inference temporaries by token row."""
        policy = self.pair_embedding_chunk_policy
        device = zij.device if zij.is_cuda else None
        if policy is None or not policy.should_chunk_size(zij.shape[-3], device):
            return self._embed_zij_dense(si_input, zij, x_pred)
        return self._embed_zij_chunked(si_input, zij, x_pred, policy)

    def iter_per_sample_pairformer_emb(
        self,
        si_input: torch.Tensor,
        si: torch.Tensor,
        zij: torch.Tensor,
        x_pred: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
        """Yield one completed confidence Pairformer sample at a time.

        Instead of expanding the pair across all samples, process one sample
        at a time so callers can choose how to consume each result.

        ``x_pred`` is ``[..., num_samples, N_token, 3]``. ``si``, ``zij``,
        and the masks may carry a sample dim of size 1 (from unsqueeze in the
        model forward) or the full sample dim immediately before their token
        dimensions. Select that axis for each sample while preserving any
        number of leading batch dimensions.
        """
        no_samples = x_pred.shape[-3]
        batch_ndim = x_pred.ndim - 3

        def select_sample(value: torch.Tensor, trailing_ndim: int, index: int) -> torch.Tensor:
            expected_ndim = batch_ndim + trailing_ndim
            if value.ndim == expected_ndim:
                return value
            sample_dim = value.ndim - trailing_ndim - 1
            if value.ndim == expected_ndim + 1 and value.shape[sample_dim] in (1, no_samples):
                return value.select(sample_dim, 0 if value.shape[sample_dim] == 1 else index)
            raise ValueError(f"Expected rank {expected_ndim} or a broadcastable sample axis, got {tuple(value.shape)}")

        for i in range(no_samples):
            zij_chunk = self.embed_zij(
                si_input=select_sample(si_input, 2, i),
                zij=select_sample(zij, 3, i),
                x_pred=x_pred.select(-3, i),
            )
            si_chunk = select_sample(si, 2, i)

            si_chunk, zij_chunk = self.pairformer_stack(
                si_chunk,
                zij_chunk,
                select_sample(single_mask, 1, i),
                select_sample(pair_mask, 2, i),
                inplace_safe=True,
            )
            yield i, si_chunk, zij_chunk
            # A suspended generator retains its locals. Release the completed
            # sample before constructing the next sample's pair embedding.
            del si_chunk, zij_chunk

    def per_sample_pairformer_emb(
        self,
        si_input: torch.Tensor,
        si: torch.Tensor,
        zij: torch.Tensor,
        x_pred: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        offload_to_cpu: bool = False,
    ):
        """Run Pairformer per sample and assemble complete output tensors."""
        no_samples = x_pred.shape[-3]
        output_device = zij.device
        si_device: torch.Tensor | None = None
        zij_device: torch.Tensor | None = None
        si_host: torch.Tensor | None = None
        zij_host: torch.Tensor | None = None

        for i, si_chunk, zij_chunk in self.iter_per_sample_pairformer_emb(
            si_input,
            si,
            zij,
            x_pred,
            single_mask,
            pair_mask,
        ):
            if offload_to_cpu:
                if si_host is None or zij_host is None:
                    si_host = torch.empty(
                        (*si_chunk.shape[:-2], no_samples, *si_chunk.shape[-2:]),
                        dtype=si_chunk.dtype,
                        device="cpu",
                    )
                    zij_host = torch.empty(
                        (*zij_chunk.shape[:-3], no_samples, *zij_chunk.shape[-3:]),
                        dtype=zij_chunk.dtype,
                        device="cpu",
                    )
                si_host.select(-3, i).copy_(si_chunk)
                zij_host.select(-4, i).copy_(zij_chunk)
            else:
                if si_device is None or zij_device is None:
                    si_device = torch.empty(
                        (*si_chunk.shape[:-2], no_samples, *si_chunk.shape[-2:]),
                        dtype=si_chunk.dtype,
                        device=si_chunk.device,
                    )
                    zij_device = torch.empty(
                        (*zij_chunk.shape[:-3], no_samples, *zij_chunk.shape[-3:]),
                        dtype=zij_chunk.dtype,
                        device=zij_chunk.device,
                    )
                si_device.select(-3, i).copy_(si_chunk)
                zij_device.select(-4, i).copy_(zij_chunk)

            del si_chunk, zij_chunk

        # [B, num_samples, N, C_s] and [B, num_samples, N, N, C_z]
        if offload_to_cpu:
            if si_host is None or zij_host is None:
                raise RuntimeError("per-sample confidence requires at least one diffusion sample")
            return si_host.to(device=output_device), zij_host.to(device=output_device)
        if si_device is None or zij_device is None:
            raise RuntimeError("per-sample confidence requires at least one diffusion sample")
        return si_device, zij_device

    def pairformer_emb(
        self,
        si_input: torch.Tensor,
        si: torch.Tensor,
        zij: torch.Tensor,
        x_pred: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ):
        zij = self.embed_zij(si_input=si_input, zij=zij, x_pred=x_pred)
        batch_dims = x_pred.shape[:-2]

        def reshape_inputs(x: torch.Tensor, feat_dims: list):
            x = x.expand(*(batch_dims + feat_dims))
            x = x.reshape(-1, *feat_dims)
            return x

        def reshape_outputs(x: torch.Tensor, feat_dims: list):
            return x.reshape(*batch_dims, *feat_dims)

        si = reshape_inputs(x=si, feat_dims=si.shape[-2:])
        zij = reshape_inputs(x=zij, feat_dims=zij.shape[-3:])
        single_mask = reshape_inputs(x=single_mask, feat_dims=single_mask.shape[-1:])
        pair_mask = reshape_inputs(x=pair_mask, feat_dims=pair_mask.shape[-2:])
        si, zij = self.pairformer_stack(
            si,
            zij,
            single_mask,
            pair_mask,
            inplace_safe=True,
        )

        si = reshape_outputs(x=si, feat_dims=si.shape[-2:])
        zij = reshape_outputs(x=zij, feat_dims=zij.shape[-3:])

        return si, zij

    def forward(
        self,
        si_input: torch.Tensor,
        si: torch.Tensor,
        zij: torch.Tensor,
        x_pred: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        apply_per_sample: bool = False,
        offload_to_cpu: bool = False,
    ):
        """
        Args:
            si_input:
                [*, N_token, C_s] Output of InputFeatureEmbedder
            si:
                [*, N_token, C_s] Single embedding
            zij:
                [*, N_token, N_token, C_z] Pairwise embedding
            x_pred:
                Representative atom predicted coordinates per token.
                Shape: [*, num_samples, N_token, 3] when apply_per_sample=True,
                or [*, N_token, 3] when apply_per_sample=False (expanded internally).
            single_mask:
                [*, N_token] Single mask
            pair_mask:
                [*, N_token, N_token] Pair mask
            apply_per_sample:
                When True, run pairformer embedding per diffusion sample
                to avoid OOM from expanding zij across all samples.
            offload_to_cpu:
                When True, stage completed per-sample outputs on the CPU until
                all Pairformer calls finish, then restore the assembled tensors
                to the input device. Only used with ``apply_per_sample=True``.
        Returns:
            si:
                [*, N_token, C_s] Updated single representation
            zij:
                [*, N_token, N_token, C_z] Updated pair representation
        """
        if apply_per_sample:
            si, zij = self.per_sample_pairformer_emb(
                si_input=si_input,
                si=si,
                zij=zij,
                x_pred=x_pred,
                single_mask=single_mask,
                pair_mask=pair_mask,
                offload_to_cpu=offload_to_cpu,
            )
        else:
            si, zij = self.pairformer_emb(
                si_input=si_input,
                si=si,
                zij=zij,
                x_pred=x_pred,
                single_mask=single_mask,
                pair_mask=pair_mask,
            )

        return si, zij


class PredictedAlignedErrorHead(nn.Module):
    """
    Implements PredictedAlignedError Head (Algorithm 31, Line 5) for
    AF3 (subsection 4.3.2)
    """

    def __init__(
        self,
        c_z: int,
        c_out: int,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-5,
        skip_create_weights: bool = False,
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c_out:
                Number of PredictedAlignedError (PAE) bins
        """
        super().__init__()

        self.c_z = c_z
        self.c_out = c_out

        self.layer_norm = nn.LayerNorm(self.c_z, dtype=dtype, eps=eps)
        self.linear = Linear(self.c_z, self.c_out, bias=False, dtype=dtype, skip_create_weights=skip_create_weights)
        self.projection_chunk_policy = CHUNK_REGISTRY.get(CONFIDENCE_PAIR_PROJECTION)

    def _compute_logits(self, zij: torch.Tensor):
        logits = _project_pair_logits(zij, self.layer_norm, self.linear, self.c_out, self.projection_chunk_policy)
        return logits

    def forward(self, zij):
        """
        Args:
            zij:
                [*, N, N, C_z] Pair embedding
        Returns:
            logits:
                [*, N, N, C_out] Logits
        """

        logits = self._compute_logits(zij=zij)

        return logits


class PredictedDistanceErrorHead(nn.Module):
    """
    Implements PredictedDistanceError Head (Algorithm 31, Line 6) for
    AF3 (subsection 4.3.3)
    """

    def __init__(
        self,
        c_z: int,
        c_out: int,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c_out:
                Number of PredictedDistanceError (PDE) bins
        """
        super().__init__()

        self.c_z = c_z
        self.c_out = c_out

        self.layer_norm = nn.LayerNorm(self.c_z, dtype=dtype, eps=eps)
        self.linear = Linear(self.c_z, self.c_out, bias=False, dtype=dtype, skip_create_weights=skip_create_weights)
        self.projection_chunk_policy = CHUNK_REGISTRY.get(CONFIDENCE_PAIR_PROJECTION)

    def _compute_logits(self, zij: torch.Tensor) -> torch.Tensor:
        policy = self.projection_chunk_policy
        device = zij.device if zij.is_cuda else None
        use_chunked_projection = (
            policy is not None
            and policy.should_chunk_size(zij.shape[-3], device)
            and (not zij.is_cuda or not torch.cuda.is_current_stream_capturing())
        )
        projection_policy = policy if use_chunked_projection else None
        logits = _project_pair_logits(zij, self.layer_norm, self.linear, self.c_out, projection_policy)
        if use_chunked_projection:
            return _symmetrize_pair_logits_(logits, policy.chunk_size)
        return logits + logits.transpose(-2, -3)

    def forward(self, zij):
        """
        Args:
            zij:
                [*, N, N, C_z] Pair embedding
        Returns:
            logits:
                [*, N, N, C_out] Logits
        """

        logits = self._compute_logits(zij=zij)

        return logits


class PerResidueLDDTAllAtom(nn.Module):
    """
    Implements Plddt Head (Algorithm 31, Line 7) for AF3 (subsection 4.3.1)
    """

    def __init__(
        self,
        c_s: int,
        c_out: int,
        max_atoms_per_token: int,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-5,
        skip_create_weights: bool = False,
    ):
        """
        Args:
            c_s:
                Input channel dimension
            max_atoms_per_token:
                Maximum atoms per token
            c_out:
                Number of PLDDT bins
        """
        super().__init__()

        self.c_s = c_s
        self.max_atoms_per_token = max_atoms_per_token
        self.c_out = c_out

        self.layer_norm = nn.LayerNorm(self.c_s, dtype=dtype, eps=eps)
        self.linear = Linear(
            self.c_s,
            self.max_atoms_per_token * self.c_out,
            bias=False,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

    def forward(self, s: torch.Tensor, max_atom_per_token_mask: torch.Tensor):
        """
        Args:
            s:
                [*, N_token, C_s] Single embedding
            max_atom_per_token_mask:
                [*, N_token * max_atoms_per_token] Flat mask of atoms per token
                padded to max_atoms_per_token
        Returns:
            logits:
                [*, N_atom, C_out] Logits
        """
        batch_dims = s.shape[:-2]
        n_token = s.shape[-2]

        # [*, N_token, max_atoms_per_token * c_out]
        logits = self.linear(self.layer_norm(s))

        # [*, N_token * max_atoms_per_token, c_out]
        logits = logits.reshape(*batch_dims, n_token * self.max_atoms_per_token, self.c_out)

        # [*, N_atom, c_out]
        logits = max_atom_per_token_masked_select(
            atom_feat=logits,
            max_atom_per_token_mask=max_atom_per_token_mask,
        )

        return logits


class ExperimentallyResolvedHeadAllAtom(nn.Module):
    """
    Implements resolvedHeads for AF3, subsection 4.3.3
    """

    def __init__(
        self,
        c_s: int,
        c_out: int,
        max_atoms_per_token: int,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-5,
        skip_create_weights: bool = False,
    ):
        """
        Args:
            c_s:
                Input channel dimension
            max_atoms_per_token:
                Maximum atoms per token
            c_out:
                Number of ExperimentallyResolved Head AllAtom bins
        """
        super().__init__()

        self.c_s = c_s
        self.max_atoms_per_token = max_atoms_per_token
        self.c_out = c_out

        self.layer_norm = nn.LayerNorm(self.c_s, dtype=dtype, eps=eps)
        self.linear = Linear(
            self.c_s,
            self.max_atoms_per_token * self.c_out,
            bias=False,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

    def forward(self, s: torch.Tensor, max_atom_per_token_mask: torch.Tensor):
        """
        Args:
            s:
                [*, N_token, C_s] Single embedding
            max_atom_per_token_mask:
                [*, N_token * max_atoms_per_token] Flat mask of atoms per token
                padded to max_atoms_per_token
        Returns:
            logits:
                [*, N_atom, C_out] Logits
        """
        batch_dims = s.shape[:-2]
        n_token = s.shape[-2]

        # [*, N_token, max_atoms_per_token * c_out]
        logits = self.linear(self.layer_norm(s))

        # [*, N_token * max_atoms_per_token, c_out]
        logits = logits.reshape(*batch_dims, n_token * self.max_atoms_per_token, self.c_out)

        # [*, N_atom, c_out]
        logits = max_atom_per_token_masked_select(
            atom_feat=logits,
            max_atom_per_token_mask=max_atom_per_token_mask,
        )

        return logits


class DistogramHead(nn.Module):
    """
    Implementation of distogram head for both AF2 and AF3.

    Computes a distogram probability distribution.
    For use in computation of distogram loss, subsection 1.9.8 (AF2), section 4.4 (AF3)
    """

    def __init__(
        self,
        c_z: int,
        c_out: int,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c_out:
                Number of distogram bins
        """
        super().__init__()

        self.c_z = c_z
        self.c_out = c_out

        self.linear = Linear(self.c_z, self.c_out, bias=False, dtype=dtype, skip_create_weights=skip_create_weights)
        self.projection_chunk_policy = CHUNK_REGISTRY.get(CONFIDENCE_PAIR_PROJECTION)

    def _compute_logits(self, z: torch.Tensor) -> torch.Tensor:
        policy = self.projection_chunk_policy
        use_chunked_inference = (
            z.is_cuda
            and policy is not None
            and policy.should_chunk_size(z.shape[-3], z.device)
            and not torch.cuda.is_current_stream_capturing()
        )
        if not use_chunked_inference:
            logits = self.linear(z)
            return logits + logits.transpose(-2, -3)

        row_dim = z.ndim - 3
        logits = z.new_empty((*z.shape[:-1], self.c_out))
        for start, length in iter_chunks(z.shape[-3], policy.chunk_size):
            projected = self.linear(z.narrow(row_dim, start, length))
            logits.narrow(row_dim, start, length).copy_(projected)
            del projected
        return _symmetrize_pair_logits_(logits, policy.chunk_size)

    def forward(self, z):
        """
        Args:
            z:
                [*, N, N, C_z] Pair embedding
        Returns:
            logit:
                [*, N, N, C_out] Distogram probability distribution

        Note:
            For symmetric pairwise PairDistanceError loss (PDE),
            logits are calculated by linear(zij + zij.transpose(-2, -3))
            In SI this happens before the linear layer is applied.
        """

        return self._compute_logits(z)


class AuxiliaryHeadsAllAtom(nn.Module):
    """
    Auxiliary head for OF3
    Implements AF3 Algorithm 31 with main inference loop (Algorithm 1) line 16 - 17.
    """

    def __init__(self, config):
        """
        Args:
            config: ConfigDict with following keys
                "pairformer_embedding": Pairformer embedding config
                "pae": PAE config
                "pde": PDE config
                "lddt": LDDT config
                "distogram": Distogram config
                "experimentally_resolved": Experimentally_resolved config
        """
        super().__init__()
        self.config = config
        self.max_atoms_per_token = config.max_atoms_per_token
        self.dtype = config.torch_dtype
        self.skip_create_weights = config.skip_create_weights
        self.apply_per_sample = config.memory_efficient_mode
        self.offload_pairformer_outputs = config.offload_pairformer_outputs

        self.pairformer_embedding = PairformerEmbedding(
            pairformer=config.pairformer,
            c_s_input=config.c_s_input,
            c_z=config.c_z,
            min_bin=config.min_bin,
            max_bin=config.max_bin,
            no_bin=config.no_bin,
            inf=config.inf,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights,
        )

        self.pde = PredictedDistanceErrorHead(
            c_z=config.pde.c_z, c_out=config.pde.c_out, dtype=self.dtype, skip_create_weights=self.skip_create_weights
        )

        self.plddt = PerResidueLDDTAllAtom(
            c_s=config.lddt.c_s,
            c_out=config.lddt.c_out,
            max_atoms_per_token=config.lddt.max_atoms_per_token,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights,
        )

        self.distogram = DistogramHead(
            c_z=config.distogram.c_z,
            c_out=config.distogram.c_out,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights,
        )

        self.experimentally_resolved = ExperimentallyResolvedHeadAllAtom(
            c_s=config.experimentally_resolved.c_s,
            c_out=config.experimentally_resolved.c_out,
            max_atoms_per_token=config.experimentally_resolved.max_atoms_per_token,
            dtype=self.dtype,
            skip_create_weights=self.skip_create_weights,
        )

        if config.pae.enabled:
            self.pae = PredictedAlignedErrorHead(
                c_z=config.pae.c_z,
                c_out=config.pae.c_out,
                dtype=self.dtype,
                skip_create_weights=self.skip_create_weights,
            )

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # Every entry of ``weights`` must have been consumed.
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weights}")

    def _should_stream_pair_heads_on_device(self, zij: torch.Tensor) -> bool:
        policy = self.pde.projection_chunk_policy
        return (
            self.apply_per_sample
            and not self.offload_pairformer_outputs
            and zij.device.type == "cuda"
            and policy is not None
            and policy.should_chunk_size(zij.shape[-3], zij.device)
            and not torch.cuda.is_current_stream_capturing()
        )

    def _should_defer_offloaded_distogram(self, zij: torch.Tensor) -> bool:
        policy = self.pde.projection_chunk_policy
        return (
            self.apply_per_sample
            and self.offload_pairformer_outputs
            and zij.device.type == "cuda"
            and policy is not None
            and policy.should_chunk_size(zij.shape[-3], zij.device)
            and not torch.cuda.is_current_stream_capturing()
        )

    def _should_stream_pair_heads_to_cpu(self, zij: torch.Tensor) -> bool:
        return (
            self.apply_per_sample
            and self.offload_pairformer_outputs
            and (not zij.is_cuda or not torch.cuda.is_current_stream_capturing())
        )

    def _stream_pair_heads(
        self,
        *,
        si_input: torch.Tensor,
        si: torch.Tensor,
        zij: torch.Tensor,
        x_pred: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        output_device: torch.device,
        pair_output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Project each completed Pairformer sample under one output policy."""
        no_samples = x_pred.shape[-3]
        if no_samples < 1:
            raise RuntimeError("per-sample confidence requires at least one diffusion sample")

        pair_heads = {}
        if self.config.pae.enabled:
            pair_heads["pae_logits"] = self.pae
        pair_heads["pde_logits"] = self.pde

        batch_shape = x_pred.shape[:-3]
        pair_shape = (*batch_shape, no_samples, zij.shape[-3], zij.shape[-2])
        pair_outputs = {
            name: torch.empty((*pair_shape, head.c_out), dtype=pair_output_dtype, device=output_device)
            for name, head in pair_heads.items()
        }
        si_output = torch.empty(
            (*batch_shape, no_samples, *si.shape[-2:]),
            dtype=si.dtype,
            device=output_device,
        )

        for sample_idx, si_sample, zij_sample in self.pairformer_embedding.iter_per_sample_pairformer_emb(
            si_input,
            si,
            zij,
            x_pred,
            single_mask,
            pair_mask,
        ):
            si_output.select(-3, sample_idx).copy_(si_sample)
            for name, head in pair_heads.items():
                logits = head(zij_sample)
                pair_outputs[name].select(-4, sample_idx).copy_(logits)
                del logits
            del si_sample, zij_sample

        return si_output, pair_outputs

    def forward(self, batch: dict, si_input: torch.Tensor, output: dict):
        """
        Args:
            batch:
                Input feature dictionary
            si_input:
                [*, N_token, C_s_input] Single (input) representation
            output:
                Dict containing outputs
                    "si_trunk" ([*, N_token, C_s]):
                        Single representation output from model trunk
                    "zij_trunk" ([*, N_token, N_token, C_z]):
                        Pair representation output from model trunk
                    "atom_positions_predicted" ([*, N_atom, 3]):
                        Predicted atom positions

        Returns:
            aux_out:
                Dict containing following keys:
                    "plddt_logits" ([*, N_atom, 50]):
                        Predicted binned PLDDT logits
                    "pae_logits" ([*, N_token, N_token, 64]):
                        Predicted binned PAE logits
                    "valid_frame_mask" ([*, N_token]):
                        Tokens with a valid frame, the ``has_frame`` input of the
                        pTM / ipTM outer maximum. Present with "pae_logits" only.
                    "pde_logits" ([*, N_token, N_token, 64]):
                        Predicted binned PDE logits
                    "experimentally_resolved_logits" ([*, N_atom, 2]):
                        Predicted binned experimentally resolved logits
                    "distogram_logits" ([*, N_token, N_token, 64]):
                        Predicted binned distogram logits
        Note:
            Previous implementations of losses include softmax so all
            heads return logits. With offload_pairformer_outputs enabled,
            PAE and PDE logits remain on CPU; other outputs stay on the
            predicted-coordinate device. All outputs use the coordinate dtype.
        """
        aux_out = {}

        out_dtype = output["atom_positions_predicted"].dtype
        si = output["si_trunk"].to(dtype=self.dtype)
        zij = output["zij_trunk"].to(dtype=self.dtype)
        atom_positions_predicted = output["atom_positions_predicted"].to(dtype=si.dtype)

        pair_offload_enabled = self.offload_pairformer_outputs and (
            not zij.is_cuda or not torch.cuda.is_current_stream_capturing()
        )
        stream_pair_heads_on_device = self._should_stream_pair_heads_on_device(zij)
        stream_pair_heads_to_cpu = self._should_stream_pair_heads_to_cpu(zij)
        defer_distogram = stream_pair_heads_on_device or self._should_defer_offloaded_distogram(zij)
        if defer_distogram:
            # Allocate persistent head outputs before Pairformer scratch so
            # later split segments can be returned at their last-use boundary.
            torch.cuda.empty_cache()

        # The deferred paths compute this after pair-head finalization so its
        # FP32 output does not span the current confidence memory peak.
        if not defer_distogram:
            aux_out["distogram_logits"] = self.distogram(z=zij)

        token_mask = batch["token_mask"]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        # Get representative atoms
        repr_x_pred, repr_x_mask = get_token_representative_atoms(
            batch=batch, x=atom_positions_predicted, atom_mask=batch["atom_mask"]
        )
        si_input = si_input.to(dtype=self.dtype)
        repr_x_pred = repr_x_pred.to(dtype=self.dtype)
        repr_x_mask = repr_x_mask.to(dtype=self.dtype)
        pair_mask = pair_mask.to(dtype=self.dtype)

        out_device = atom_positions_predicted.device
        # Embed trunk outputs
        if stream_pair_heads_on_device or stream_pair_heads_to_cpu:
            pair_output_device = torch.device("cpu") if stream_pair_heads_to_cpu else zij.device
            pair_output_dtype = out_dtype if stream_pair_heads_to_cpu else zij.dtype
            si, streamed_pair_outputs = self._stream_pair_heads(
                si_input=si_input,
                si=si,
                zij=zij,
                x_pred=repr_x_pred,
                single_mask=repr_x_mask,
                pair_mask=pair_mask,
                output_device=pair_output_device,
                pair_output_dtype=pair_output_dtype,
            )
            del zij
        else:
            si, zij = self.pairformer_embedding(
                si_input=si_input,
                si=si,
                zij=zij,
                x_pred=repr_x_pred,
                single_mask=repr_x_mask,
                pair_mask=pair_mask,
                apply_per_sample=self.apply_per_sample,
                offload_to_cpu=pair_offload_enabled,
            )

        # Get atom mask padded to MAX_ATOMS_PER_TOKEN
        # Required to extract pLDDT and experimentally resolved logits for
        # the flat atom representation

        max_atom_per_token_mask = broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=token_mask,
            max_num_atoms_per_token=self.max_atoms_per_token,
        )

        si = si.to(device=out_device)
        aux_out["plddt_logits"] = self.plddt(s=si, max_atom_per_token_mask=max_atom_per_token_mask)

        aux_out["experimentally_resolved_logits"] = self.experimentally_resolved(si, max_atom_per_token_mask)
        del si

        if stream_pair_heads_on_device or stream_pair_heads_to_cpu:
            aux_out.update(streamed_pair_outputs)
            # Drop the second dictionary reference so finalization can release
            # each device archive immediately after replacing it with FP32.
            streamed_pair_outputs.clear()
        else:
            if self.config.pae.enabled:
                aux_out["pae_logits"] = self.pae(zij)
                if pair_offload_enabled:
                    aux_out["pae_logits"] = aux_out["pae_logits"].to(device="cpu")
            aux_out["pde_logits"] = self.pde(zij)
            if pair_offload_enabled:
                aux_out["pde_logits"] = aux_out["pde_logits"].to(device="cpu")

        if self.config.pae.enabled:
            # has_frame for the pTM / ipTM outer maximum, from the sampled
            # coordinates, so it is per diffusion sample. Only the PAE head feeds
            # pTM / ipTM, so nothing needs it when that head is off.
            aux_out["valid_frame_mask"] = get_token_frame_mask(
                batch=batch, x=atom_positions_predicted, atom_mask=batch["atom_mask"]
            ).to(device=out_device)
        reclaim_pair_head_cache = stream_pair_heads_on_device
        if not (stream_pair_heads_on_device or stream_pair_heads_to_cpu):
            pair_projection_policy = self.pde.projection_chunk_policy
            reclaim_pair_head_cache = (
                self.apply_per_sample
                and not pair_offload_enabled
                and zij.device.type == "cuda"
                and zij.dtype != out_dtype
                and pair_projection_policy is not None
                and pair_projection_policy.should_chunk_size(zij.shape[-3], zij.device)
            )
            del zij

        aux_out = _finalize_aux_outputs_(
            aux_out,
            out_device=out_device,
            out_dtype=out_dtype,
            keep_pair_logits_on_cpu=pair_offload_enabled,
            reclaim_cuda_cache=reclaim_pair_head_cache,
        )
        if defer_distogram:
            # Recompute the cheap trunk cast after the BF16 pair-head sources
            # are dead instead of retaining distogram logits across them.
            zij_distogram = output["zij_trunk"].to(dtype=self.dtype)
            distogram_logits = self.distogram(z=zij_distogram)
            del zij_distogram
            distogram_logits = _finalize_aux_output(
                "distogram_logits",
                distogram_logits,
                out_device=out_device,
                out_dtype=out_dtype,
                keep_pair_logits_on_cpu=False,
            )
            aux_out = {"distogram_logits": distogram_logits, **aux_out}
        return aux_out
