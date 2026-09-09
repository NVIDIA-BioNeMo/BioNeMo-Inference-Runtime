# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math

import torch
import torch.nn as nn

from bionemo_ir._torch.layers.linear import Linear
from bionemo_ir._torch.modules.openfold2.utils.geometry.rigid_matrix_vector import Rigid3Array
from bionemo_ir._torch.modules.openfold2.utils.rigid_utils import Rigid
from bionemo_ir._torch.utils import flatten_final_dims, permute_final_dims


class PointProjection(nn.Module):
    def __init__(
        self,
        c_hidden: int,
        num_points: int,
        no_heads: int,
        is_multimer: bool,
        return_local_points: bool = False,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
    ):
        super().__init__()
        self.return_local_points = return_local_points
        self.no_heads = no_heads
        self.num_points = num_points
        self.is_multimer = is_multimer

        self.linear = Linear(
            c_hidden,
            no_heads * 3 * num_points,
            bias=True,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

    def forward(
        self,
        activations: torch.Tensor,
        rigids: Rigid | Rigid3Array,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        points_local = self.linear(activations)
        out_shape = points_local.shape[:-1] + (self.no_heads, self.num_points, 3)

        if self.is_multimer:
            points_local = points_local.view(points_local.shape[:-1] + (self.no_heads, -1))

        points_local = torch.split(points_local, points_local.shape[-1] // 3, dim=-1)

        points_local = torch.stack(points_local, dim=-1).view(out_shape)
        points_global = rigids[..., None, None].apply(points_local)

        if self.return_local_points:
            return points_global, points_local

        return points_global


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
        is_multimer: bool = False,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps
        self.is_multimer = is_multimer

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(
            self.c_s, hc, bias=(not is_multimer), dtype=dtype, skip_create_weights=skip_create_weights
        )

        self.linear_q_points = PointProjection(
            self.c_s,
            self.no_qk_points,
            self.no_heads,
            self.is_multimer,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

        if is_multimer:
            self.linear_k = Linear(self.c_s, hc, bias=False, dtype=dtype, skip_create_weights=skip_create_weights)

            self.linear_v = Linear(self.c_s, hc, bias=False, dtype=dtype, skip_create_weights=skip_create_weights)

            self.linear_k_points = PointProjection(
                self.c_s,
                self.no_qk_points,
                self.no_heads,
                self.is_multimer,
                dtype=dtype,
                skip_create_weights=skip_create_weights,
            )

            self.linear_v_points = PointProjection(
                self.c_s,
                self.no_v_points,
                self.no_heads,
                self.is_multimer,
                dtype=dtype,
                skip_create_weights=skip_create_weights,
            )
        else:
            self.linear_kv = Linear(self.c_s, 2 * hc, bias=True, dtype=dtype, skip_create_weights=skip_create_weights)
            self.linear_kv_points = PointProjection(
                self.c_s,
                self.no_qk_points + self.no_v_points,
                self.no_heads,
                self.is_multimer,
                dtype=dtype,
                skip_create_weights=skip_create_weights,
            )

        self.linear_b = Linear(self.c_z, self.no_heads, bias=True, dtype=dtype, skip_create_weights=skip_create_weights)

        self.head_weights = nn.Parameter(torch.zeros(no_heads))

        concat_out_dim = self.no_heads * (self.c_z + self.c_hidden + self.no_v_points * 4)
        self.linear_out = Linear(
            concat_out_dim, self.c_s, bias=True, dtype=dtype, skip_create_weights=skip_create_weights
        )

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(self, s: torch.Tensor, z: torch.Tensor, r: Rigid | Rigid3Array, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, P_qk]
        q_pts = self.linear_q_points(s, r)

        # The following two blocks are equivalent
        # They're separated only to preserve compatibility with old AF weights
        if self.is_multimer:
            # [*, N_res, H * C_hidden]
            k = self.linear_k(s)
            v = self.linear_v(s)

            # [*, N_res, H, C_hidden]
            k = k.view(k.shape[:-1] + (self.no_heads, -1))
            v = v.view(v.shape[:-1] + (self.no_heads, -1))

            # [*, N_res, H, P_qk, 3]
            k_pts = self.linear_k_points(s, r)

            # [*, N_res, H, P_v, 3]
            v_pts = self.linear_v_points(s, r)
        else:
            # [*, N_res, H * 2 * C_hidden]
            kv = self.linear_kv(s)

            # [*, N_res, H, 2 * C_hidden]
            kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

            # [*, N_res, H, C_hidden]
            k, v = torch.split(kv, self.c_hidden, dim=-1)

            kv_pts = self.linear_kv_points(s, r)

            # [*, N_res, H, P_q/P_v, 3]
            k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z)

        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )

        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)

        pt_att = pt_att**2

        pt_att = sum(torch.unbind(pt_att, dim=-1))

        head_weights = self.softplus(self.head_weights).view(*((1,) * len(pt_att.shape[:-2]) + (-1, 1)))
        head_weights = head_weights * math.sqrt(1.0 / (3 * (self.no_qk_points * 9.0 / 2)))

        pt_att *= head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]

        v_pts = permute_final_dims(v_pts, (1, 3, 0, 2))
        o_pt = [torch.matmul(a, v.to(a.dtype)) for v in torch.unbind(v_pts, dim=-3)]
        o_pt = torch.stack(o_pt, dim=-3)

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps), 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
        o_pt = torch.unbind(o_pt, dim=-1)

        # [*, N_res, H, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z.to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        s = self.linear_out(torch.cat((o, *o_pt, o_pt_norm, o_pair), dim=-1).to(dtype=z.dtype))

        return s
