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

"""Fused weighted rigid alignment (Kabsch) for inference.

One launch replaces the eager SVD path: ``_rigid_align_solve`` reduces the
weighted centroids and cross-covariance per ``[B, M]`` cloud and solves the
proper rotation plus shift on device; pointwise float32 arithmetic applies
them. Nothing reads back to the host, so the path is CUDA-graph safe.

The rotation is Horn's closed form: the optimal unit quaternion is the top
eigenvector of a symmetric 4x4 built from the cross-covariance, which cyclic
Jacobi sweeps diagonalize in float64 inside the kernel. Quaternions only
parameterize proper rotations, so reflections never need a determinant fix.
See https://doi.org/10.1364/JOSAA.4.000629 (Horn 1987).
"""

import threading
from functools import cache

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from bionemo_ir.dsl_kernels.triton_cache import TritonKernelCache

# Floor on the summed alignment weights (empty mask -> identity transform).
WEIGHT_EPS = 1e-8
# Diagonal jitter on the cross-covariance (empty or collapsed clouds -> identity rotation).
COV_EPS = 1e-6
# Triton kernels only read module globals wrapped as constexpr.
_WEIGHT_EPS = tl.constexpr(WEIGHT_EPS)
_COV_EPS = tl.constexpr(COV_EPS)
_SWEEPS = 8
_SOLVE_BLOCK = 128
# The per-device DriverLauncher shares one params list across threads;
# !530's ``DriverLauncher.launch_with`` supersedes this lock.
_LAUNCH_LOCK = threading.Lock()


@triton.jit
def _entry(a, i: tl.constexpr, j: tl.constexpr):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 4)[None, :]
    return tl.sum(tl.where((rows == i) & (cols == j), a, 0.0))


@triton.jit
def _matmul4(a, b):
    return tl.sum(a[:, :, None] * b[None, :, :], axis=1)


@triton.jit
def _jacobi_rotate(a, v, p: tl.constexpr, q: tl.constexpr):
    """One Jacobi rotation zeroing ``a[p, q]`` (Golub & Van Loan 8.4.1)."""
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 4)[None, :]
    app = _entry(a, p, p)
    aqq = _entry(a, q, q)
    apq = _entry(a, p, q)
    skip = tl.abs(apq) <= 1e-300
    theta = (aqq - app) / (2.0 * tl.where(skip, 1.0, apq))
    sign = tl.where(theta >= 0.0, 1.0, -1.0)
    t = sign / (tl.abs(theta) + libdevice.sqrt(theta * theta + 1.0))
    t = tl.where(skip, 0.0, t)
    c = 1.0 / libdevice.sqrt(t * t + 1.0)
    s = t * c
    diag = (rows == cols).to(tl.float64)
    on_pq = ((rows == p) & (cols == p)) | ((rows == q) & (cols == q))
    g = diag * tl.where(on_pq, c, 1.0)
    g = g + tl.where((rows == p) & (cols == q), s, 0.0) + tl.where((rows == q) & (cols == p), -s, 0.0)
    a = _matmul4(tl.trans(g), _matmul4(a, g))
    v = _matmul4(v, g)
    return a, v


@triton.jit(do_not_specialize=["n_atoms"])
def _rigid_align_solve(
    true_ptr,
    pred_ptr,
    weight_ptr,
    mask_ptr,
    rot_ptr,
    shift_ptr,
    n_atoms,
    SWEEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    cloud = tl.program_id(0)
    base = cloud.to(tl.int64) * n_atoms
    zeros = tl.zeros([BLOCK], tl.float64)
    sw = tl.sum(zeros)
    st0 = tl.sum(zeros)
    st1 = tl.sum(zeros)
    st2 = tl.sum(zeros)
    sp0 = tl.sum(zeros)
    sp1 = tl.sum(zeros)
    sp2 = tl.sum(zeros)
    h00 = tl.sum(zeros)
    h01 = tl.sum(zeros)
    h02 = tl.sum(zeros)
    h10 = tl.sum(zeros)
    h11 = tl.sum(zeros)
    h12 = tl.sum(zeros)
    h20 = tl.sum(zeros)
    h21 = tl.sum(zeros)
    h22 = tl.sum(zeros)
    for start in range(0, n_atoms, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        valid = idx < n_atoms
        row = base + idx
        w = tl.load(weight_ptr + row, valid, 0.0).to(tl.float64) * tl.load(mask_ptr + row, valid, 0.0).to(tl.float64)
        t0 = tl.load(true_ptr + row * 3, valid, 0.0).to(tl.float64)
        t1 = tl.load(true_ptr + row * 3 + 1, valid, 0.0).to(tl.float64)
        t2 = tl.load(true_ptr + row * 3 + 2, valid, 0.0).to(tl.float64)
        p0 = tl.load(pred_ptr + row * 3, valid, 0.0).to(tl.float64)
        p1 = tl.load(pred_ptr + row * 3 + 1, valid, 0.0).to(tl.float64)
        p2 = tl.load(pred_ptr + row * 3 + 2, valid, 0.0).to(tl.float64)
        sw += tl.sum(w)
        st0 += tl.sum(w * t0)
        st1 += tl.sum(w * t1)
        st2 += tl.sum(w * t2)
        wp0 = w * p0
        wp1 = w * p1
        wp2 = w * p2
        sp0 += tl.sum(wp0)
        sp1 += tl.sum(wp1)
        sp2 += tl.sum(wp2)
        h00 += tl.sum(wp0 * t0)
        h01 += tl.sum(wp0 * t1)
        h02 += tl.sum(wp0 * t2)
        h10 += tl.sum(wp1 * t0)
        h11 += tl.sum(wp1 * t1)
        h12 += tl.sum(wp1 * t2)
        h20 += tl.sum(wp2 * t0)
        h21 += tl.sum(wp2 * t1)
        h22 += tl.sum(wp2 * t2)
    mass = tl.maximum(sw, _WEIGHT_EPS)
    tc0 = st0 / mass
    tc1 = st1 / mass
    tc2 = st2 / mass
    pc0 = sp0 / mass
    pc1 = sp1 / mass
    pc2 = sp2 / mass
    # Centered cross-covariance H = sum w (p - pc)(t - tc)^T / mass.
    h00 = h00 / mass - pc0 * tc0 + _COV_EPS
    h01 = h01 / mass - pc0 * tc1
    h02 = h02 / mass - pc0 * tc2
    h10 = h10 / mass - pc1 * tc0
    h11 = h11 / mass - pc1 * tc1 + _COV_EPS
    h12 = h12 / mass - pc1 * tc2
    h20 = h20 / mass - pc2 * tc0
    h21 = h21 / mass - pc2 * tc1
    h22 = h22 / mass - pc2 * tc2 + _COV_EPS
    # Horn's N with S = H^T (left cloud = true, right cloud = pred).
    sxx = h00
    sxy = h10
    sxz = h20
    syx = h01
    syy = h11
    syz = h21
    szx = h02
    szy = h12
    szz = h22
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 4)[None, :]
    n = tl.where((rows == 0) & (cols == 0), sxx + syy + szz, 0.0)
    n += tl.where((rows == 1) & (cols == 1), sxx - syy - szz, 0.0)
    n += tl.where((rows == 2) & (cols == 2), -sxx + syy - szz, 0.0)
    n += tl.where((rows == 3) & (cols == 3), -sxx - syy + szz, 0.0)
    n += tl.where(((rows == 0) & (cols == 1)) | ((rows == 1) & (cols == 0)), syz - szy, 0.0)
    n += tl.where(((rows == 0) & (cols == 2)) | ((rows == 2) & (cols == 0)), szx - sxz, 0.0)
    n += tl.where(((rows == 0) & (cols == 3)) | ((rows == 3) & (cols == 0)), sxy - syx, 0.0)
    n += tl.where(((rows == 1) & (cols == 2)) | ((rows == 2) & (cols == 1)), sxy + syx, 0.0)
    n += tl.where(((rows == 1) & (cols == 3)) | ((rows == 3) & (cols == 1)), szx + sxz, 0.0)
    n += tl.where(((rows == 2) & (cols == 3)) | ((rows == 3) & (cols == 2)), syz + szy, 0.0)
    v = (rows == cols).to(tl.float64)
    # Runtime loop over sweeps: unrolling all 48 rotations made the cold
    # compile take ~18 s per process.
    for _ in range(SWEEPS):
        for p in tl.static_range(4):
            for q in tl.static_range(p + 1, 4):
                n, v = _jacobi_rotate(n, v, p, q)
    # Top eigenvector = optimal unit quaternion (w, x, y, z).
    lam = tl.sum(tl.where(rows == cols, n, 0.0), axis=1)
    best = tl.argmax(lam, axis=0)
    qw = tl.sum(tl.where((rows == 0) & (cols == best), v, 0.0))
    qx = tl.sum(tl.where((rows == 1) & (cols == best), v, 0.0))
    qy = tl.sum(tl.where((rows == 2) & (cols == best), v, 0.0))
    qz = tl.sum(tl.where((rows == 3) & (cols == best), v, 0.0))
    norm = qw * qw + qx * qx + qy * qy + qz * qz
    r00 = (qw * qw + qx * qx - qy * qy - qz * qz) / norm
    r01 = 2.0 * (qx * qy - qw * qz) / norm
    r02 = 2.0 * (qx * qz + qw * qy) / norm
    r10 = 2.0 * (qx * qy + qw * qz) / norm
    r11 = (qw * qw - qx * qx + qy * qy - qz * qz) / norm
    r12 = 2.0 * (qy * qz - qw * qx) / norm
    r20 = 2.0 * (qx * qz - qw * qy) / norm
    r21 = 2.0 * (qy * qz + qw * qx) / norm
    r22 = (qw * qw - qx * qx - qy * qy + qz * qz) / norm
    out = rot_ptr + cloud * 9
    tl.store(out, r00)
    tl.store(out + 1, r01)
    tl.store(out + 2, r02)
    tl.store(out + 3, r10)
    tl.store(out + 4, r11)
    tl.store(out + 5, r12)
    tl.store(out + 6, r20)
    tl.store(out + 7, r21)
    tl.store(out + 8, r22)
    # aligned = R (t - tc) + pc = R t + shift
    tl.store(shift_ptr + cloud * 3, pc0 - (r00 * tc0 + r01 * tc1 + r02 * tc2))
    tl.store(shift_ptr + cloud * 3 + 1, pc1 - (r10 * tc0 + r11 * tc1 + r12 * tc2))
    tl.store(shift_ptr + cloud * 3 + 2, pc2 - (r20 * tc0 + r21 * tc1 + r22 * tc2))


class _RigidAlignKernel(TritonKernelCache):
    def __init__(self) -> None:
        self.kernel = self.compile_for_dtypes(
            _rigid_align_solve,
            dtypes=[torch.float32],
            make_dummy_args=lambda dtype: (
                torch.empty(3, device="cuda", dtype=dtype),
                torch.empty(3, device="cuda", dtype=dtype),
                torch.empty(1, device="cuda", dtype=dtype),
                torch.empty(1, device="cuda", dtype=dtype),
                torch.empty(9, device="cuda", dtype=dtype),
                torch.empty(3, device="cuda", dtype=dtype),
                1,
            ),
            grid=(0,),
            SWEEPS=_SWEEPS,
            BLOCK=_SOLVE_BLOCK,
        )[torch.float32]


@cache
def _rigid_align_kernel(device: int) -> _RigidAlignKernel:
    with torch.cuda.device(device):
        return _RigidAlignKernel()


def rigid_align_transform(
    true_coords: torch.Tensor, pred_coords: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the weighted proper rotation and shift mapping ``true`` onto ``pred``.

    Args:
        true_coords: ``[B, M, N, 3]`` contiguous CUDA float32 coordinates to move.
        pred_coords: ``[B, M, N, 3]`` target coordinates, same layout.
        weights: ``[B, M, N]`` float32 alignment weights.
        mask: ``[B, M, N]`` float32 atom mask multiplied into ``weights``.

    Returns:
        ``(rotation, shift)`` with shapes ``[B, M, 3, 3]`` and ``[B, M, 3]`` such
        that ``aligned = true @ rotation.mT + shift``.
    """
    batch, multiplicity, n_atoms, _ = true_coords.shape
    rotation = torch.empty((batch, multiplicity, 3, 3), device=true_coords.device, dtype=torch.float32)
    shift = torch.empty((batch, multiplicity, 3), device=true_coords.device, dtype=torch.float32)
    clouds = batch * multiplicity
    if clouds:
        with torch.cuda.device(true_coords.device):
            kernel = _rigid_align_kernel(true_coords.device.index).kernel
            driver = kernel.driver
            if driver is not None:
                with _LAUNCH_LOCK:
                    driver.params[0].value = true_coords.data_ptr()
                    driver.params[1].value = pred_coords.data_ptr()
                    driver.params[2].value = weights.data_ptr()
                    driver.params[3].value = mask.data_ptr()
                    driver.params[4].value = rotation.data_ptr()
                    driver.params[5].value = shift.data_ptr()
                    driver.params[6].value = n_atoms
                    driver.launch(clouds)
            else:
                kernel.launch(
                    (clouds,),
                    true_coords,
                    pred_coords,
                    weights,
                    mask,
                    rotation,
                    shift,
                    n_atoms,
                    _SWEEPS,
                    _SOLVE_BLOCK,
                )
    return rotation, shift


def supports_rigid_align(
    true_coords: torch.Tensor, pred_coords: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor
) -> bool:
    """Check whether the inputs can take the fused inference kernel."""
    tensors = (true_coords, pred_coords, weights, mask)
    return (
        true_coords.is_cuda
        and not torch.compiler.is_compiling()
        and true_coords.ndim == 4
        and true_coords.shape[-1] == 3
        and pred_coords.shape == true_coords.shape
        and weights.shape == true_coords.shape[:-1]
        and mask.shape == true_coords.shape[:-1]
        and true_coords.numel() < 2**31
        and all(t.dtype == torch.float32 and t.device == true_coords.device for t in tensors)
        and all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in tensors)
        and not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))
    )
