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
Jacobi sweeps diagonalize in float32 inside the kernel. Quaternions only
parameterize proper rotations, so reflections never need a determinant fix.
See https://doi.org/10.1364/JOSAA.4.000629 (Horn 1987).
"""

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
_SOLVE_BLOCK = 1024
_SOLVE_WARPS = 8


@triton.jit
def _entry(a, i: tl.constexpr, j: tl.constexpr):
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 4)[None, :]
    return tl.sum(tl.where((rows == i) & (cols == j), a, 0.0))


@triton.jit
def _jacobi_rotate(a, v, p: tl.constexpr, q: tl.constexpr):
    """One Jacobi rotation zeroing ``a[p, q]`` (Golub & Van Loan 8.4.1)."""
    rows = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, 4)[None, :]
    app = _entry(a, p, p)
    aqq = _entry(a, q, q)
    apq = _entry(a, p, q)
    skip = tl.abs(apq) <= 1e-20
    theta = (aqq - app) / (2.0 * tl.where(skip, 1.0, apq))
    sign = tl.where(theta >= 0.0, 1.0, -1.0)
    t = sign / (tl.abs(theta) + libdevice.sqrt(theta * theta + 1.0))
    t = tl.where(skip, 0.0, t)
    c = 1.0 / libdevice.sqrt(t * t + 1.0)
    s = t * c
    # Update the two columns directly and mirror them to preserve symmetry.
    # Forming G.T @ A @ G needlessly rounds the untouched entries.
    ap = tl.sum(tl.where(cols == p, a, 0.0), axis=1)
    aq = tl.sum(tl.where(cols == q, a, 0.0), axis=1)
    new_p = c * ap - s * aq
    new_q = s * ap + c * aq
    a = tl.where(cols == p, new_p[:, None], a)
    a = tl.where(cols == q, new_q[:, None], a)
    a = tl.where(rows == p, new_p[None, :], a)
    a = tl.where(rows == q, new_q[None, :], a)
    a = tl.where((rows == p) & (cols == p), app - t * apq, a)
    a = tl.where((rows == q) & (cols == q), aqq + t * apq, a)
    a = tl.where(((rows == p) & (cols == q)) | ((rows == q) & (cols == p)), 0.0, a)
    vp = tl.sum(tl.where(cols == p, v, 0.0), axis=1)
    vq = tl.sum(tl.where(cols == q, v, 0.0), axis=1)
    v = tl.where(cols == p, (c * vp - s * vq)[:, None], v)
    v = tl.where(cols == q, (s * vp + c * vq)[:, None], v)
    return a, v


@triton.jit
def _refine_rotation(r00, r01, r02, r10, r11, r12, r20, r21, r22, h00, h01, h02, h10, h11, h12, h20, h21, h22):
    """Refine the Horn result without subtracting nearly equal eigenvalues."""
    q00 = r00 * h00 + r10 * h10 + r20 * h20
    q01 = r00 * h01 + r10 * h11 + r20 * h21
    q02 = r00 * h02 + r10 * h12 + r20 * h22
    q10 = r01 * h00 + r11 * h10 + r21 * h20
    q11 = r01 * h01 + r11 * h11 + r21 * h21
    q12 = r01 * h02 + r11 * h12 + r21 * h22
    q20 = r02 * h00 + r12 * h10 + r22 * h20
    q21 = r02 * h01 + r12 * h11 + r22 * h21
    q22 = r02 * h02 + r12 * h12 + r22 * h22
    # The tangent Hessian is trace(Q) I - sym(Q). Sum the two other
    # diagonal entries directly so elongated clouds keep their small axis.
    a = q11 + q22
    b = -0.5 * (q01 + q10)
    c = -0.5 * (q02 + q20)
    d = q00 + q22
    e = -0.5 * (q12 + q21)
    f = q00 + q11
    tx = q21 - q12
    ty = q02 - q20
    tz = q10 - q01
    aa = d * f - e * e
    ab = c * e - b * f
    ac = b * e - c * d
    bb = a * f - c * c
    bc = b * c - a * e
    cc = a * d - b * b
    det = a * aa + b * ab + c * ac
    safe_det = tl.where(det > 1e-20, det, 1.0)
    x = (aa * tx + ab * ty + ac * tz) / safe_det
    y = (ab * tx + bb * ty + bc * tz) / safe_det
    z = (ac * tx + bc * ty + cc * tz) / safe_det
    # A nearly singular tangent Hessian cannot resolve a useful correction
    # in FP32, especially for almost isotropic reflected clouds.
    bnorm = tl.maximum(tl.abs(a) + tl.abs(b) + tl.abs(c), tl.abs(b) + tl.abs(d) + tl.abs(e))
    bnorm = tl.maximum(bnorm, tl.abs(c) + tl.abs(e) + tl.abs(f))
    adj_norm = tl.maximum(tl.abs(aa) + tl.abs(ab) + tl.abs(ac), tl.abs(ab) + tl.abs(bb) + tl.abs(bc))
    adj_norm = tl.maximum(adj_norm, tl.abs(ac) + tl.abs(bc) + tl.abs(cc))
    use = (a > 0.0) & (cc > 0.0) & (det > 1e-20) & (bnorm * adj_norm < 1e5 * det)
    use &= aa + bb + cc > 1e-5 * bnorm * bnorm
    use &= x * x + y * y + z * z < 0.01
    x = tl.where(use, x * 0.5, 0.0)
    y = tl.where(use, y * 0.5, 0.0)
    z = tl.where(use, z * 0.5, 0.0)
    # This is half the objective gain times the quaternion norm. Evaluate
    # it directly instead of subtracting nearly equal objective values.
    gain = x * tx + y * ty + z * tz
    gain -= a * x * x + d * y * y + f * z * z + 2.0 * (b * x * y + c * x * z + e * y * z)
    x = tl.where(gain >= 0.0, x, 0.0)
    y = tl.where(gain >= 0.0, y, 0.0)
    z = tl.where(gain >= 0.0, z, 0.0)
    norm = 1.0 + x * x + y * y + z * z
    d00 = (1.0 + x * x - y * y - z * z) / norm
    d01 = 2.0 * (x * y - z) / norm
    d02 = 2.0 * (x * z + y) / norm
    d10 = 2.0 * (x * y + z) / norm
    d11 = (1.0 - x * x + y * y - z * z) / norm
    d12 = 2.0 * (y * z - x) / norm
    d20 = 2.0 * (x * z - y) / norm
    d21 = 2.0 * (y * z + x) / norm
    d22 = (1.0 - x * x - y * y + z * z) / norm
    return (
        r00 * d00 + r01 * d10 + r02 * d20,
        r00 * d01 + r01 * d11 + r02 * d21,
        r00 * d02 + r01 * d12 + r02 * d22,
        r10 * d00 + r11 * d10 + r12 * d20,
        r10 * d01 + r11 * d11 + r12 * d21,
        r10 * d02 + r11 * d12 + r12 * d22,
        r20 * d00 + r21 * d10 + r22 * d20,
        r20 * d01 + r21 * d11 + r22 * d21,
        r20 * d02 + r21 * d12 + r22 * d22,
    )


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
    # Anchor at the largest effective weight: a negligible-weight outlier
    # must not set the scale of the FP32 centroid subtraction.
    best_weight = tl.zeros([BLOCK], tl.float32)
    best_index = tl.full([BLOCK], n_atoms, tl.int32)
    for start in range(0, n_atoms, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        valid = idx < n_atoms
        w = tl.load(weight_ptr + base + idx, valid, 0.0) * tl.load(mask_ptr + base + idx, valid, 0.0)
        better = tl.abs(w) > best_weight
        best_index = tl.where(better, idx, best_index)
        best_weight = tl.maximum(best_weight, tl.abs(w))
    first = tl.min(tl.where(best_weight == tl.max(best_weight), best_index, n_atoms))
    # Accumulate offsets so translated and constant clouds keep their centroid.
    to0 = tl.load(true_ptr + (base + first) * 3 + 0, first < n_atoms, 0.0)
    to1 = tl.load(true_ptr + (base + first) * 3 + 1, first < n_atoms, 0.0)
    to2 = tl.load(true_ptr + (base + first) * 3 + 2, first < n_atoms, 0.0)
    po0 = tl.load(pred_ptr + (base + first) * 3 + 0, first < n_atoms, 0.0)
    po1 = tl.load(pred_ptr + (base + first) * 3 + 1, first < n_atoms, 0.0)
    po2 = tl.load(pred_ptr + (base + first) * 3 + 2, first < n_atoms, 0.0)
    zeros = tl.zeros([BLOCK], tl.float32)
    sw = zeros
    st0 = zeros
    st1 = zeros
    st2 = zeros
    sp0 = zeros
    sp1 = zeros
    sp2 = zeros
    for start in range(0, n_atoms, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        valid = idx < n_atoms
        row = base + idx
        w = tl.load(weight_ptr + row, valid, 0.0) * tl.load(mask_ptr + row, valid, 0.0)
        t0 = tl.load(true_ptr + row * 3, valid, 0.0) - to0
        t1 = tl.load(true_ptr + row * 3 + 1, valid, 0.0) - to1
        t2 = tl.load(true_ptr + row * 3 + 2, valid, 0.0) - to2
        p0 = tl.load(pred_ptr + row * 3, valid, 0.0) - po0
        p1 = tl.load(pred_ptr + row * 3 + 1, valid, 0.0) - po1
        p2 = tl.load(pred_ptr + row * 3 + 2, valid, 0.0) - po2
        sw += w
        st0 += w * t0
        st1 += w * t1
        st2 += w * t2
        sp0 += w * p0
        sp1 += w * p1
        sp2 += w * p2
    # Reduce once after the atom loop, rather than once per tile.
    weight_sum = tl.sum(sw)
    mass = tl.maximum(weight_sum, _WEIGHT_EPS)
    origin_weight = tl.where(weight_sum >= _WEIGHT_EPS, 1.0, weight_sum / _WEIGHT_EPS)
    tc0 = tl.sum(st0) / mass + origin_weight * to0
    tc1 = tl.sum(st1) / mass + origin_weight * to1
    tc2 = tl.sum(st2) / mass + origin_weight * to2
    pc0 = tl.sum(sp0) / mass + origin_weight * po0
    pc1 = tl.sum(sp1) / mass + origin_weight * po1
    pc2 = tl.sum(sp2) / mass + origin_weight * po2
    # Center before multiplying: raw FP32 moments lose the covariance when
    # a cloud's translation is large compared with its spatial extent.
    h00 = zeros
    h01 = zeros
    h02 = zeros
    h10 = zeros
    h11 = zeros
    h12 = zeros
    h20 = zeros
    h21 = zeros
    h22 = zeros
    for start in range(0, n_atoms, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        valid = idx < n_atoms
        row = base + idx
        w = tl.load(weight_ptr + row, valid, 0.0) * tl.load(mask_ptr + row, valid, 0.0)
        t0 = tl.load(true_ptr + row * 3, valid, 0.0) - tc0
        t1 = tl.load(true_ptr + row * 3 + 1, valid, 0.0) - tc1
        t2 = tl.load(true_ptr + row * 3 + 2, valid, 0.0) - tc2
        p0 = (tl.load(pred_ptr + row * 3, valid, 0.0) - pc0) * w
        p1 = (tl.load(pred_ptr + row * 3 + 1, valid, 0.0) - pc1) * w
        p2 = (tl.load(pred_ptr + row * 3 + 2, valid, 0.0) - pc2) * w
        h00 += p0 * t0
        h01 += p0 * t1
        h02 += p0 * t2
        h10 += p1 * t0
        h11 += p1 * t1
        h12 += p1 * t2
        h20 += p2 * t0
        h21 += p2 * t1
        h22 += p2 * t2
    h00 = tl.sum(h00) / mass + _COV_EPS
    h01 = tl.sum(h01) / mass
    h02 = tl.sum(h02) / mass
    h10 = tl.sum(h10) / mass
    h11 = tl.sum(h11) / mass + _COV_EPS
    h12 = tl.sum(h12) / mass
    h20 = tl.sum(h20) / mass
    h21 = tl.sum(h21) / mass
    h22 = tl.sum(h22) / mass + _COV_EPS
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
    v = (rows == cols).to(tl.float32)
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
    # A small Newton correction recovers accuracy along weak axes that the
    # FP32 Horn eigenvalue gap cannot resolve. Normalize to avoid overflow.
    hscale = tl.maximum(tl.maximum(tl.abs(h00), tl.abs(h01)), tl.abs(h02))
    hscale = tl.maximum(hscale, tl.maximum(tl.maximum(tl.abs(h10), tl.abs(h11)), tl.abs(h12)))
    hscale = tl.maximum(hscale, tl.maximum(tl.maximum(tl.abs(h20), tl.abs(h21)), tl.abs(h22)))
    hscale = tl.maximum(hscale, 1e-20)
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = _refine_rotation(
        r00,
        r01,
        r02,
        r10,
        r11,
        r12,
        r20,
        r21,
        r22,
        h00 / hscale,
        h01 / hscale,
        h02 / hscale,
        h10 / hscale,
        h11 / hscale,
        h12 / hscale,
        h20 / hscale,
        h21 / hscale,
        h22 / hscale,
    )
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
            num_warps=_SOLVE_WARPS,
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
                values = (
                    true_coords.data_ptr(),
                    pred_coords.data_ptr(),
                    weights.data_ptr(),
                    mask.data_ptr(),
                    rotation.data_ptr(),
                    shift.data_ptr(),
                    n_atoms,
                )
                driver.launch_with(values, clouds)
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
