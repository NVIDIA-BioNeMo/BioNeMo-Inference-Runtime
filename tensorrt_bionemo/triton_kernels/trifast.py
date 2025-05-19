import triton
import triton.language as tl
import triton.testing


@triton.jit
def trifast_attention_kernel_fwd(
        o_ptr, stride_oh, stride_om, stride_on, stride_od, lse_ptr, stride_lseh,
        stride_lsem, stride_lsen, q_ptr, stride_qh, stride_qm, stride_qn,
        stride_qd, k_ptr, stride_kh, stride_km, stride_kn, stride_kd, v_ptr,
        stride_vh, stride_vm, stride_vn, stride_vd, b_ptr, stride_bh, stride_bm,
        stride_bn, mask_ptr, stride_maskh, stride_maskm, stride_maskn, sm_scale,
        neg_inf, batch_size, si, seq_len, heads, DIM: tl.constexpr,
        BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr, CLOSEST_N: tl.constexpr):
    """
    This code from trifast repository: https://github.com/latkins/trifast
    But modified to be enable for building with TensorRT plugins and also used in Torch.
    Disable auto-tunning for TRT and moves bs, si, n and h to be arguments.
    """

    input_dtype = q_ptr.dtype.element_ty
    pid_j = tl.program_id(0)  # Parallelize over chunks of j
    pid_i = tl.program_id(1)  # Parallelize along i
    pid_h = tl.program_id(2)  # Parallelize along h

    inv_ln2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    ln2: tl.constexpr = 0.6931471824645996  # = ln(2)

    # One mask per batch item, not repeated per head.
    mask_start_h = pid_h // heads
    start_h = pid_h
    start_i = pid_i
    start_j = pid_j * BLOCK_J
    start_k = 0  # we iterate over k, so each pid starts at 0

    # Indices of blocks.
    k_idxs = tl.arange(0, BLOCK_K)
    j_idxs = tl.arange(0, BLOCK_J) + start_j
    d_idxs = tl.arange(0, DIM)

    # Set up ptrs to blocks.
    base_q_ptr = q_ptr + (start_h * stride_qh) + (start_i * stride_qm)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (
        d_idxs[None, :] * stride_qd)  # [j,d]

    base_kt_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
    kt_ptrs = base_kt_ptr + (d_idxs[:, None]) * stride_kd + (
        k_idxs[None, :] * stride_kn)  # [d,k]

    base_b_ptr = b_ptr + (start_h * stride_bh)
    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (
        k_idxs[None, :] * stride_bn)  # [j,k]

    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    v_ptrs = base_v_ptr + (k_idxs[:, None] * stride_vn) + (
        d_idxs[None, :] * stride_vd)  # [k,d]

    base_lse_ptr = lse_ptr + (start_h * stride_lseh) + (start_i * stride_lsem)
    lse_ptrs = base_lse_ptr + (j_idxs * stride_lsen)  # [j]

    base_mask_ptr = mask_ptr + (mask_start_h * stride_maskh)
    mask_ptrs = base_mask_ptr + (start_i * stride_maskm) + (
        k_idxs * stride_maskn)  # [k]

    base_o_ptr = o_ptr + (start_h * stride_oh) + (start_i * stride_om)
    o_ptrs = base_o_ptr + (j_idxs[:, None] * stride_on) + (
        d_idxs[None, :] * stride_od)  # [j,d]

    scores_max = tl.full([BLOCK_J], value=-float("inf"), dtype=tl.float32)
    sm_denom = tl.full([BLOCK_J], value=0, dtype=tl.float32)
    acc = tl.full([BLOCK_J, DIM], value=0, dtype=tl.float32)

    mask_j = j_idxs < seq_len

    q_block = tl.load(q_ptrs, mask_j[:, None])  # [j,d]
    q_block = q_block * tl.full(
        [1], value=sm_scale, dtype=q_block.type.element_ty)

    for start_k in tl.range(0, seq_len, BLOCK_K):
        start_k = tl.multiple_of(start_k, BLOCK_K)
        mask_k = (k_idxs + start_k) < seq_len

        kt_block = tl.load(kt_ptrs, mask_k[None, :])  # [d,k]
        b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :])  # [j,k]
        m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg")  # [k]

        scores = b_block.to(tl.float32)
        scores = tl.dot(q_block, kt_block, scores,
                        input_precision="ieee")  # [j,k]
        scores *= inv_ln2  # 1.0 / ln(2), [j,k]

        # we want to make scores -inf at mask locations
        scores = tl.where(m_block[None, :], neg_inf, scores)  # [j,k]
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)

        # Iterative softmax
        block_max = tl.maximum(scores_max, tl.max(scores, 1))  # [j]
        scores = scores - block_max[:, None]  # [j,k]
        exp_scores = tl.math.exp2(scores)  # [j,k]

        summed_exp_scores = tl.sum(exp_scores, 1)  # [j]
        exp_scale = tl.math.exp2(scores_max - block_max)  # [j]

        sm_denom = sm_denom * exp_scale + summed_exp_scores  # [j]

        acc = acc * exp_scale[:, None]  # [j,d]
        v_block = tl.load(v_ptrs, mask_k[:, None])  # [k,d]
        exp_scores = exp_scores.to(input_dtype)  # [j,k]

        acc = tl.dot(exp_scores, v_block, acc, input_precision="ieee")  # [j,d]

        scores_max = block_max

        # Advance to next block along the k dimension.
        kt_ptrs += BLOCK_K * stride_kn
        v_ptrs += BLOCK_K * stride_vn
        b_ptrs += BLOCK_K * stride_bn
        mask_ptrs += BLOCK_K * stride_maskn

    normalize = acc / sm_denom[:, None]
    final_output = normalize.to(input_dtype)
    tl.store(o_ptrs, final_output, mask=mask_j[:, None])

    lse = (scores_max * ln2) + tl.log(sm_denom)

    tl.store(lse_ptrs, lse, mask=mask_j)


def create_autotuner() -> triton.runtime.Autotuner:
    configs = [
        triton.Config(kwargs={
            "BLOCK_J": 16,
            "BLOCK_K": 32
        },
                      num_warps=1,
                      num_stages=2),
        triton.Config(kwargs={
            "BLOCK_J": 32,
            "BLOCK_K": 16
        },
                      num_warps=1,
                      num_stages=2),
        triton.Config(kwargs={
            "BLOCK_J": 64,
            "BLOCK_K": 32
        },
                      num_warps=4,
                      num_stages=2),
        triton.Config(kwargs={
            "BLOCK_J": 64,
            "BLOCK_K": 16
        },
                      num_warps=2,
                      num_stages=3),
        triton.Config(kwargs={
            "BLOCK_J": 32,
            "BLOCK_K": 32
        },
                      num_warps=1,
                      num_stages=1),
        triton.Config(kwargs={
            "BLOCK_J": 128,
            "BLOCK_K": 16
        },
                      num_warps=2,
                      num_stages=2),
        triton.Config(kwargs={
            "BLOCK_J": 128,
            "BLOCK_K": 32
        },
                      num_warps=2,
                      num_stages=2),
        triton.Config(kwargs={
            "BLOCK_J": 32,
            "BLOCK_K": 64
        },
                      num_warps=2,
                      num_stages=2),
        triton.Config(kwargs={
            "BLOCK_J": 64,
            "BLOCK_K": 32
        },
                      num_warps=2,
                      num_stages=3),
        triton.Config(kwargs={
            "BLOCK_J": 32,
            "BLOCK_K": 16
        },
                      num_warps=1,
                      num_stages=4),
        triton.Config({
            "BLOCK_J": 32,
            "BLOCK_K": 32
        }, num_warps=1, num_stages=5),
        triton.Config({
            "BLOCK_J": 64,
            "BLOCK_K": 32
        }, num_warps=4, num_stages=2),
        triton.Config({
            "BLOCK_J": 128,
            "BLOCK_K": 32
        },
                      num_warps=4,
                      num_stages=2),
        triton.Config({
            "BLOCK_J": 32,
            "BLOCK_K": 64
        }, num_warps=4, num_stages=3),
        triton.Config({
            "BLOCK_J": 64,
            "BLOCK_K": 16
        }, num_warps=2, num_stages=3),
        triton.Config({
            "BLOCK_J": 128,
            "BLOCK_K": 16
        },
                      num_warps=2,
                      num_stages=2),
        triton.Config({
            "BLOCK_J": 32,
            "BLOCK_K": 16
        }, num_warps=2, num_stages=4),
        triton.Config({
            "BLOCK_J": 16,
            "BLOCK_K": 32
        }, num_warps=4, num_stages=2),
        triton.Config({
            "BLOCK_J": 32,
            "BLOCK_K": 16
        }, num_warps=4, num_stages=1),
        triton.Config({
            "BLOCK_J": 16,
            "BLOCK_K": 64
        }, num_warps=4, num_stages=1),
        triton.Config({
            "BLOCK_J": 128,
            "BLOCK_K": 16
        },
                      num_warps=4,
                      num_stages=2),
        triton.Config({
            "BLOCK_J": 32,
            "BLOCK_K": 32
        }, num_warps=8, num_stages=1),
        triton.Config({
            "BLOCK_J": 64,
            "BLOCK_K": 32
        }, num_warps=8, num_stages=1),
        triton.Config({
            "BLOCK_J": 32,
            "BLOCK_K": 16
        }, num_warps=2, num_stages=5),
        triton.Config({
            "BLOCK_J": 16,
            "BLOCK_K": 16
        }, num_warps=8, num_stages=1),
        triton.Config({
            "BLOCK_J": 32,
            "BLOCK_K": 64
        }, num_warps=8, num_stages=1),
        triton.Config({
            "BLOCK_J": 64,
            "BLOCK_K": 64
        }, num_warps=4, num_stages=1),
        triton.Config({
            "BLOCK_J": 128,
            "BLOCK_K": 32
        },
                      num_warps=8,
                      num_stages=1),
        triton.Config({
            "BLOCK_J": 16,
            "BLOCK_K": 32
        }, num_warps=8, num_stages=2),
        triton.Config({
            "BLOCK_J": 32,
            "BLOCK_K": 128
        },
                      num_warps=4,
                      num_stages=2),
    ]
    key = ["CLOSEST_N"]
    fn = trifast_attention_kernel_fwd
    return triton.runtime.Autotuner(fn,
                                    fn.arg_names,
                                    configs=configs,
                                    key=key,
                                    reset_to_zero=None,
                                    restore_value=None)
