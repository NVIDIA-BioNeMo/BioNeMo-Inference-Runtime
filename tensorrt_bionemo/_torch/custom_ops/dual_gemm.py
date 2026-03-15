from typing import Callable, Optional

import torch
from cuequivariance_ops_torch.gated_gemm_torch import (
    fused_sigmoid_gated_dual_gemm, fused_sigmoid_gated_dual_gemm_dual_x)
from tensorrt_llm_lite._utils import get_sm_version


def _invoke_vanilla_dual_gemm_x_x(x: torch.Tensor,
                                  w1: torch.Tensor,
                                  w2: torch.Tensor,
                                  bias1: Optional[torch.Tensor] = None,
                                  bias2: Optional[torch.Tensor] = None,
                                  mask: Optional[torch.Tensor] = None,
                                  transpose_out: bool = False) -> torch.Tensor:
    if bias1 is not None and bias2 is not None:
        ret = (x @ w1.T + bias1).sigmoid() * (x @ w2.T + bias2)
    else:
        ret = (x @ w1.T).sigmoid() * (x @ w2.T)
    if mask is not None:
        ret = ret * mask.unsqueeze(-1)
    if transpose_out:
        ret = ret.moveaxis(-1,
                           0)  # move the last dimension to the first dimension
    return ret.contiguous()


def _invoke_vanilla_dual_gemm_x0_x1(
        x1: torch.Tensor,
        x2: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        bias1: Optional[torch.Tensor] = None,
        bias2: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        transpose_out: bool = False) -> torch.Tensor:
    if bias1 is not None and bias2 is not None:
        ret = (x1 @ w1.T + bias1).sigmoid() * (x2 @ w2.T + bias2)
    else:
        ret = (x1 @ w1.T).sigmoid() * (x2 @ w2.T)
    if mask is not None:
        ret = ret * mask.unsqueeze(-1)
    if transpose_out:
        ret = ret.moveaxis(-1,
                           0)  # move the last dimension to the first dimension
    return ret.contiguous()


def _invoke_cutlass_dual_gemm_x_x(x: torch.Tensor,
                                  w1: torch.Tensor,
                                  w2: torch.Tensor,
                                  bias1: Optional[torch.Tensor] = None,
                                  bias2: Optional[torch.Tensor] = None,
                                  mask: Optional[torch.Tensor] = None,
                                  transpose_out: bool = False) -> torch.Tensor:
    # TODO: transpose_out is accepted but not yet forwarded to the CUTLASS
    # kernel.  When the kernel supports it, pass it through instead of
    # ignoring it here.
    x = x.contiguous()
    mask = mask.contiguous()
    x = torch.ops._C.x_x_dual_gemm(x, w1, w2, bias1, bias2, mask)
    return x


def _invoke_cutlass_dual_gemm_x0_x1(
        x1: torch.Tensor,
        x2: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        bias1: Optional[torch.Tensor] = None,
        bias2: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        transpose_out: bool = False) -> torch.Tensor:
    # TODO: transpose_out is accepted but not yet forwarded to the CUTLASS
    # kernel.  When the kernel supports it, pass it through instead of
    # ignoring it here.
    # mask is not support for cutlass dual gemm with x0_x1
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    x = torch.ops._C.x0_x1_dual_gemm(x1, x2, w1, w2, bias1, bias2)
    return x


def _invoke_cuequiv_dual_gemm_x_x(x: torch.Tensor,
                                  w1: torch.Tensor,
                                  w2: torch.Tensor,
                                  bias1: Optional[torch.Tensor] = None,
                                  bias2: Optional[torch.Tensor] = None,
                                  mask: Optional[torch.Tensor] = None,
                                  transpose_out: bool = False) -> torch.Tensor:
    x = fused_sigmoid_gated_dual_gemm(x,
                                      w1,
                                      w2,
                                      mask,
                                      transpose_out=transpose_out,
                                      b1=bias1,
                                      b2=bias2,
                                      precision=-1)
    return x


def _invoke_cuequiv_dual_gemm_x0_x1(
        x1: torch.Tensor,
        x2: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        bias1: Optional[torch.Tensor] = None,
        bias2: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        transpose_out: bool = False) -> torch.Tensor:
    x = fused_sigmoid_gated_dual_gemm_dual_x(x1,
                                             x2,
                                             w1,
                                             w2,
                                             mask,
                                             transpose_out=transpose_out,
                                             b1=bias1,
                                             b2=bias2,
                                             precision=-1)
    return x


def get_dual_gemm_op(dtype: torch.dtype,
                     transpose_out=False,
                     dual_gemm_type: str = "x_x",
                     N: int = 128,
                     K: int = 128) -> Callable:
    sm = get_sm_version()

    if not dual_gemm_type in ["x_x", "x0_x1"]:
        raise ValueError(f"Invalid dual_gemm_type: {dual_gemm_type}")

    if N not in [128, 256] or K not in [128] or dtype not in [
            torch.float16, torch.bfloat16
    ]:
        return _invoke_vanilla_dual_gemm_x_x if dual_gemm_type == "x_x" else _invoke_vanilla_dual_gemm_x0_x1

    if transpose_out:
        if dual_gemm_type == "x_x":
            return _invoke_cuequiv_dual_gemm_x_x
        elif dual_gemm_type == "x0_x1":
            return _invoke_cuequiv_dual_gemm_x0_x1

    if dual_gemm_type == "x_x":
        if sm in ["80", "86", "89", "90"]:
            return _invoke_cutlass_dual_gemm_x_x
        else:
            return _invoke_cuequiv_dual_gemm_x_x

    if dual_gemm_type == "x0_x1":
        if sm in ["80", "86", "89", "90"]:
            return _invoke_cutlass_dual_gemm_x0_x1
        else:
            return _invoke_cuequiv_dual_gemm_x0_x1

    return _invoke_vanilla_dual_gemm_x_x if dual_gemm_type == "x_x" else _invoke_vanilla_dual_gemm_x0_x1
