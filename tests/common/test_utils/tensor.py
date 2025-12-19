import torch


def mismatch_percentage(a: torch.Tensor,
                        b: torch.Tensor,
                        *,
                        atol: float = 1e-8,
                        rtol: float = 1e-5) -> float:
    if a.shape != b.shape:
        raise ValueError("Shape mismatch")

    diff = torch.abs(a - b)
    tol = atol + rtol * torch.abs(b)

    mismatches = diff > tol
    num_mismatch = mismatches.sum().item()
    total = mismatches.numel()

    return 100.0 * num_mismatch / total
