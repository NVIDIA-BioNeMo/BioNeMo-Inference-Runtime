import torch


def permute_final_dims(tensor: torch.Tensor, inds: list[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    all_atom_mask = all_atom_mask.unsqueeze(-1)
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(eps + torch.sum(
        (all_atom_positions[..., None, :] -
         all_atom_positions[..., None, :, :])**2,
        dim=-1,
    ))

    dmat_pred = torch.sqrt(eps + torch.sum(
        (all_atom_pred_pos[..., None, :] -
         all_atom_pred_pos[..., None, :, :])**2,
        dim=-1,
    ))
    dists_to_score = ((dmat_true < cutoff) * all_atom_mask *
                      permute_final_dims(all_atom_mask, (1, 0)) *
                      (1.0 - torch.eye(n, device=all_atom_mask.device)))

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = ((dist_l1 < 0.5).type(dist_l1.dtype) +
             (dist_l1 < 1.0).type(dist_l1.dtype) +
             (dist_l1 < 2.0).type(dist_l1.dtype) +
             (dist_l1 < 4.0).type(dist_l1.dtype))
    score = score * 0.25

    dims = (-1, ) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score


def kabsch_torch(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.
    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=0)
    centroid_Q = torch.mean(Q, dim=0)

    # Optimal translation
    centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(0, 1), q)

    # SVD
    U, S, Vt = torch.linalg.svd(H)

    R = torch.matmul(U, Vt)
    # Validate right-handed coordinate system
    if torch.det(R) < 0.0:
        R[:, -1] = R[:, -1] * -1.0

    rmsd = torch.sqrt(
        torch.sum(torch.square(torch.matmul(p, R) - q)) / P.shape[0])

    return rmsd
