from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F


def masked_mean(mask, value, dim, eps=1e-4):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def pad_dim(t: torch.Tensor,
            dim: int,
            pad_len: int,
            value: float = 0) -> torch.Tensor:
    if pad_len == 0:
        return t

    total_dims = len(t.shape)
    padding = [0] * (2 * (total_dims - dim))
    padding[2 * (total_dims - 1 - dim) + 1] = pad_len
    return F.pad(t, tuple(padding), value=value)


def permute_final_dims(tensor: torch.Tensor, inds: list[int]):
    num_first_dims = len(tensor.shape) - len(inds)
    first_inds = list(range(num_first_dims))
    return tensor.permute(first_inds + [num_first_dims + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1, ))


def dist_one_hot(x: torch.Tensor, v_bins: torch.Tensor) -> torch.Tensor:
    """
    Computes one-hot encoding with absolute difference.
    Args:
        x:
            Input tensor of shape [*, N]
        v_bins:
            Bins of shape [no_bins]
    Returns:
        One-hot encoded tensor of shape [*, N, no_bins]
    """
    reshaped_bins = v_bins.view(((1, ) * len(x.shape)) + (len(v_bins), ))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return F.one_hot(am, num_classes=len(v_bins)).float()


def batched_gather(data: torch.Tensor,
                   inds: torch.Tensor,
                   dim: int = 0,
                   no_batch_dims: int = 0) -> torch.Tensor:
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1, ) * i), -1, *((1, ) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[tuple(ranges)]


def dict_multimap(fn: Callable, dicts: list[dict]) -> dict:
    """
    Applies a function to a list of dictionaries, element-wise.
    """
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


# With tree_map, a poor man's JAX tree_map
def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        # raise ValueError(f"Tree of type {type(tree)} not supported")
        pass


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)
