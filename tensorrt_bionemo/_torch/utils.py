from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorrt_llm.llmapi.utils import print_colored_debug


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


def recursive_calling_load_weights(module: nn.Module,
                                   weights: dict,
                                   filter_func: Callable = None) -> set[str]:
    """
    DFS calling load_weights for the module.
    Args:
        module: The module to load weights for.
        weights: The weights to load.
        filter_func: The function to filter the modules to load weights for.
    Returns:
        The set of loaded weights.
    """
    loaded_weight = set()

    for name, module in module.named_modules():
        if filter_func is not None and filter_func(name, module):
            continue
        if len(module._parameters) > 0:
            print_colored_debug(f"loading for: {name}")
            try:
                if hasattr(module, 'load_weights'):
                    module.load_weights(weights=weights[name])
                else:
                    print_colored_debug(f" use copy_ to load {name}")
                    module_weights = weights[name][0]
                    for n, p in module._parameters.items():
                        if p is not None:
                            weight = module_weights[n][:]
                            if p.dtype != weight.dtype:
                                weight = weight.to(p.dtype)
                            p.data.copy_(weight)

            except Exception as e:
                print(name)
                raise e
        loaded_weight.add(name)
    return loaded_weight
