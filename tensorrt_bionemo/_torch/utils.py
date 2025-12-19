from typing import Callable

import torch.nn as nn


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
            # Uncomment to debug
            # print(f"loading for: {name}")
            try:
                if hasattr(module, 'load_weights'):
                    module.load_weights(weights=weights[name])
                else:
                    # Uncomment to debug
                    # print(f" use copy_ to load {name}")
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
