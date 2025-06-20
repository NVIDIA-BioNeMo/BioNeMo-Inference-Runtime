# Boltz-2

Boltz-2 repo: https://github.com/jwohlwend/boltz/tree/main

## Overview

## Build engines

The building progress is similar to the Boltz-1 model in the Boltz1's example folder. Please see: [`examples/boltz1`](examples/boltz1)

## Generate test samples

```
$ python ${_boltz2_dir_}/src/boltz/main.py predict --cache _cache_dir --use_msa_server ${_boltz2_dir_}/examples/prot.yaml
```

## Patching Dropout

```bash
$ vi ${_boltz2_dir_}/model/layers/dropout.py
```

Edit the file to:

```python
import torch
from torch import Tensor


def get_dropout_mask(
    dropout: float,
    z: Tensor,
    training: bool,
    columnwise: bool = False,
) -> Tensor:
    """Get the dropout mask.

    Parameters
    ----------
    dropout : float
        The dropout rate
    z : torch.Tensor
        The tensor to apply dropout to
    training : bool
        Whether the model is in training mode
    columnwise : bool, optional
        Whether to apply dropout columnwise

    Returns
    -------
    torch.Tensor
        The dropout mask

    """
    if not training:
        return 1.
    dropout = dropout * training
    v = z[:, 0:1, :, 0:1] if columnwise else z[:, :, 0:1, 0:1]
    d = torch.rand_like(v) > dropout
    d = d * 1.0 / (1.0 - dropout)
    return d

```
